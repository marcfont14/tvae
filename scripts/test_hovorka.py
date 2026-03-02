"""
test_hovorka.py
===============
Compares odeint vs Euler integration for Hovorka PI and RA models
on 10 patients. If signals are equivalent, we use Euler for the full
preprocessing pipeline.

Usage:
    python scripts/test_hovorka.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pandas as pd
from scipy.integrate import odeint
from src.settings import Settings

cfg = Settings()
p_pi = cfg.preprocessing.params_pi
p_ra = cfg.preprocessing.params_ra


# ── odeint implementation ─────────────────────────────────────────────────────

def pi_odeint(bolus: np.ndarray, basal: np.ndarray) -> np.ndarray:
    def system(y, t, params, u):
        s1, s2, ifa = y
        kdia = 1 / params.tmaxI
        u_t = u[int(t / params.dt)] if int(t / params.dt) < len(u) else 0.0
        return [u_t - kdia * s1,
                kdia * (s1 - s2),
                (s2 / (params.tmaxI * params.VI)) - params.Ke * ifa]

    n = len(bolus)
    t = np.linspace(0, (n-1) * p_pi.dt, n)
    sol_b = odeint(system, [0, 0, 0], t, args=(p_pi, bolus / p_pi.dt), hmax=p_pi.dt)
    sol_s = odeint(system, [0, 0, 0], t, args=(p_pi, basal / p_pi.dt), hmax=p_pi.dt)
    return p_pi.pi_sign * (sol_b[:, 2] + sol_s[:, 2])


def ra_odeint(carbs: np.ndarray) -> np.ndarray:
    def system(y, t, params, u):
        d1, d2 = y
        u_t = u[int(t / params.dt)] if int(t / params.dt) < len(u) else 0.0
        return [params.A_G * u_t - (1 / params.tau_D) * d1,
                (1 / params.tau_D) * d1 - (1 / params.tau_D) * d2]

    n = len(carbs)
    t = np.linspace(0, (n-1) * p_pi.dt, n)
    sol = odeint(system, [0, 0], t, args=(p_ra, carbs), hmax=p_ra.dt)
    return sol[:, 1] / p_ra.tau_D


# ── Euler implementation ──────────────────────────────────────────────────────

def pi_euler(bolus: np.ndarray, basal: np.ndarray) -> np.ndarray:
    n = len(bolus)
    dt = p_pi.dt
    kdia = 1 / p_pi.tmaxI

    # Bolus component
    s1, s2, ifa_b = 0.0, 0.0, 0.0
    ifa_bolus = np.zeros(n)
    for i in range(n):
        u = bolus[i] / dt
        s1_new  = s1  + dt * (u - kdia * s1)
        s2_new  = s2  + dt * (kdia * (s1 - s2))
        ifa_new = ifa_b + dt * ((s2 / (p_pi.tmaxI * p_pi.VI)) - p_pi.Ke * ifa_b)
        s1, s2, ifa_b = s1_new, s2_new, ifa_new
        ifa_bolus[i] = ifa_b

    # Basal component
    s1, s2, ifa_s = 0.0, 0.0, 0.0
    ifa_basal = np.zeros(n)
    for i in range(n):
        u = basal[i] / dt
        s1_new  = s1  + dt * (u - kdia * s1)
        s2_new  = s2  + dt * (kdia * (s1 - s2))
        ifa_new = ifa_s + dt * ((s2 / (p_pi.tmaxI * p_pi.VI)) - p_pi.Ke * ifa_s)
        s1, s2, ifa_s = s1_new, s2_new, ifa_new
        ifa_basal[i] = ifa_s

    return p_pi.pi_sign * (ifa_bolus + ifa_basal)


def ra_euler(carbs: np.ndarray) -> np.ndarray:
    n = len(carbs)
    dt = p_ra.dt
    d1, d2 = 0.0, 0.0
    ra = np.zeros(n)
    for i in range(n):
        d1_new = d1 + dt * (p_ra.A_G * carbs[i] - (1 / p_ra.tau_D) * d1)
        d2_new = d2 + dt * ((1 / p_ra.tau_D) * d1 - (1 / p_ra.tau_D) * d2)
        d1, d2 = d1_new, d2_new
        ra[i] = d2 / p_ra.tau_D
    return ra


# ── Load 10 patients ──────────────────────────────────────────────────────────

print("Loading 10 patients...")
pf = pq.ParquetFile(cfg.paths.combined_parquet)
patient_data = {}

for batch in pf.iter_batches(batch_size=500_000,
                              columns=['id', 'bolus', 'basal', 'carbs']):
    df = batch.to_pandas()
    for pid, grp in df.groupby('id'):
        if pid not in patient_data:
            patient_data[pid] = []
        patient_data[pid].append(grp)
        if len(patient_data) >= 10:
            break
    if len(patient_data) >= 10:
        break

print(f"Loaded {len(patient_data)} patients")

# ── Compare ───────────────────────────────────────────────────────────────────

pi_maes, ra_maes = [], []
pi_maxes, ra_maxes = [], []

fig, axes = plt.subplots(10, 4, figsize=(20, 40))
fig.suptitle('odeint vs Euler — Hovorka PI and RA (10 patients)', fontsize=14)

for row, (pid, chunks) in enumerate(patient_data.items()):
    df = pd.concat(chunks).reset_index(drop=True)
    bolus = df['bolus'].fillna(0).values.astype(np.float64)
    basal = df['basal'].fillna(0).values.astype(np.float64)
    carbs = df['carbs'].fillna(0).values.astype(np.float64)

    # Compute both
    pi_od = pi_odeint(bolus, basal)
    pi_eu = pi_euler(bolus, basal)
    ra_od = ra_odeint(carbs)
    ra_eu = ra_euler(carbs)

    # Metrics
    pi_mae = np.mean(np.abs(pi_od - pi_eu))
    ra_mae = np.mean(np.abs(ra_od - ra_eu))
    pi_max = np.max(np.abs(pi_od - pi_eu))
    ra_max = np.max(np.abs(ra_od - ra_eu))
    pi_maes.append(pi_mae)
    ra_maes.append(ra_mae)
    pi_maxes.append(pi_max)
    ra_maxes.append(ra_max)

    # Plot first 2000 steps (~7 days)
    n = min(2000, len(pi_od))
    t = np.arange(n) * 5 / 60  # hours

    axes[row, 0].plot(t, pi_od[:n], label='odeint', alpha=0.8)
    axes[row, 0].plot(t, pi_eu[:n], label='Euler',  alpha=0.8, linestyle='--')
    axes[row, 0].set_title(f'Patient {pid} — PI')
    axes[row, 0].legend(fontsize=7)

    axes[row, 1].plot(t, np.abs(pi_od[:n] - pi_eu[:n]), color='red')
    axes[row, 1].set_title(f'PI absolute error (MAE={pi_mae:.2e})')

    axes[row, 2].plot(t, ra_od[:n], label='odeint', alpha=0.8)
    axes[row, 2].plot(t, ra_eu[:n], label='Euler',  alpha=0.8, linestyle='--')
    axes[row, 2].set_title(f'Patient {pid} — RA')
    axes[row, 2].legend(fontsize=7)

    axes[row, 3].plot(t, np.abs(ra_od[:n] - ra_eu[:n]), color='red')
    axes[row, 3].set_title(f'RA absolute error (MAE={ra_mae:.2e})')

plt.tight_layout()
plt.savefig('plots/hovorka_odeint_vs_euler.png', dpi=100, bbox_inches='tight')
plt.close()

print("\n── Results ──────────────────────────────────────────")
print(f"PI  — mean MAE: {np.mean(pi_maes):.2e}  |  max error: {np.max(pi_maxes):.2e}")
print(f"RA  — mean MAE: {np.mean(ra_maes):.2e}  |  max error: {np.max(ra_maxes):.2e}")
print(f"\nPlot saved to plots/hovorka_odeint_vs_euler.png")


import pyarrow.parquet as pq
import pandas as pd

pf = pq.ParquetFile('data/interim/combined_filtered.parquet')
patient_data = {}

for batch in pf.iter_batches(batch_size=500_000, columns=['id', 'bolus', 'basal', 'carbs']):
    df = batch.to_pandas()
    for pid, grp in df.groupby('id'):
        if pid not in patient_data:
            patient_data[pid] = []
        patient_data[pid].append(grp)
        if len(patient_data) >= 10:
            break
    if len(patient_data) >= 10:
        break

for pid in ['1', '2']:
    df = pd.concat(patient_data[pid]).reset_index(drop=True)
    bolus = df['bolus'].fillna(0)
    basal = df['basal'].fillna(0)
    print(f"\nPatient {pid}:")
    print(f"  bolus max: {bolus.max():.3f}, mean: {bolus[bolus>0].mean():.3f}")
    print(f"  basal max: {basal.max():.3f}, mean: {basal[basal>0].mean():.3f}")
    print(f"  n rows: {len(df)}")


print(f"Patient {pid} — PI NaN: {np.isnan(pi_od).sum()}, RA NaN: {np.isnan(ra_od).sum()}")
print(f"Patient {pid} — PI Euler NaN: {np.isnan(pi_eu).sum()}, RA Euler NaN: {np.isnan(ra_eu).sum()}")