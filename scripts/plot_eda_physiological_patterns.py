"""EDA figure 4: population circadian CGM / PI / RA profiles.

Windows start at different times of day (stride=6h), so averaging by window
position would smear the time axis. Instead we use the hour_sin/cos features
(cols 3,4) to recover each timestep's actual 5-min slot in the day (0–287),
then average CGM/PI/RA within each slot across all windows.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = '/mnt/workspace/tvae/data/processed/adults_global_norm'
OUT_PATH = '/mnt/workspace/tvae/results/eda/physiological_patterns.png'

CGM_MEAN, CGM_STD = 144.40, 57.11

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10, 'axes.labelsize': 10,
    'axes.titlesize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 8.5, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'axes.edgecolor': 'black',
    'axes.linewidth': 0.8, 'axes.grid': True, 'grid.color': '#cccccc',
    'grid.linewidth': 0.5, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'lines.linewidth': 1.6,
})
C_BLUE  = '#0072BD'
C_RED   = '#D95319'
C_GREEN = '#77AC30'

N_SLOTS = 288  # 288 × 5 min = 24 h

# Accumulators: sum and count per time slot
cgm_sum  = np.zeros(N_SLOTS)
cgm_sum2 = np.zeros(N_SLOTS)
pi_sum   = np.zeros(N_SLOTS)
pi_sum2  = np.zeros(N_SLOTS)
ra_sum   = np.zeros(N_SLOTS)
ra_sum2  = np.zeros(N_SLOTS)
slot_n   = np.zeros(N_SLOTS)

print('Accumulating circadian profiles (time-of-day aligned)...')
files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.npz'))
for i, fn in enumerate(files):
    if i % 200 == 0:
        print(f'  {i}/{len(files)}')
    d  = np.load(os.path.join(DATA_DIR, fn))
    ws = d['windows'].astype(np.float32)  # (N, 288, 11)

    # Recover actual 5-min day slot from hour_sin (col 3) and hour_cos (col 4)
    # sin=sin(2π*slot/288), cos=cos(2π*slot/288) → slot = round(atan2(sin,cos)*288/(2π)) % 288
    h_sin = ws[:, :, 3]  # (N, 288)
    h_cos = ws[:, :, 4]  # (N, 288)
    angles = np.arctan2(h_sin, h_cos)            # in (-π, π]
    slots  = np.round(angles * N_SLOTS / (2 * np.pi)).astype(int) % N_SLOTS  # (N, 288)

    cgm = ws[:, :, 0]  # z-score
    pi  = ws[:, :, 1]
    ra  = ws[:, :, 2]

    # Accumulate into slot bins using np.add.at
    np.add.at(slot_n,   slots.ravel(), 1)
    np.add.at(cgm_sum,  slots.ravel(), cgm.ravel())
    np.add.at(cgm_sum2, slots.ravel(), cgm.ravel()**2)
    np.add.at(pi_sum,   slots.ravel(), pi.ravel())
    np.add.at(pi_sum2,  slots.ravel(), pi.ravel()**2)
    np.add.at(ra_sum,   slots.ravel(), ra.ravel())
    np.add.at(ra_sum2,  slots.ravel(), ra.ravel()**2)

print('Computing mean and SD per slot...')
n = np.maximum(slot_n, 1)
cgm_mean_z = cgm_sum / n
cgm_std_z  = np.sqrt(np.maximum(cgm_sum2 / n - cgm_mean_z**2, 0))
pi_mean    = pi_sum / n
pi_std     = np.sqrt(np.maximum(pi_sum2 / n - pi_mean**2, 0))
ra_mean    = ra_sum / n
ra_std     = np.sqrt(np.maximum(ra_sum2 / n - ra_mean**2, 0))

cgm_mean_mg = cgm_mean_z * CGM_STD + CGM_MEAN
cgm_std_mg  = cgm_std_z  * CGM_STD

# Time axis (hours)
t_hours = np.arange(N_SLOTS) * 24 / N_SLOTS

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# Panel A — CGM
ax = axes[0]
ax.fill_between(t_hours, cgm_mean_mg - cgm_std_mg, cgm_mean_mg + cgm_std_mg,
                color=C_BLUE, alpha=0.15, label='±1 SD')
ax.plot(t_hours, cgm_mean_mg, color=C_BLUE, linewidth=1.6, label='Mean')
ax.axhline(70,  color=C_RED, linestyle=':', linewidth=1.0)
ax.axhline(180, color=C_RED, linestyle=':', linewidth=1.0, label='70 / 180 mg/dL')
ax.set_xlabel('Time of day (h)')
ax.set_ylabel('CGM (mg/dL)')
ax.set_title('Population mean CGM')
ax.set_xlim(0, 24)
ax.set_xticks([0, 6, 12, 18, 24])
ax.legend(framealpha=0.9, edgecolor='#cccccc')

# Panel B — PI
ax = axes[1]
ax.fill_between(t_hours, pi_mean - pi_std, pi_mean + pi_std, color=C_RED, alpha=0.15, label='±1 SD')
ax.plot(t_hours, pi_mean, color=C_RED, linewidth=1.6, label='Mean')
ax.set_xlabel('Time of day (h)')
ax.set_ylabel('Plasma insulin (z-score)')
ax.set_title('Population mean PI')
ax.set_xlim(0, 24)
ax.set_xticks([0, 6, 12, 18, 24])
ax.legend(framealpha=0.9, edgecolor='#cccccc')

# Panel C — RA
ax = axes[2]
ax.fill_between(t_hours, ra_mean - ra_std, ra_mean + ra_std, color=C_GREEN, alpha=0.15, label='±1 SD')
ax.plot(t_hours, ra_mean, color=C_GREEN, linewidth=1.6, label='Mean')
ax.set_xlabel('Time of day (h)')
ax.set_ylabel('Carb absorption RA (z-score)')
ax.set_title('Population mean RA')
ax.set_xlim(0, 24)
ax.set_xticks([0, 6, 12, 18, 24])
ax.legend(framealpha=0.9, edgecolor='#cccccc')

fig.tight_layout(pad=1.5, w_pad=2.5)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
print(f'Saved {OUT_PATH}')
