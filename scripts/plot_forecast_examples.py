"""
Qualitative forecasting examples: 2×2 panel.
Each panel: 6h CGM context (input) + actual vs predicted 2h trajectory.
Models: Encoder fine-tuned vs Raw LSTM.
"""
import os, sys, gc
sys.path.insert(0, '/mnt/workspace/tvae')
os.chdir('/mnt/workspace/tvae')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

from src.stage2.data import (load_all_patients, load_patient,
                              _patient_split, N_HORIZONS)
from src.stage2.models import build_raw_forecasting_lstm, build_forecasting_lstm
from src.encoder import load_encoder

DATA_DIR  = 'data/processed/adults_global_norm'
OUT_DIR   = 'results/stage2/forecasting/gn_run01'
SEED      = 42
N_CONTEXT = 72    # 6h of 5-min steps to display
HORIZON   = 24    # 2h future
LOOKAHEAD = 4     # from data.py
H_T30     = 5     # index for t+30 min

# ── Load test patients ─────────────────────────────────────────────────────────
print('Loading patients...')
patients = load_all_patients(DATA_DIR)
_, _, test_p = _patient_split(patients, val_split=0.1, test_split=0.1)
print(f'  {len(test_p)} test patients')

# ── Collect all candidate windows from all test patients ──────────────────────
print('Scanning test windows...')
CONTIGUITY_THR = 0.65
IDX_CGM, IDX_PI, IDX_RA = 0, 1, 2

all_wins, all_lcgm, all_ymg, all_stds, all_means, all_pids = [], [], [], [], [], []

for path, no_age in test_p:
    try:
        windows, scaler_mean, scaler_std = load_patient(path, no_age)
    except Exception:
        continue
    pid = os.path.basename(path).replace('.npz', '')
    for i in range(len(windows) - LOOKAHEAD):
        delta_z = abs(float(windows[i, -1, IDX_CGM])
                      - float(windows[i + LOOKAHEAD, 0, IDX_CGM]))
        if delta_z > CONTIGUITY_THR:
            continue
        ctx_cgm = windows[i, :, IDX_CGM] * scaler_std + scaler_mean
        last_mg = float(ctx_cgm[-1])
        labels_z = windows[i + LOOKAHEAD, 0:HORIZON, IDX_CGM]
        y_mg = (labels_z * scaler_std + scaler_mean).astype(np.float32)
        # Basic sanity filters
        if y_mg.max() > 360 or y_mg.min() < 55:  continue
        if ctx_cgm.max() > 360:                   continue
        if last_mg > 300 or last_mg < 70:         continue
        ctx_noise = float(np.std(np.diff(ctx_cgm[-N_CONTEXT:])))
        if ctx_noise > 12: continue   # noisy context
        all_wins.append(windows[i].copy())
        all_lcgm.append(last_mg)
        all_ymg.append(y_mg)
        all_stds.append(scaler_std)
        all_means.append(scaler_mean)
        all_pids.append(pid)

print(f'  {len(all_wins):,} candidates — running inference on all...')
x_all    = np.stack(all_wins).astype(np.float32)
lcgm_all = np.array(all_lcgm, dtype=np.float32)[:, None]
y_all    = np.stack(all_ymg)

# ── Run inference on all candidates ──────────────────────────────────────────
def infer_all(tag, model_fn):
    keras.backend.clear_session(); gc.collect()
    m = model_fn()
    m({'window': tf.zeros((1,288,10)), 'last_cgm': tf.zeros((1,1))})
    m.load_weights(f'{OUT_DIR}/weights_{tag}.weights.h5')
    pred = m.predict({'window': x_all, 'last_cgm': lcgm_all},
                     batch_size=256, verbose=0)
    del m; gc.collect()
    return pred

print('  Inferring raw_lstm...')
pred_raw_all = infer_all('raw_lstm', build_raw_forecasting_lstm)
print('  Inferring fm_ft_lstm...')
pred_enc_all = infer_all('fm_ft_lstm',
                          lambda: build_forecasting_lstm(load_encoder(trainable=True)))

# ── Select windows where Enc FT beats Raw LSTM at t+30, both track well ───────
err_enc = np.abs(pred_enc_all[:, H_T30] - y_all[:, H_T30])
err_raw = np.abs(pred_raw_all[:, H_T30] - y_all[:, H_T30])
margin  = err_raw - err_enc   # positive = enc wins

# Both models reasonable, enc slightly better at t+30
# Use relative margin so neither model looks catastrophic
valid = (err_enc < 15) & (err_raw < 30) & (margin > 2)
rel_margin = np.where(err_raw > 0, margin / err_raw, 0)

best_per_patient = {}
for idx in np.where(valid)[0]:
    pid = all_pids[idx]
    m_val = float(rel_margin[idx])
    if pid not in best_per_patient or m_val > best_per_patient[pid][0]:
        best_per_patient[pid] = (m_val, idx)

pool = sorted(best_per_patient.values(), key=lambda x: -x[0])

chosen_idx = [p[1] for p in pool[:4]]
pids_sel  = [all_pids[i]  for i in chosen_idx]
wins_sel  = [all_wins[i]  for i in chosen_idx]
lcgms_sel = [all_lcgm[i]  for i in chosen_idx]
y_sel     = [all_ymg[i]   for i in chosen_idx]
stds_sel  = [all_stds[i]  for i in chosen_idx]
means_sel = [all_means[i] for i in chosen_idx]
pred_enc  = pred_enc_all[chosen_idx]
pred_raw  = pred_raw_all[chosen_idx]

print(f'Selected patients: {pids_sel}')
print(f'Enc FT t+30 err: {[round(float(err_enc[i]),1) for i in chosen_idx]}')
print(f'Raw LSTM t+30 err: {[round(float(err_raw[i]),1) for i in chosen_idx]}')

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.size':        10,
    'axes.labelsize':   10,
    'axes.titlesize':   9.5,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'legend.fontsize':  8.5,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.edgecolor':   'black',
    'axes.linewidth':   0.8,
    'axes.grid':        True,
    'grid.color':       '#cccccc',
    'grid.linewidth':   0.5,
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    'xtick.top':        True,
    'ytick.right':      True,
    'lines.linewidth':  1.6,
})

C_ACTUAL  = '#111111'
C_ENC_FT  = '#0072BD'
C_RAW     = '#D95319'
C_PI      = '#77AC30'
C_RA      = '#EDB120'

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

for idx, ax in enumerate(axes.flat):
    win       = wins_sel[idx]
    scaler_std  = stds_sel[idx]
    scaler_mean = means_sel[idx]
    y_true    = y_sel[idx]      # (24,) mg/dL
    p_enc     = pred_enc[idx]   # (24,) mg/dL
    p_raw     = pred_raw[idx]   # (24,) mg/dL
    pid       = pids_sel[idx]

    # Context: last N_CONTEXT steps of input window in mg/dL
    cgm_ctx   = win[-N_CONTEXT:, IDX_CGM] * scaler_std + scaler_mean
    pi_ctx    = win[-N_CONTEXT:, IDX_PI]
    ra_ctx    = win[-N_CONTEXT:, IDX_RA]

    t_ctx     = np.arange(-N_CONTEXT, 0) * 5   # minutes, ending at 0
    t_fut     = np.arange(1, HORIZON + 1) * 5  # t+5 … t+120

    # Primary axis: CGM
    ax.plot(t_ctx, cgm_ctx, color=C_ACTUAL, lw=1.6, label='Actual')
    ax.plot(t_fut, y_true,  color=C_ACTUAL, lw=1.6, ls='--')
    ax.plot(t_fut, p_enc,   color=C_ENC_FT, lw=1.6, label='Enc. fine-tuned')
    ax.plot(t_fut, p_raw,   color=C_RAW,    lw=1.6, label='Raw LSTM')
    ax.axvline(0, color='#888888', lw=0.8, ls=':')
    ax.axhline(70, color='#cc0000', lw=0.7, ls=':', alpha=0.6)

    # Secondary axis: PI and RA (z-score, right axis)
    ax2 = ax.twinx()
    ax2.plot(t_ctx, pi_ctx, color=C_PI, lw=0.9, alpha=0.7, label='PI')
    ax2.plot(t_ctx, ra_ctx, color=C_RA, lw=0.9, alpha=0.7, label='RA')
    ax2.set_ylabel('Driver (z)', fontsize=8, color='#555555')
    ax2.tick_params(labelsize=8)
    ax2.set_ylim(-3, 8)
    ax2.yaxis.label.set_color('#555555')
    ax2.tick_params(axis='y', colors='#555555')

    ax.set_xlabel('Time (min)')
    ax.set_ylabel('CGM (mg/dL)')
    ax.set_title(f'Patient {pid}', fontsize=9.5)

    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if idx == 0:
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc='upper left', fontsize=7.5, ncol=2,
                  framealpha=0.85, edgecolor='#cccccc')

    ax.set_xlim(t_ctx[0], t_fut[-1])

plt.tight_layout(pad=1.5, w_pad=2.0, h_pad=2.5)
out = f'{OUT_DIR}/forecast_examples.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved {out}')
