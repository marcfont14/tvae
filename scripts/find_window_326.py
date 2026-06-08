"""
Find reconstruction examples where the FM produces rich, dynamic outputs.
Matches the original plot_reconstruction_examples approach:
  - Sample 2000 random test windows
  - Apply random masks
  - Run inference
  - Rank by CGM range within the masked region (shows best FM outputs)
  - Keep one window per patient for diversity
"""
import os, sys
sys.path.insert(0, '/mnt/workspace/tvae')
os.chdir('/mnt/workspace/tvae')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

from scripts.experiment_mtsm import (
    D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT,
    WINDOW_LEN, CGM_IDX, PI_IDX, RA_IDX, BOLUS_IDX, CARBS_IDX,
    MASK_RATIO, MASK_MIN_LEN, MASK_MAX_LEN, MASK_TOKEN,
    build_mtsm_model, K_BINS, BIN_Z_MIN, BIN_Z_MAX, bin_to_cgm_z, create_mask,
)

CGM_MEAN, CGM_STD = 144.40, 57.11
N_SAMPLE   = 10000   # random test windows to evaluate
KEEP_FIXED = {70489, 72955, 71840}  # confirmed good; always include

# ── Build model → expected-z-score predictor ─────────────────────────────────
print('Building model...')
model, encoder = build_mtsm_model(
    WINDOW_LEN, 10, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT,
    no_causal=True, ce_head=True
)
model(tf.zeros((1, WINDOW_LEN, 10)), training=False)
model.load_weights('results/mtsm/encoder_global_norm/model_weights.weights.h5')

bin_z_vals = tf.constant(
    bin_to_cgm_z(np.arange(K_BINS, dtype=np.float32)), dtype=tf.float32
)
_inp    = keras.Input(shape=(WINDOW_LEN, 10))
_logits = model(_inp)
_probs  = tf.nn.softmax(_logits, axis=-1)
_pred_z = tf.reduce_sum(_probs * bin_z_vals[None, None, :], axis=-1)
pred_model = keras.Model(_inp, _pred_z, name='pred_z')
print('Loaded.')

# ── Load true test windows ────────────────────────────────────────────────────
with open('results/outlier_analysis/pretrain_patients.txt') as f:
    pretrain = set(os.path.basename(l.strip()) for l in f if l.strip())

all_files = sorted([
    os.path.join('data/processed/adults_global_norm', fn)
    for fn in os.listdir('data/processed/adults_global_norm')
    if fn.endswith('.npz')
])
test_files = [fp for fp in all_files if os.path.basename(fp) not in pretrain]
print(f'True test patients: {len(test_files)}')

all_windows, all_info = [], []
for fp in test_files:
    d  = np.load(fp)
    ws = d['windows'][:, :, :10].astype(np.float32)
    pid = os.path.basename(fp).replace('.npz', '')
    for wi, w in enumerate(ws):
        all_windows.append(w)
        all_info.append((pid, wi))

all_windows = np.array(all_windows)
print(f'Total test windows: {len(all_windows)}')

# ── Random sample ─────────────────────────────────────────────────────────────
np.random.seed(123)
sample_idx = np.random.choice(len(all_windows), min(N_SAMPLE, len(all_windows)), replace=False)
# Always include the confirmed good windows
for fixed in KEEP_FIXED:
    if fixed not in sample_idx:
        sample_idx = np.append(sample_idx, fixed)
sample_windows = all_windows[sample_idx]
sample_info    = [all_info[i] for i in sample_idx]

# ── Random masks ──────────────────────────────────────────────────────────────
np.random.seed(123)
masks_sample = np.stack([
    create_mask(WINDOW_LEN, MASK_RATIO, MASK_MIN_LEN, MASK_MAX_LEN)
    for _ in range(len(sample_idx))
])

# ── Inference ─────────────────────────────────────────────────────────────────
x_masked = sample_windows.copy()
for j in range(len(sample_idx)):
    m = masks_sample[j].astype(bool)
    x_masked[j, m, CGM_IDX] = MASK_TOKEN

print(f'Running inference on {len(sample_idx)} windows...')
pred_z_all = pred_model.predict(x_masked, batch_size=128, verbose=0)  # (N, 288) z
pred_mg    = pred_z_all * CGM_STD + CGM_MEAN

# ── Rank by FM reconstruction range within mask (richest predictions first) ──
recon_range = np.array([
    pred_mg[j][masks_sample[j].astype(bool)].ptp()
    for j in range(len(sample_idx))
])

# Also require: actual CGM not hitting ceiling/floor
cgm_mg_sample = sample_windows[:, :, CGM_IDX] * CGM_STD + CGM_MEAN
max_cgm = cgm_mg_sample.max(axis=1)
min_cgm = cgm_mg_sample.min(axis=1)
mean_step = np.abs(np.diff(cgm_mg_sample, axis=1)).mean(axis=1)
quality   = (max_cgm < 330) & (min_cgm > 60) & (mean_step < 5.0)

ranked_local = np.argsort(recon_range)[::-1]

# Pin confirmed good windows first
seen_patients = set()
SHOW_LOCAL = []
for j in range(len(sample_idx)):
    if sample_idx[j] in KEEP_FIXED:
        pid, _ = sample_info[j]
        seen_patients.add(pid)
        SHOW_LOCAL.append(j)

# Fill remaining slots from ranked candidates (one per patient)
for j in ranked_local:
    if not quality[j]:
        continue
    if sample_idx[j] in KEEP_FIXED:
        continue  # already added
    pid, _ = sample_info[j]
    if pid not in seen_patients:
        seen_patients.add(pid)
        SHOW_LOCAL.append(j)
    if len(SHOW_LOCAL) == 8:   # 3 confirmed + 5 new
        break

print(f'Selected {len(SHOW_LOCAL)} windows')
print('Recon ranges (mg/dL):', [f'{recon_range[j]:.1f}' for j in SHOW_LOCAL])

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': 'black', 'axes.linewidth': 0.8,
    'axes.grid': True, 'grid.color': '#cccccc', 'grid.linewidth': 0.5,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'lines.linewidth': 1.5,
})

n_rows = len(SHOW_LOCAL)
fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.2 * n_rows))
if n_rows == 1:
    axes = [axes]

t = np.arange(WINDOW_LEN) * 5 / 60

for row, j in enumerate(SHOW_LOCAL):
    ax        = axes[row]
    pid, wi   = sample_info[j]
    w         = sample_windows[j]
    m         = masks_sample[j].astype(bool)
    cgm_mg    = w[:, CGM_IDX] * CGM_STD + CGM_MEAN
    pi_z      = w[:, PI_IDX]
    ra_z      = w[:, RA_IDX]
    recon_mg  = np.where(m, pred_mg[j], np.nan)

    # Linear interpolation over each masked span
    lin = cgm_mg.copy()
    in_mask = False
    for tt in range(WINDOW_LEN):
        if m[tt] and not in_mask:
            start = tt; in_mask = True
        if (not m[tt] or tt == WINDOW_LEN - 1) and in_mask:
            end = tt
            v0 = cgm_mg[start - 1] if start > 0 else cgm_mg[end]
            v1 = cgm_mg[end]       if end < WINDOW_LEN else cgm_mg[start - 1]
            lin[start:end] = np.linspace(v0, v1, end - start)
            in_mask = False

    # Shade masked regions
    for tt in range(WINDOW_LEN - 1):
        if m[tt]:
            ax.axvspan(t[tt], t[tt+1], alpha=0.10, color='steelblue', lw=0)

    ax.plot(t, cgm_mg,          color='#000000', lw=1.8, label='Actual CGM', zorder=5)
    ax.plot(t, recon_mg,        color='#0072BD', lw=1.8, ls='--', label='FM recon.', zorder=6)
    ax.plot(t[m], lin[m],       color='#D95319', lw=1.4, ls=':', label='Linear', zorder=4)
    ax.axhline(70,  color='#bbbbbb', lw=0.7, ls=':')
    ax.axhline(180, color='#bbbbbb', lw=0.7, ls=':')

    ax_r = ax.twinx()
    ax_r.plot(t, pi_z, color='#77AC30', lw=1.0, alpha=0.7, label='PI (z)')
    ax_r.plot(t, ra_z, color='#EDB120', lw=1.0, alpha=0.7, label='RA (z)')
    ax_r.set_ylabel('Driver signal (z)', fontsize=9)
    ax_r.tick_params(labelsize=8)

    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc='upper right', fontsize=8, ncol=2)

    ax.set_ylabel('Glucose (mg/dL)')
    ax.set_xlabel('Time (h)')
    global_idx = sample_idx[j]
    ax.set_title(
        f'global_idx={global_idx}  patient={pid}  win={wi}  '
        f'recon_range={recon_range[j]:.1f} mg/dL',
        fontsize=9
    )
    ax.set_xlim(0, 24)

plt.tight_layout(pad=1.2)
out = 'results/mtsm/encoder_global_norm/window_326_candidates.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved {out}')
print('Global indices:', [sample_idx[j] for j in SHOW_LOCAL])
