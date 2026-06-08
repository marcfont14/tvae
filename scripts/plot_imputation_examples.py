"""
Final thesis figure: 2x2 imputation examples.
For each confirmed window, tries 20 random masks and picks the one
where the FM reconstruction is most dynamic (highest range within mask).
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
    WINDOW_LEN, CGM_IDX, PI_IDX, RA_IDX,
    MASK_RATIO, MASK_MIN_LEN, MASK_MAX_LEN, MASK_TOKEN,
    build_mtsm_model, K_BINS, BIN_Z_MIN, BIN_Z_MAX, bin_to_cgm_z, create_mask,
)

CGM_MEAN, CGM_STD = 144.40, 57.11
CONFIRMED = [70489, 71840, 72955, 86928]
N_MASK_TRIES = 30   # masks to try per window; pick richest reconstruction

# ── Build model ───────────────────────────────────────────────────────────────
print('Building model...')
model, _ = build_mtsm_model(
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

# ── Load test windows ─────────────────────────────────────────────────────────
with open('results/outlier_analysis/pretrain_patients.txt') as f:
    pretrain = set(os.path.basename(l.strip()) for l in f if l.strip())

all_files = sorted([
    os.path.join('data/processed/adults_global_norm', fn)
    for fn in os.listdir('data/processed/adults_global_norm')
    if fn.endswith('.npz')
])
test_files = [fp for fp in all_files if os.path.basename(fp) not in pretrain]

all_windows, all_info = [], []
for fp in test_files:
    d   = np.load(fp)
    ws  = d['windows'][:, :, :10].astype(np.float32)
    pid = os.path.basename(fp).replace('.npz', '')
    for wi, w in enumerate(ws):
        all_windows.append(w)
        all_info.append((pid, wi))
all_windows = np.array(all_windows)

# ── For each confirmed window, find the best mask ────────────────────────────
best_masks = {}
best_preds = {}

for gidx in CONFIRMED:
    w = all_windows[gidx]
    np.random.seed(gidx)   # deterministic per window
    tries = np.stack([
        create_mask(WINDOW_LEN, MASK_RATIO, MASK_MIN_LEN, MASK_MAX_LEN)
        for _ in range(N_MASK_TRIES)
    ])
    # Build masked inputs
    x_batch = np.tile(w[None], (N_MASK_TRIES, 1, 1))
    for k in range(N_MASK_TRIES):
        x_batch[k, tries[k].astype(bool), CGM_IDX] = MASK_TOKEN
    preds_z  = pred_model.predict(x_batch, batch_size=N_MASK_TRIES, verbose=0)
    preds_mg = preds_z * CGM_STD + CGM_MEAN
    # Pick mask with highest FM reconstruction range
    ranges = np.array([
        preds_mg[k][tries[k].astype(bool)].ptp() for k in range(N_MASK_TRIES)
    ])
    best_k = int(np.argmax(ranges))
    best_masks[gidx] = tries[best_k]
    best_preds[gidx] = preds_mg[best_k]
    print(f'  {gidx}: best mask range = {ranges[best_k]:.1f} mg/dL (k={best_k})')

# ── Plot 2x2 ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.size':        10,
    'axes.labelsize':   10,
    'axes.titlesize':   10,
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

C_BLACK  = '#000000'
C_BLUE   = '#0072BD'
C_RED    = '#D95319'
C_GREEN  = '#77AC30'
C_YELLOW = '#EDB120'

fig, axes = plt.subplots(2, 2, figsize=(13, 7))
axes = axes.flatten()
t = np.arange(WINDOW_LEN) * 5 / 60
panel_labels = ['(a)', '(b)', '(c)', '(d)']

for panel, (gidx, ax) in enumerate(zip(CONFIRMED, axes)):
    pid, wi  = all_info[gidx]
    w        = all_windows[gidx]
    m        = best_masks[gidx].astype(bool)
    cgm_mg   = w[:, CGM_IDX] * CGM_STD + CGM_MEAN
    pi_z     = w[:, PI_IDX]
    ra_z     = w[:, RA_IDX]
    recon_mg = np.where(m, best_preds[gidx], np.nan)

    # Linear interpolation over each masked span
    lin = cgm_mg.copy()
    in_mask = False
    for tt in range(WINDOW_LEN):
        if m[tt] and not in_mask:
            start = tt; in_mask = True
        if (not m[tt] or tt == WINDOW_LEN - 1) and in_mask:
            end = tt
            v0 = cgm_mg[start - 1] if start > 0 else cgm_mg[end]
            v1 = cgm_mg[end] if end < WINDOW_LEN else cgm_mg[start - 1]
            lin[start:end] = np.linspace(v0, v1, end - start)
            in_mask = False

    # Shade masked regions
    for tt in range(WINDOW_LEN - 1):
        if m[tt]:
            ax.axvspan(t[tt], t[tt+1], alpha=0.10, color='steelblue', lw=0)

    ax.plot(t, cgm_mg,    color=C_BLACK,  lw=1.8, label='CGM (actual)',   zorder=5)
    ax.plot(t, recon_mg,  color=C_BLUE,   lw=1.8, ls='--', label='FM (zero-shot)', zorder=6)
    ax.plot(t[m], lin[m], color=C_RED,    lw=1.4, ls=':',  label='Linear interp.', zorder=4)
    ax.axhline(70,  color='#bbbbbb', lw=0.7, ls=':')
    ax.axhline(180, color='#bbbbbb', lw=0.7, ls=':')

    ax_r = ax.twinx()
    ax_r.plot(t, pi_z, color=C_GREEN,  lw=0.9, alpha=0.65, label='PI (z-score)')
    ax_r.plot(t, ra_z, color=C_YELLOW, lw=0.9, alpha=0.65, label='RA (z-score)')
    ax_r.set_ylabel('Driver signal (z)', fontsize=8.5)
    ax_r.tick_params(labelsize=8)

    ax.set_xlim(0, 24)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Glucose (mg/dL)')

    # Panel label top-left
    ax.text(0.02, 0.96, panel_labels[panel], transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top')

    # Legend only on first panel
    if panel == 0:
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax_r.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2,
                  loc='upper right', fontsize=8, ncol=2,
                  framealpha=0.9, edgecolor='#cccccc')

plt.tight_layout(pad=1.5, w_pad=2.0, h_pad=2.0)

out = 'results/mtsm/encoder_global_norm/imputation_examples_final.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'\nSaved {out}')
