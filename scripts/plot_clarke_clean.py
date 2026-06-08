"""
Clarke Error Grid — t+30 and t+60 min, Encoder fine-tuned, 30 test patients.
Uses the validated _clarke_point / clarke_zones from src/stage2/evaluate.py.
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

from src.stage2.data import load_all_patients, make_eval_dataset, _patient_split
from src.stage2.models import build_forecasting_lstm
from src.stage2.evaluate import _clarke_point, clarke_zones
from src.encoder import load_encoder

DATA_DIR    = 'data/processed/adults_global_norm'
OUT_DIR     = 'results/stage2/forecasting/gn_run01'
N_PATIENTS  = 30
SEED        = 42
H_T30       = 5    # 0-indexed: t+30 min
H_T60       = 11   # 0-indexed: t+60 min

# ── Load 30 test patients ──────────────────────────────────────────────────────
print('Loading patients...')
patients = load_all_patients(DATA_DIR)
_, _, test_p = _patient_split(patients, val_split=0.1, test_split=0.1)

np.random.seed(SEED)
idx = np.random.choice(len(test_p), N_PATIENTS, replace=False)
subset = [test_p[i] for i in sorted(idx)]
print(f'  Using {len(subset)} test patients')

test_ds = make_eval_dataset(subset, batch_size=256)

print('Collecting test windows...')
win_list, lcgm_list, y_list = [], [], []
for xb, yb in test_ds:
    win_list.append(xb['window'].numpy())
    lcgm_list.append(xb['last_cgm'].numpy())
    y_list.append(yb.numpy())
windows  = np.concatenate(win_list)
last_cgm = np.concatenate(lcgm_list)
y_mg     = np.concatenate(y_list)    # (N, 24) mg/dL
print(f'  {len(windows):,} windows')

# ── Inference — Encoder fine-tuned ────────────────────────────────────────────
print('Inferring fm_ft_lstm...')
keras.backend.clear_session(); gc.collect()
m = build_forecasting_lstm(load_encoder(trainable=True))
m({'window': tf.zeros((1,288,10)), 'last_cgm': tf.zeros((1,1))})
m.load_weights(f'{OUT_DIR}/weights_fm_ft_lstm.weights.h5')
pred_mg = m.predict({'window': windows, 'last_cgm': last_cgm},
                    batch_size=256, verbose=0)   # (N, 24) mg/dL
del m; gc.collect()

# ── rcParams ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.size':        10,
    'axes.labelsize':   10,
    'axes.titlesize':   9.5,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.edgecolor':   'black',
    'axes.linewidth':   0.8,
    'axes.grid':        False,
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    'xtick.top':        True,
    'ytick.right':      True,
})

ZONE_COLORS = {'A': '#77AC30', 'B': '#0072BD', 'C': '#EDB120',
               'D': '#D95319', 'E': '#7E2F8E'}

def draw_clarke(ax, ref, pred, title):
    point_zones = [_clarke_point(float(r), float(p)) for r, p in zip(ref, pred)]
    colors = [ZONE_COLORS[z] for z in point_zones]
    ax.scatter(ref, pred, c=colors, s=4, alpha=0.3, linewidths=0, rasterized=True)

    # Diagonal
    ax.plot([0, 400], [0, 400], 'k:', lw=0.9)

    # Zone boundaries — exact MATLAB lines from evaluate.py
    kw = dict(color='k', lw=0.9)
    ax.plot([0,       175/3],    [70,  70],  **kw)
    ax.plot([175/3,   400/1.2],  [70,  400], **kw)
    ax.plot([70,      70],       [84,  400], **kw)
    ax.plot([0,       70],       [180, 180], **kw)
    ax.plot([70,      290],      [180, 400], **kw)
    ax.plot([70,      70],       [0,   56],  **kw)
    ax.plot([70,      400],      [56,  320], **kw)
    ax.plot([180,     180],      [0,   70],  **kw)
    ax.plot([180,     400],      [70,  70],  **kw)
    ax.plot([240,     240],      [70,  180], **kw)
    ax.plot([240,     400],      [180, 180], **kw)
    ax.plot([130,     180],      [0,   70],  **kw)

    # Zone labels
    fs = 9
    ax.text(30,  20,  'A', fontsize=fs, color='#444444')
    ax.text(30,  150, 'D', fontsize=fs, color='#444444')
    ax.text(30,  380, 'E', fontsize=fs, color='#444444')
    ax.text(150, 380, 'C', fontsize=fs, color='#444444')
    ax.text(160, 20,  'C', fontsize=fs, color='#444444')
    ax.text(380, 20,  'E', fontsize=fs, color='#444444')
    ax.text(380, 120, 'D', fontsize=fs, color='#444444')
    ax.text(380, 260, 'B', fontsize=fs, color='#444444')
    ax.text(280, 380, 'B', fontsize=fs, color='#444444')

    zones = clarke_zones(ref, pred)
    ann = '  '.join(f'{z}:{zones[z]:.1%}' for z in 'ABCDE')
    ax.text(0.02, 0.98, ann, transform=ax.transAxes,
            fontsize=7.5, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cccccc', lw=0.5))

    ax.set_xlim(0, 400); ax.set_ylim(0, 400)
    ax.set_aspect('equal')
    ax.set_xlabel('Actual CGM (mg/dL)')
    ax.set_ylabel('Predicted CGM (mg/dL)')
    ax.set_title(title, fontsize=9.5)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

draw_clarke(axes[0], y_mg[:, H_T30], pred_mg[:, H_T30],
            'Encoder fine-tuned  (t+30 min)')
draw_clarke(axes[1], y_mg[:, H_T60], pred_mg[:, H_T60],
            'Encoder fine-tuned  (t+60 min)')

plt.tight_layout(pad=1.5, w_pad=3.0)
out = f'{OUT_DIR}/clarke_clean.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved {out}')
