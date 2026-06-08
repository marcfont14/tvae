"""
Preview candidate test patients for imputation figure selection.
Shows actual CGM + gap region + linear interpolation only (no model needed).
"""
import os, sys
sys.path.insert(0, '/mnt/workspace/tvae')
os.chdir('/mnt/workspace/tvae')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.stage2.data import load_all_patients, make_forecasting_dataset

PATIENT_INDICES = [13, 3, 7, 26, 24, 15]   # 6 candidates
GAP_H           = 6                          # preview at 6h gap

CGM_MEAN, CGM_STD = 144.40, 57.11

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': 'black', 'axes.linewidth': 0.8,
    'axes.grid': True, 'grid.color': '#cccccc', 'grid.linewidth': 0.5,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'lines.linewidth': 1.5,
})

patients = load_all_patients('data/processed/adults_global_norm')
splits   = make_forecasting_dataset(patients, batch_size=128)
test_pts = splits['test_patients']

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
axes = axes.flatten()

for i, pt_idx in enumerate(PATIENT_INDICES):
    path, _ = test_pts[pt_idx]
    data     = np.load(path)
    windows  = data['windows']
    pid      = os.path.basename(path).replace('.npz', '')

    # Pick window: highest CGM std among windows with at least one bolus event
    stds   = windows[:, :, 0].std(axis=1)
    has_ev = windows[:, :, 5].sum(axis=1) > 0
    cands  = np.where(has_ev)[0]
    if len(cands) == 0:
        cands = np.arange(len(windows))
    win_idx = cands[stds[cands].argmax()]
    w       = windows[win_idx]

    cgm    = w[:, 0] * CGM_STD + CGM_MEAN
    time_h = np.arange(288) * 5 / 60

    gap_steps = GAP_H * 12
    np.random.seed(42 + pt_idx)
    gap_start = np.random.randint(48, 288 - gap_steps - 48)

    v0 = cgm[gap_start - 1]
    v1 = cgm[gap_start + gap_steps]
    linear_gap = np.linspace(v0, v1, gap_steps)
    t_gap = time_h[gap_start:gap_start + gap_steps]

    ax = axes[i]
    ax.plot(time_h, cgm, color='#000000', lw=1.4, label='Actual CGM')
    ax.axvspan(time_h[gap_start], time_h[gap_start + gap_steps - 1],
               alpha=0.12, color='steelblue')
    ax.plot(t_gap, linear_gap, color='#D95319', lw=1.4, ls='--', label='Linear')
    ax.axhline(70,  color='#aaaaaa', lw=0.7, ls=':')
    ax.axhline(180, color='#aaaaaa', lw=0.7, ls=':')
    ax.set_title(f'Patient {pid} (idx={pt_idx})')
    ax.set_xlim(0, 24)
    ax.set_ylabel('Glucose (mg/dL)')
    ax.set_xlabel('Time (h)')
    if i == 0:
        ax.legend(loc='upper right', fontsize=8)

plt.tight_layout(pad=1.5)
out = 'results/stage2/imputation/gn_run01/preview_patients.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
