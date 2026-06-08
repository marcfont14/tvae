"""EDA figure 3: per-patient glycaemic profile (mean CGM and TIR). MATLAB style."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = '/mnt/workspace/tvae/data/processed/adults_global_norm'
OUT_PATH = '/mnt/workspace/tvae/results/eda/glycaemic_profile.png'

CGM_MEAN, CGM_STD = 144.40, 57.11

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10, 'axes.labelsize': 10,
    'axes.titlesize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 8.5, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'axes.edgecolor': 'black',
    'axes.linewidth': 0.8, 'axes.grid': False,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'lines.linewidth': 1.6,
})
C_BLUE = '#0072BD'
C_RED  = '#D95319'

print('Computing per-patient glycaemic stats...')
mean_cgms, tirs = [], []
for fn in sorted(os.listdir(DATA_DIR)):
    if not fn.endswith('.npz'):
        continue
    d = np.load(os.path.join(DATA_DIR, fn))
    cgm_mg = d['windows'][:, :, 0] * CGM_STD + CGM_MEAN  # (N, 288)
    mean_cgms.append(cgm_mg.mean())
    tirs.append(((cgm_mg >= 70) & (cgm_mg <= 180)).mean())

mean_cgms = np.array(mean_cgms)
tirs      = np.array(tirs)
print(f'n={len(mean_cgms)}, mean CGM={mean_cgms.mean():.1f} mg/dL, '
      f'median TIR={np.median(tirs)*100:.1f}%')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Panel A — per-patient mean CGM
ax = axes[0]
ax.hist(mean_cgms, bins=30, color=C_BLUE, edgecolor='white', linewidth=0.4)
ax.axvline(70,  color=C_RED, linestyle=':', linewidth=1.2)
ax.axvline(180, color=C_RED, linestyle=':', linewidth=1.2, label='70 / 180 mg/dL')
ax.axvline(mean_cgms.mean(), color='#000000', linestyle='--', linewidth=1.2,
           label=f'Mean = {mean_cgms.mean():.0f} mg/dL')
ax.set_xlabel('Per-patient mean CGM (mg/dL)')
ax.set_ylabel('Patients')
ax.set_title('Mean glucose distribution')
ax.legend(framealpha=0.9, edgecolor='#cccccc')
ax.yaxis.grid(True, color='#cccccc', linewidth=0.5, alpha=0.6)
ax.set_axisbelow(True)

# Panel B — per-patient TIR(70–180)
ax = axes[1]
ax.hist(tirs * 100, bins=30, color=C_BLUE, edgecolor='white', linewidth=0.4)
ax.axvline(70, color=C_RED, linestyle='--', linewidth=1.2,
           label='70% guideline target')
ax.axvline(np.median(tirs) * 100, color='#000000', linestyle='--', linewidth=1.2,
           label=f'Median = {np.median(tirs)*100:.0f}%')
ax.set_xlabel('Per-patient TIR 70–180 mg/dL (%)')
ax.set_ylabel('Patients')
ax.set_title('Time-in-range distribution')
ax.legend(framealpha=0.9, edgecolor='#cccccc')
ax.yaxis.grid(True, color='#cccccc', linewidth=0.5, alpha=0.6)
ax.set_axisbelow(True)

fig.tight_layout(pad=1.5, w_pad=2.5)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
print(f'Saved {OUT_PATH}')
