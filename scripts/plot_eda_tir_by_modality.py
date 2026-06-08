"""EDA figure: TIR(70-180) and mean CGM stratified by therapy modality (AID / SAP / MDI)."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = '/mnt/workspace/tvae/data/processed/adults_global_norm'
OUT_PATH = '/mnt/workspace/tvae/results/eda/tir_by_modality.png'

CGM_MEAN, CGM_STD = 144.40, 57.11

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10, 'axes.labelsize': 10,
    'axes.titlesize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 8.5, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'axes.edgecolor': 'black',
    'axes.linewidth': 0.8, 'axes.grid': False,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
})
C_AID  = '#0072BD'
C_SAP  = '#D95319'
C_MDI  = '#77AC30'
COLORS = {'AID': C_AID, 'SAP': C_SAP, 'MDI': C_MDI}

tirs   = {'AID': [], 'SAP': [], 'MDI': []}
means  = {'AID': [], 'SAP': [], 'MDI': []}

print('Loading npz files...')
for fn in sorted(os.listdir(DATA_DIR)):
    if not fn.endswith('.npz'):
        continue
    d   = np.load(os.path.join(DATA_DIR, fn))
    mod = str(d['modality'][0])
    if mod not in tirs:
        continue
    cgm_mg = d['windows'][:, :, 0].astype(np.float32) * CGM_STD + CGM_MEAN
    tirs[mod].append(float(((cgm_mg >= 70) & (cgm_mg <= 180)).mean()) * 100)
    means[mod].append(float(cgm_mg.mean()))

for mod in ['AID', 'SAP', 'MDI']:
    arr = np.array(tirs[mod])
    print(f'{mod:4s}  N={len(arr):3d}  TIR med={np.median(arr):.1f}%  '
          f'mean CGM med={np.median(means[mod]):.1f} mg/dL')

ORDER = ['AID', 'SAP', 'MDI']
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

def add_grid(ax):
    ax.yaxis.grid(True, color='#cccccc', linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

# Panel A — TIR boxplot
ax = axes[0]
data_tir = [tirs[m] for m in ORDER]
bp = ax.boxplot(data_tir, patch_artist=True, widths=0.45,
                medianprops=dict(color='black', linewidth=1.8),
                whiskerprops=dict(linewidth=0.9),
                capprops=dict(linewidth=0.9),
                flierprops=dict(marker='o', markersize=2.5,
                                markerfacecolor='#888888', alpha=0.5,
                                linestyle='none'))
for patch, mod in zip(bp['boxes'], ORDER):
    patch.set_facecolor(COLORS[mod])
    patch.set_alpha(0.75)
ax.axhline(70, color='#D95319', linestyle='--', linewidth=1.1,
           label='70% guideline target')
ax.set_xticks([1, 2, 3])
ax.set_xticklabels([f'{m}\n(n={len(tirs[m])})' for m in ORDER])
ax.set_ylabel('TIR 70–180 mg/dL (%)')
ax.set_title('Time-in-range by therapy modality')
ax.legend(framealpha=0.9, edgecolor='#cccccc', fontsize=8)
add_grid(ax)

# Panel B — mean CGM boxplot
ax = axes[1]
data_cgm = [means[m] for m in ORDER]
bp = ax.boxplot(data_cgm, patch_artist=True, widths=0.45,
                medianprops=dict(color='black', linewidth=1.8),
                whiskerprops=dict(linewidth=0.9),
                capprops=dict(linewidth=0.9),
                flierprops=dict(marker='o', markersize=2.5,
                                markerfacecolor='#888888', alpha=0.5,
                                linestyle='none'))
for patch, mod in zip(bp['boxes'], ORDER):
    patch.set_facecolor(COLORS[mod])
    patch.set_alpha(0.75)
ax.axhline(180, color='#D95319', linestyle=':', linewidth=1.0)
ax.axhline(70,  color='#D95319', linestyle=':', linewidth=1.0, label='70 / 180 mg/dL')
ax.set_xticks([1, 2, 3])
ax.set_xticklabels([f'{m}\n(n={len(means[m])})' for m in ORDER])
ax.set_ylabel('Mean CGM (mg/dL)')
ax.set_title('Mean glucose by therapy modality')
ax.legend(framealpha=0.9, edgecolor='#cccccc', fontsize=8)
add_grid(ax)

fig.tight_layout(pad=1.5, w_pad=3.0)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
print(f'Saved {OUT_PATH}')
