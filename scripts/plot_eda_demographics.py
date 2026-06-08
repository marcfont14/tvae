"""EDA figure 1: cohort demographics (age, modality, dataset, sex). MATLAB style."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR   = '/mnt/workspace/tvae/data/processed/adults_global_norm'
OUT_PATH   = '/mnt/workspace/tvae/results/eda/demographics.png'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10, 'axes.labelsize': 10,
    'axes.titlesize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 8.5, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'axes.edgecolor': 'black',
    'axes.linewidth': 0.8, 'axes.grid': False,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'lines.linewidth': 1.6,
})
C_BLUE   = '#0072BD'
C_RED    = '#D95319'
C_GREEN  = '#77AC30'
C_GREY   = '#888888'

print('Loading patient metadata...')
ages, modalities, datasets = [], [], []
for fn in sorted(os.listdir(DATA_DIR)):
    if not fn.endswith('.npz'):
        continue
    d = np.load(os.path.join(DATA_DIR, fn))
    ages.append(float(d['age'][0]))
    modalities.append(str(d['modality'][0]))
    pid = str(d['patient_id'][0])
    datasets.append('T1DEXI' if pid.startswith('T_') else 'METABONET')

ages = np.array(ages)
print(f'Loaded {len(ages)} patients. Age range {ages.min():.0f}–{ages.max():.0f}')

# --- Sex (not stored in npz — hardcode from notebook stats)
sex_labels  = ['Female', 'Male', 'Unknown']
sex_counts  = [650, 342, 45]
sex_colours = [C_BLUE, C_RED, C_GREY]

# --- Modality counts
mod_order   = ['AID', 'SAP', 'MDI']
mod_counts  = [modalities.count(m) for m in mod_order]
mod_colours = [C_BLUE, C_RED, C_GREEN]

# --- Dataset counts
ds_labels  = ['METABONET', 'T1DEXI']
ds_counts  = [datasets.count('METABONET'), datasets.count('T1DEXI')]
ds_colours = [C_BLUE, C_RED]

fig, axes = plt.subplots(1, 4, figsize=(13, 4))

# Panel A — Age histogram
ax = axes[0]
bins = np.arange(18, ages.max() + 6, 5)
ax.hist(ages, bins=bins, color=C_BLUE, edgecolor='white', linewidth=0.4)
ax.set_xlabel('Age (years)')
ax.set_ylabel('Patients')
ax.set_title('Age distribution')
ax.set_xlim(18, ages.max() + 2)

# Panel B — Therapy modality
ax = axes[1]
bars = ax.bar(mod_order, mod_counts, color=mod_colours, edgecolor='white', linewidth=0.4, width=0.5)
for bar, cnt in zip(bars, mod_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
            str(cnt), ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Patients')
ax.set_title('Therapy modality')
ax.set_ylim(0, max(mod_counts) * 1.15)

# Panel C — Dataset source
ax = axes[2]
bars = ax.bar(ds_labels, ds_counts, color=ds_colours, edgecolor='white', linewidth=0.4, width=0.5)
for bar, cnt in zip(bars, ds_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
            str(cnt), ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Patients')
ax.set_title('Dataset source')
ax.set_ylim(0, max(ds_counts) * 1.15)

# Panel D — Sex
ax = axes[3]
bars = ax.bar(sex_labels, sex_counts, color=sex_colours, edgecolor='white', linewidth=0.4, width=0.5)
for bar, cnt in zip(bars, sex_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            str(cnt), ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Patients')
ax.set_title('Sex')
ax.set_ylim(0, max(sex_counts) * 1.15)

for ax in axes:
    ax.yaxis.grid(True, color='#cccccc', linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

fig.tight_layout(pad=1.5, w_pad=2.5)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
print(f'Saved {OUT_PATH}')
