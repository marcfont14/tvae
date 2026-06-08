"""EDA figure: per-patient carbohydrate annotation rate — initial vs final cohort.

Streams raw parquets for both cohorts so the metric is identical and the
comparison is fair. Metric: fraction of calendar days with >= 1 logged carb event.
"""
import os
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NPZ_DIR  = '/mnt/workspace/tvae/data/processed/adults_global_norm'
OUT_PATH = '/mnt/workspace/tvae/results/eda/driver_blindness.png'

RAW_FILES = [
    '/mnt/workspace/tvae/data/raw/metabonet_public_train.parquet',
    '/mnt/workspace/tvae/data/raw/metabonet_public_test.parquet',
    '/mnt/workspace/tvae/data/raw/t1dexi_parsed.parquet',
]

ALPHA_INIT  = 0.55
ALPHA_FINAL = 0.85
C_INIT  = '#AAAAAA'
C_FINAL = '#0072BD'
C_MED   = '#D95319'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10, 'axes.labelsize': 10,
    'axes.titlesize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 8.5, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'axes.edgecolor': 'black',
    'axes.linewidth': 0.8, 'axes.grid': False,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'lines.linewidth': 1.6,
})

final_ids = set(
    fn.replace('.npz', '')
    for fn in os.listdir(NPZ_DIR)
    if fn.endswith('.npz')
)
print(f'Final cohort IDs: {len(final_ids)}')

# Per-patient: days with >= 1 CGM reading, days with >= 1 carb event
init_stats  = {}
final_stats = {}

def update(store, pid, cgm, carbs, dates):
    if pid not in store:
        store[pid] = {'days': set(), 'carb_days': set()}
    s = store[pid]
    s['days'].update(dates[cgm.notna()].dropna())
    s['carb_days'].update(dates[carbs.fillna(0) > 0].dropna())

for fpath in RAW_FILES:
    print(f'Streaming {os.path.basename(fpath)} ...')
    pf = pq.ParquetFile(fpath)
    for batch in pf.iter_batches(batch_size=500_000,
                                  columns=['id', 'CGM', 'carbs', 'date']):
        df = batch.to_pandas()
        df['_date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        for pid, grp in df.groupby('id', sort=False):
            pid_str = str(pid)
            update(init_stats, pid_str, grp['CGM'], grp['carbs'], grp['_date'])
            if pid_str in final_ids:
                update(final_stats, pid_str, grp['CGM'], grp['carbs'], grp['_date'])

def carb_fracs(store):
    return np.array([
        len(s['carb_days']) / max(len(s['days']), 1)
        for s in store.values()
    ])

init_frac  = carb_fracs(init_stats)
final_frac = carb_fracs(final_stats)
N_INIT, N_FINAL = len(init_frac), len(final_frac)

print(f'Initial (N={N_INIT}): median={np.median(init_frac):.3f}')
print(f'Final   (N={N_FINAL}): median={np.median(final_frac):.3f}')

fig, ax = plt.subplots(figsize=(7, 4.5))

bins = np.linspace(0, 1, 26)
ax.hist(init_frac,  bins=bins, color=C_INIT,  alpha=ALPHA_INIT,
        edgecolor='none', label=f'Initial cohort  (N = {N_INIT})')
ax.hist(final_frac, bins=bins, color=C_FINAL, alpha=ALPHA_FINAL,
        edgecolor='none', label=f'Final cohort  (N = {N_FINAL})')
ax.axvline(np.median(init_frac),  color=C_INIT,  linestyle='--', linewidth=1.4, alpha=0.9,
           label=f'Median {np.median(init_frac):.0%}')
ax.axvline(np.median(final_frac), color=C_FINAL, linestyle='--', linewidth=1.4,
           label=f'Median {np.median(final_frac):.0%}')

ax.set_xlabel('Fraction of days with ≥ 1 logged carbohydrate event')
ax.set_ylabel('Patients')
ax.set_title('Per-patient carbohydrate annotation rate')
ax.set_xlim(0, 1)
ax.legend(framealpha=0.9, edgecolor='#cccccc')
ax.yaxis.grid(True, color='#cccccc', linewidth=0.5, alpha=0.6)
ax.set_axisbelow(True)

fig.tight_layout(pad=1.5)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
print(f'Saved {OUT_PATH}')
