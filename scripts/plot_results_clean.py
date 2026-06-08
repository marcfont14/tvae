"""
Generate clean, MATLAB-like result figures for the thesis.
Figures that only need existing JSON metrics (no model re-run):
  1. Imputation performance by gap (RMSE + R²)
  2. Forecasting RMSE vs horizon (key models)
"""
import os, json
os.chdir('/mnt/workspace/tvae')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── MATLAB-like style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'axes.edgecolor':     'black',
    'axes.linewidth':     0.8,
    'axes.grid':          True,
    'grid.color':         '#cccccc',
    'grid.linewidth':     0.5,
    'grid.alpha':         1.0,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'xtick.top':          True,
    'ytick.right':        True,
    'lines.linewidth':    1.8,
    'lines.markersize':   6,
})

# MATLAB default colours
C_BLUE   = '#0072BD'
C_RED    = '#D95319'
C_YELLOW = '#EDB120'
C_GREEN  = '#77AC30'
C_LBLUE  = '#4DBEEE'
C_BLACK  = '#000000'

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Imputation performance by gap
# ═══════════════════════════════════════════════════════════════════════════════
gaps   = ['4h', '5h', '6h', '8h']
x_mins = [240, 300, 360, 480]   # gap in minutes (for x label)

fm_rmse, fm_r2   = [], []
lin_rmse, lin_r2 = [], []
raw_rmse, raw_r2 = [], []

for gap in gaps:
    d = json.load(open(f'results/stage2/imputation/gn_run01/metrics_{gap}.json'))
    fm_rmse.append(d['fm']['RMSE_mg']);    fm_r2.append(d['fm']['R2_z'])
    lin_rmse.append(d['linear']['RMSE_mg']); lin_r2.append(d['linear']['R2_z'])
    raw_rmse.append(d['raw']['RMSE_mg']);  raw_r2.append(d['raw']['R2_z'])

x = [4, 5, 6, 8]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.plot(x, fm_rmse,  color=C_BLUE,   marker='o', label='FM (zero-shot)')
ax.plot(x, lin_rmse, color=C_RED,    marker='s', label='Linear interp.')
ax.plot(x, raw_rmse, color=C_YELLOW, marker='^', label='Raw LSTM')
ax.set_xlabel('Gap duration (h)')
ax.set_ylabel('RMSE (mg/dL)')
ax.set_xticks(x)
ax.legend()

ax = axes[1]
ax.plot(x, fm_r2,  color=C_BLUE,   marker='o', label='FM (zero-shot)')
ax.plot(x, lin_r2, color=C_RED,    marker='s', label='Linear interp.')
ax.plot(x, raw_r2, color=C_YELLOW, marker='^', label='Raw LSTM')
ax.set_xlabel('Gap duration (h)')
ax.set_ylabel(r'$R^2$')
ax.set_xticks(x)
ax.legend()

plt.tight_layout(pad=1.5)
out = 'results/stage2/imputation/gn_run01/imputation_by_gap_clean.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved {out}')
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Forecasting RMSE vs horizon
# ═══════════════════════════════════════════════════════════════════════════════
model_specs = [
    ('naive',                'Naive',            C_BLACK,  '--', 'None'),
    ('raw_lstm',             'Raw LSTM',          C_RED,    '-',  'o'),
    ('fm_lstm',              'Encoder frozen',    C_LBLUE,  '-',  'None'),
    ('fm_ft_lstm',           'Encoder fine-tuned',C_BLUE,   '-',  'None'),
    ('fm_decoder_ft_lstm',   'Decoder fine-tuned',C_GREEN,  '-',  'None'),
]

base = 'results/stage2/forecasting/gn_run01'
horizons = list(range(24))
horizon_min = [(h + 1) * 5 for h in horizons]   # 5, 10, ..., 120

fig, ax = plt.subplots(figsize=(8, 5))

for tag, label, color, ls, marker in model_specs:
    d = json.load(open(f'{base}/metrics_{tag}.json'))
    rmse = [d[f'RMSE_h{h}'] for h in horizons]
    mk = None if marker == 'None' else marker
    ax.plot(horizon_min, rmse, color=color, linestyle=ls,
            marker=mk, markevery=4, label=label)

ax.set_xlabel('Forecast horizon (min)')
ax.set_ylabel('RMSE (mg/dL)')
ax.set_xticks([5, 30, 60, 90, 120])
ax.legend(loc='upper left')

plt.tight_layout(pad=1.5)
out = f'{base}/horizon_comparison_clean.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved {out}')
plt.close()

print('Done.')
