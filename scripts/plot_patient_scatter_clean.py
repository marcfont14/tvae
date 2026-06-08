"""
Clean patient-level scatter: 3 panels, best representation per target.
Imports label derivation directly from patient_level_analysis.py.
"""
import os, sys
sys.path.insert(0, '/mnt/workspace/tvae')
os.chdir('/mnt/workspace/tvae')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Import data loading functions from the existing analysis script
import importlib.util
spec = importlib.util.spec_from_file_location(
    'pla', '/mnt/workspace/tvae/scripts/patient_level_analysis.py')
pla = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pla)

SEED = 42
np.random.seed(SEED)

# ── Load embeddings and labels ────────────────────────────────────────────────
print('Loading embeddings...')
enc, dec, pids, cv = pla.load_cached()

print('Computing CGM stats...')
stats_dict = pla.compute_cgm_stats(pids)
cgm_stats_arr = np.array([stats_dict.get(p, np.zeros(6)) for p in pids])

print('Loading labels...')
hba1c_dict = pla.load_hba1c()
cr_dict    = pla.derive_cr()
isf_dict   = pla.derive_isf()

print(f'  HbA1c: {len(hba1c_dict)}, CR: {len(cr_dict)}, ISF: {len(isf_dict)}')

# ── Train/test split ──────────────────────────────────────────────────────────
with open('results/outlier_analysis/pretrain_patients.txt') as f:
    pretrain_set = set(os.path.basename(l.strip()) for l in f if l.strip())
train_mask = np.array([p + '.npz' in pretrain_set for p in pids])

def probe(X, y_dict):
    valid = np.array([p in y_dict for p in pids])
    X_v   = X[valid]
    y_v   = np.array([y_dict[p] for p in pids[valid]])
    tr    = train_mask[valid]
    pipe  = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 20)))
    pipe.fit(X_v[tr], y_v[tr])
    y_pred = pipe.predict(X_v[~tr])
    y_true = y_v[~tr]
    ss_res = ((y_true - y_pred)**2).sum()
    ss_tot = ((y_true - y_true.mean())**2).sum()
    return y_true, y_pred, 1 - ss_res / ss_tot

print('Running probes...')
yt_isf,  yp_isf,  r2_isf  = probe(dec,           isf_dict)
yt_cr,   yp_cr,   r2_cr   = probe(enc,            cr_dict)
yt_hba,  yp_hba,  r2_hba  = probe(cgm_stats_arr,  hba1c_dict)
print(f'  ISF R²={r2_isf:.3f}  CR R²={r2_cr:.3f}  HbA1c R²={r2_hba:.3f}')

# ── Plot ──────────────────────────────────────────────────────────────────────
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
    'axes.grid':        True,
    'grid.color':       '#cccccc',
    'grid.linewidth':   0.5,
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    'xtick.top':        True,
    'ytick.right':      True,
})
C_BLUE = '#0072BD'

panels = [
    (yt_isf,  yp_isf,  r2_isf,  'ISF (mg/dL/U)', 'Decoder $\\overline{H}$'),
    (yt_cr,   yp_cr,   r2_cr,   'CR (g/U)',       'Encoder $h_{cls}$'),
    (yt_hba,  yp_hba,  r2_hba,  'HbA1c (%)',      'CGM statistics'),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, (yt, yp, r2, ylabel, repr_label) in zip(axes, panels):
    lo = min(yt.min(), yp.min()) * 0.95
    hi = max(yt.max(), yp.max()) * 1.05
    ax.scatter(yt, yp, s=12, alpha=0.55, color=C_BLUE, linewidths=0)
    ax.plot([lo, hi], [lo, hi], color='#D95319', lw=1.2, ls='--')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f'Actual {ylabel}')
    ax.set_ylabel(f'Predicted {ylabel}')
    ax.set_title(f'{repr_label}   $R^2={r2:.3f}$,  $n={len(yt)}$', fontsize=9.5)

plt.tight_layout(pad=1.5, w_pad=2.0)
out = 'results/patient_level_global_norm/patient_scatter_clean.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved {out}')
