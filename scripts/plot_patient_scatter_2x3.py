"""
Patient-level 2×3 scatter: ISF and CR vs Encoder h_cls / Decoder H / CGM+PI+RA curves.
Uses all patients with 5-fold CV for reported R²; full-fit scatter for visual.
Approved thesis aesthetic.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/mnt/workspace/tvae')
os.chdir('/mnt/workspace/tvae')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline

# ── Paths ──────────────────────────────────────────────────────────────────────
EMB_DIR  = 'results/embedding_study_global_norm'
OUT_DIR  = 'results/patient_level_global_norm'
DATA_DIR = 'data/processed/adults_global_norm'

METABONET_TRAIN = 'data/raw/metabonet_train_filtered.parquet'
METABONET_TEST  = 'data/raw/metabonet_test_filtered.parquet'

GLOBAL_CGM_MEAN = 144.40
GLOBAL_CGM_STD  = 57.11
CGM_IDX, PI_IDX, RA_IDX = 0, 1, 2
SEED = 42
np.random.seed(SEED)

# ── rcParams ───────────────────────────────────────────────────────────────────
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
    'lines.linewidth':  1.0,
    'lines.markersize': 4,
})

COL_COLORS = ['#0072BD', '#77AC30', '#EDB120']   # Enc / Dec / CGM+PI+RA


# ── Load embeddings ────────────────────────────────────────────────────────────
print('Loading embeddings...')
enc_embs = np.load(f'{EMB_DIR}/encoder_embeddings.npy')   # (1037, 128)
dec_embs = np.load(f'{EMB_DIR}/decoder_embeddings.npy')   # (1037, 128)
cv       = np.load(f'{EMB_DIR}/clinical_vars.npz', allow_pickle=True)
pids     = cv['patient_ids'].astype(str)
print(f'  Enc {enc_embs.shape}, Dec {dec_embs.shape}, {len(pids)} patients')


# ── CGM+PI+RA curves feature vector ───────────────────────────────────────────
def compute_full_stats(pids):
    stats = {}
    for pid in pids:
        fpath = f'{DATA_DIR}/{pid}.npz'
        if not os.path.exists(fpath):
            continue
        try:
            d = np.load(fpath)
            windows = d['windows']
            cgm_z   = windows[:, :, CGM_IDX].ravel()
            cgm_mg  = cgm_z * GLOBAL_CGM_STD + GLOBAL_CGM_MEAN
            tir = float(((cgm_mg >= 70) & (cgm_mg <= 180)).mean())
            tar = float((cgm_mg > 180).mean())
            tbr = float((cgm_mg < 70).mean())
            cv_val = float(cgm_mg.std() / (cgm_mg.mean() + 1e-8))
            cgm_feats = np.array([cgm_mg.mean(), cgm_mg.std(), tir, tar, tbr, cv_val],
                                  dtype=np.float32)
            mean_pi = windows[:, :, PI_IDX].mean(axis=0).astype(np.float32)
            mean_ra = windows[:, :, RA_IDX].mean(axis=0).astype(np.float32)
            stats[pid] = np.concatenate([cgm_feats, mean_pi, mean_ra])
        except Exception:
            pass
    return stats


# ── Label derivation ───────────────────────────────────────────────────────────
def derive_cr():
    dfs = []
    for path in [METABONET_TRAIN, METABONET_TEST]:
        dfs.append(pd.read_parquet(path, columns=['id', 'bolus', 'carbs']))
    df   = pd.concat(dfs, ignore_index=True)
    meal = df[(df['bolus'] > 0) & (df['carbs'] > 0)].copy()
    meal['cr'] = meal['carbs'] / meal['bolus']
    meal = meal[(meal['cr'] >= 2) & (meal['cr'] <= 50)]
    counts = meal.groupby('id')['cr'].count()
    meal   = meal[meal['id'].isin(counts[counts >= 5].index)]
    return {str(k): v for k, v in meal.groupby('id')['cr'].median().items()}


def derive_isf():
    import pyarrow.parquet as pq
    records = []
    STEPS_90 = 18
    patient_rows = {}
    for path in [METABONET_TRAIN, METABONET_TEST]:
        for batch in pq.ParquetFile(path).iter_batches(
                batch_size=100_000, columns=['id', 'date', 'CGM', 'bolus', 'carbs']):
            df = batch.to_pandas()
            for pid, grp in df.groupby('id'):
                patient_rows.setdefault(str(pid), []).append(grp)
    for pid, chunks in patient_rows.items():
        grp   = pd.concat(chunks).sort_values('date').reset_index(drop=True)
        cgm   = grp['CGM'].values
        bolus = grp['bolus'].values
        carbs = grp['carbs'].fillna(0).values
        for i in range(len(grp) - STEPS_90):
            if bolus[i] <= 0 or carbs[i] > 0:
                continue
            if np.isnan(cgm[i]) or cgm[i] < 150:
                continue
            fut = cgm[i+1:i+1+STEPS_90]
            valid = fut[~np.isnan(fut)]
            if len(valid) < 6 or valid.min() >= cgm[i] - 10:
                continue
            est = (cgm[i] - valid.min()) / bolus[i]
            if 5 <= est <= 200:
                records.append({'id': pid, 'isf': est})
    if not records:
        return {}
    df2    = pd.DataFrame(records)
    counts = df2.groupby('id')['isf'].count()
    df2    = df2[df2['id'].isin(counts[counts >= 5].index)]
    return df2.groupby('id')['isf'].median().to_dict()


# ── Ridge helpers ──────────────────────────────────────────────────────────────
def make_pipe(n_pca=None):
    steps = [StandardScaler()]
    if n_pca is not None:
        steps.append(PCA(n_components=n_pca, random_state=SEED))
    steps.append(RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5))
    return make_pipeline(*steps)

def cv_r2(X, y, n_pca=None):
    p = make_pipe(n_pca)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(p, X, y, cv=kf, scoring='r2')
    return float(scores.mean())

def full_pred(X, y, n_pca=None):
    p = make_pipe(n_pca)
    p.fit(X, y)
    return p.predict(X)


# ── Compute stats ──────────────────────────────────────────────────────────────
print('Computing CGM+PI+RA stats...')
full_stats = compute_full_stats(pids)
print(f'  {len(full_stats)} patients')

print('Deriving CR...')
cr_dict = derive_cr()
print(f'  {len(cr_dict)} patients')

print('Deriving ISF...')
isf_dict = derive_isf()
print(f'  {len(isf_dict)} patients')

pid_idx = {p: i for i, p in enumerate(pids)}


# ── Build figure ───────────────────────────────────────────────────────────────
ROWS = [
    ('ISF (mg/dL/U)', isf_dict),
    ('CR (g/U)',      cr_dict),
]
COLS = [
    ('Encoder $h_{\\mathrm{cls}}$', lambda idx: enc_embs[idx],  None),
    ('Decoder $H$',                  lambda idx: dec_embs[idx],  None),
    ('CGM + PI + RA curves',         lambda idx: np.stack([full_stats[p] for p in common]),  32),
]

fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))

for row_i, (tgt_name, label_dict) in enumerate(ROWS):
    common = [p for p in pids if p in label_dict and p in full_stats]
    idx    = [pid_idx[p] for p in common]
    y      = np.array([label_dict[p] for p in common])
    print(f'\n{tgt_name}: {len(common)} patients')

    X_enc  = enc_embs[idx]
    X_dec  = dec_embs[idx]
    X_full = np.stack([full_stats[p] for p in common])

    Xs     = [X_enc, X_dec, X_full]
    n_pcas = [None,  None,  32]
    labels = ['Encoder $h_{\\mathrm{cls}}$',
              'Decoder $\\bar{H}$',
              'CGM + PI + RA curves']

    for col_i, (X, n_pca, col_label) in enumerate(zip(Xs, n_pcas, labels)):
        ax = axes[row_i, col_i]
        n_pca_actual = None if n_pca is None else min(n_pca, X.shape[0]-1, X.shape[1])

        r2  = cv_r2(X, y, n_pca=n_pca_actual)
        yp  = full_pred(X, y, n_pca=n_pca_actual)
        print(f'  {col_label}: CV R²={r2:.3f}')

        vmin = min(y.min(), yp.min())
        vmax = max(y.max(), yp.max())
        pad  = (vmax - vmin) * 0.05

        ax.scatter(y, yp, s=18, alpha=0.55, color=COL_COLORS[col_i],
                   linewidths=0, zorder=3)
        ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad],
                color='#333333', lw=0.9, ls='--', zorder=4)
        ax.set_xlim(vmin - pad, vmax + pad)
        ax.set_ylim(vmin - pad, vmax + pad)
        ax.set_aspect('equal')

        unit = 'mg/dL/U' if 'ISF' in tgt_name else 'g/U'
        ax.set_xlabel(f'Actual {tgt_name}')
        ax.set_ylabel(f'Predicted {tgt_name}')
        ax.set_title(f'{col_label}\n$R^2={r2:.3f}$ (5-fold CV)   $n={len(common)}$',
                     fontsize=9)

plt.tight_layout(pad=1.5, w_pad=2.5, h_pad=3.0)
out = f'{OUT_DIR}/patient_scatter_2x3.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'\nSaved {out}')
