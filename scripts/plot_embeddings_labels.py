"""
plot_embeddings_labels.py
=========================
UMAP of encoder h_cls and decoder mean-H, colored by HbA1c and ISF.
Checks whether embeddings show meaningful clinical stratification.

Uses cached embeddings from results/embedding_study/.
Labels cached to results/patient_level/labels_cache.npz on first run.

Output: results/patient_level/umap_labels.png  (2×2 grid)

Run inside docker from /mnt/workspace/tvae:
  python -u scripts/plot_embeddings_labels.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from umap import UMAP

# ── Paths ──────────────────────────────────────────────────────────────────────
EMB_DIR         = 'results/embedding_study'
OUT_DIR         = 'results/patient_level'
DATA_DIR        = 'data/processed/adults'
T1DEXI_LB       = 'data/raw/T1DEXI/LB.csv'
METABONET_TRAIN = 'data/raw/metabonet_train_filtered.parquet'
METABONET_TEST  = 'data/raw/metabonet_test_filtered.parquet'
CACHE_PATH      = os.path.join(OUT_DIR, 'labels_cache.npz')

SEED = 42
os.makedirs(OUT_DIR, exist_ok=True)


# ── Load embeddings ────────────────────────────────────────────────────────────

def load_cached():
    enc  = np.load(os.path.join(EMB_DIR, 'encoder_embeddings.npy'))
    dec  = np.load(os.path.join(EMB_DIR, 'decoder_embeddings.npy'))
    cv   = np.load(os.path.join(EMB_DIR, 'clinical_vars.npz'), allow_pickle=True)
    pids = cv['patient_ids'].astype(str)
    return enc, dec, pids


# ── Label derivation / caching ─────────────────────────────────────────────────

def load_hba1c():
    lb    = pd.read_csv(T1DEXI_LB)
    h     = lb[lb['LBTESTCD'] == 'HBA1C'][['USUBJID', 'LBSTRESN']].copy()
    h['USUBJID'] = h['USUBJID'].apply(lambda x: f'T_{int(x)}')
    return h.groupby('USUBJID')['LBSTRESN'].mean().to_dict()


def derive_isf():
    import pyarrow.parquet as pq
    print('  Deriving ISF (iter_batches)...')
    STEPS_90MIN = 18
    COLS = ['id', 'date', 'CGM', 'bolus', 'carbs']
    patient_rows = {}
    for path in [METABONET_TRAIN, METABONET_TEST]:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=100_000, columns=COLS):
            df = batch.to_pandas()
            for pid, grp in df.groupby('id'):
                pid = str(pid)
                if pid not in patient_rows:
                    patient_rows[pid] = []
                patient_rows[pid].append(grp)
            del df
    isf_records = []
    for pid, chunks in patient_rows.items():
        grp   = pd.concat(chunks).sort_values('date').reset_index(drop=True)
        cgm   = grp['CGM'].values
        bolus = grp['bolus'].values
        carbs = grp['carbs'].fillna(0).values
        for i in range(len(grp) - STEPS_90MIN):
            if bolus[i] <= 0 or carbs[i] > 0 or np.isnan(cgm[i]) or cgm[i] < 150:
                continue
            future = cgm[i+1 : i+1+STEPS_90MIN]
            valid  = future[~np.isnan(future)]
            if len(valid) < 6 or valid.min() >= cgm[i] - 10:
                continue
            est = (cgm[i] - valid.min()) / bolus[i]
            if 5 <= est <= 200:
                isf_records.append({'id': pid, 'isf': est})
    del patient_rows
    if not isf_records:
        return {}
    isf_df  = pd.DataFrame(isf_records)
    counts  = isf_df.groupby('id')['isf'].count()
    valid   = counts[counts >= 5].index
    result  = isf_df[isf_df['id'].isin(valid)].groupby('id')['isf'].median().to_dict()
    print(f'  ISF: {len(result)} patients')
    return result


def load_labels(pids):
    """Load from cache if available, otherwise derive and cache."""
    if os.path.exists(CACHE_PATH):
        print('  Loading labels from cache...')
        c = np.load(CACHE_PATH, allow_pickle=True)
        hba1c = dict(zip(c['hba1c_pids'].tolist(), c['hba1c_vals'].tolist()))
        isf   = dict(zip(c['isf_pids'].tolist(),   c['isf_vals'].tolist()))
        print(f'  HbA1c: {len(hba1c)} patients, ISF: {len(isf)} patients')
        return hba1c, isf

    print('  Deriving labels (first run — will cache)...')
    hba1c = load_hba1c()
    print(f'  HbA1c: {len(hba1c)} patients')
    isf   = derive_isf()

    np.savez(CACHE_PATH,
             hba1c_pids=np.array(list(hba1c.keys())),
             hba1c_vals=np.array(list(hba1c.values())),
             isf_pids=np.array(list(isf.keys())),
             isf_vals=np.array(list(isf.values())))
    print(f'  Labels cached to {CACHE_PATH}')
    return hba1c, isf


# ── UMAP ───────────────────────────────────────────────────────────────────────

def fit_umap(X, label=''):
    print(f'  Running UMAP on {X.shape}  [{label}]...')
    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                   metric='euclidean', random_state=SEED)
    return reducer.fit_transform(X)


# ── Single panel ───────────────────────────────────────────────────────────────

def scatter_panel(ax, umap2d, pids, label_dict, xlabel, ylabel, title, cmap='plasma'):
    pid_to_row = {p: i for i, p in enumerate(pids)}
    labeled    = [(pid_to_row[p], v) for p, v in label_dict.items() if p in pid_to_row]
    if not labeled:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center')
        return

    lab_idx  = np.array([r for r, _ in labeled])
    lab_vals = np.array([v for _, v in labeled])

    # unlabeled grey background
    all_idx      = np.arange(len(pids))
    unlabeled    = np.setdiff1d(all_idx, lab_idx)
    ax.scatter(umap2d[unlabeled, 0], umap2d[unlabeled, 1],
               c='#d0d0d0', s=4, alpha=0.4, linewidths=0, rasterized=True)

    sc = ax.scatter(umap2d[lab_idx, 0], umap2d[lab_idx, 1],
                    c=lab_vals, cmap=cmap, s=12, alpha=0.85,
                    linewidths=0, rasterized=True)
    plt.colorbar(sc, ax=ax, label=xlabel, pad=0.02)

    # Pearson / Spearman with UMAP axes
    r1, _  = pearsonr(umap2d[lab_idx, 0], lab_vals)
    r2, _  = pearsonr(umap2d[lab_idx, 1], lab_vals)
    rs1, _ = spearmanr(umap2d[lab_idx, 0], lab_vals)
    rs2, _ = spearmanr(umap2d[lab_idx, 1], lab_vals)
    best_r  = max(abs(r1), abs(r2), key=abs)
    best_rs = max(abs(rs1), abs(rs2), key=abs)

    info = (f'n={len(lab_idx)}\n'
            f'r={best_r:.3f}  ρ={best_rs:.3f}\n'
            f'[{lab_vals.min():.1f}, {lab_vals.max():.1f}]')
    ax.text(0.02, 0.98, info, transform=ax.transAxes, va='top', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel(ylabel + ' dim 1', fontsize=8)
    ax.set_ylabel(ylabel + ' dim 2', fontsize=8)
    ax.tick_params(labelsize=7)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print('\n=== Embedding UMAP — HbA1c & ISF ===\n')

    print('Loading embeddings...')
    enc_embs, dec_embs, pids = load_cached()
    print(f'  Encoder: {enc_embs.shape}, Decoder: {dec_embs.shape}')

    print('\nLoading labels...')
    hba1c_dict, isf_dict = load_labels(pids)

    print('\nFitting UMAP projections...')
    umap_enc = fit_umap(enc_embs, 'encoder')
    umap_dec = fit_umap(dec_embs, 'decoder')

    print('\nPlotting...')
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    scatter_panel(axes[0, 0], umap_enc, pids, hba1c_dict,
                  'HbA1c (%)', 'Encoder UMAP',
                  'Encoder h_cls — HbA1c', cmap='RdYlGn_r')
    scatter_panel(axes[0, 1], umap_enc, pids, isf_dict,
                  'ISF (mg/dL/U)', 'Encoder UMAP',
                  'Encoder h_cls — ISF', cmap='viridis')
    scatter_panel(axes[1, 0], umap_dec, pids, hba1c_dict,
                  'HbA1c (%)', 'Decoder UMAP',
                  'Decoder mean-H — HbA1c', cmap='RdYlGn_r')
    scatter_panel(axes[1, 1], umap_dec, pids, isf_dict,
                  'ISF (mg/dL/U)', 'Decoder UMAP',
                  'Decoder mean-H — ISF', cmap='viridis')

    fig.suptitle('UMAP embeddings colored by clinical labels\n'
                 '(grey = unlabeled patients)', fontsize=12, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, 'umap_labels.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {out_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
