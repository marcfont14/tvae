"""
Plot feature importance summary figures from already-computed ablation results.

Two figures:
  1. Grouped bar chart — R² drop per feature group across 4 gap lengths.
  2. PC–feature heatmap (updated: therapy columns removed, 7 active features only).

Run from /mnt/workspace/tvae:
  python -u scripts/plot_feature_importance.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR   = 'results/feature_importance'
EMBED_DIR = 'results/embedding_study_global_norm'
DATA_DIR  = 'data/processed/adults_global_norm'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Figure 1: Ablation R² drop ────────────────────────────────────────────────

# Pre-computed from feature_ablation.py (no_therapy excluded — feature discarded)
GAPS = ['4h', '5h', '6h', '8h']
ABLATIONS = {
    'PI':          [-0.1038, -0.1165, -0.1466, -0.2296],
    'RA':          [-0.0509, -0.0606, -0.0781, -0.0909],
    'Bolus & carbs\nflags': [-0.0230, -0.0300, -0.0315, -0.0388],
    'Time\n(circ.)':  [-0.0082, -0.0098, -0.0099, -0.0109],
}

COLORS = {
    'PI':   '#1d4ed8',
    'RA':   '#2563eb',
    'Bolus & carbs\nflags': '#16a34a',
    'Time\n(circ.)':  '#ca8a04',
}

def plot_ablation():
    features  = list(ABLATIONS.keys())
    n_feat    = len(features)
    n_gap     = len(GAPS)
    bar_w     = 0.18
    x         = np.arange(n_gap)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    offsets = np.linspace(-(n_feat - 1) / 2, (n_feat - 1) / 2, n_feat) * bar_w
    for i, (feat, vals) in enumerate(ABLATIONS.items()):
        bars = ax.bar(x + offsets[i], [-v for v in vals],   # flip sign: show as positive drop
                      bar_w, label=feat,
                      color=COLORS[feat], alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{g} gap' for g in GAPS], fontsize=10)
    ax.set_ylabel('R² drop (absolute)', fontsize=10)
    ax.set_title('Feature importance — zero-shot imputation\n'
                 '(R² drop when feature group is zeroed out)', fontsize=10)
    ax.legend(loc='upper left', fontsize=8.5, framealpha=0.85)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 0.27)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))

    # Annotate max value on each cluster
    for i, (feat, vals) in enumerate(ABLATIONS.items()):
        for j, v in enumerate(vals):
            ax.text(x[j] + offsets[i], -v + 0.004, f'{-v:.2f}',
                    ha='center', va='bottom', fontsize=6.5, color='#333333')

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'ablation_r2_drop.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved {path}')
    plt.close(fig)


# ── Figure 2: PC–feature heatmap (7 active features, no therapy) ─────────────

def plot_pc_heatmap():
    import glob
    from sklearn.decomposition import PCA

    IDX_CGM, IDX_PI, IDX_RA = 0, 1, 2
    IDX_BOLUS, IDX_CARBS    = 5, 6

    print('Loading embeddings...')
    enc = np.load(os.path.join(EMBED_DIR, 'encoder_embeddings.npy'))
    cv  = np.load(os.path.join(EMBED_DIR, 'clinical_vars.npz'), allow_pickle=True)
    patient_ids = cv['patient_ids'].tolist()

    print('Loading per-patient feature statistics...')
    files = {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in glob.glob(os.path.join(DATA_DIR, '*.npz'))
    }

    IDX_HSIN, IDX_HCOS = 3, 4

    rows = []
    for pid in patient_ids:
        path = files.get(pid)
        if path is None:
            rows.append({k: np.nan for k in ['mean_cgm', 'mean_pi', 'mean_ra',
                                               'night_frac', 'bolus_rate', 'carb_rate']})
            continue
        d = np.load(path, allow_pickle=True)
        w = d['windows']                          # (N_windows, 288, 10)

        # Per-patient mean of each model input channel (cols 0–6)
        # Cols 0,1,2,5,6: continuous/binary → mean is meaningful
        # Cols 3,4 (hour_sin/cos): per-patient mean ≈ 0 for uniform day coverage.
        #   Instead, compute the fraction of window-timesteps in night/dawn (20:00–08:00),
        #   derived from hour_cos < 0 OR hour < 8 proxy. Using arctan2:
        #   hour = arctan2(sin, cos) * 24 / (2π) — values in [0,24)
        hsin = w[:, :, IDX_HSIN]
        hcos = w[:, :, IDX_HCOS]
        hour = (np.arctan2(hsin, hcos) / (2 * np.pi) * 24) % 24
        night_frac = float(((hour >= 20) | (hour < 8)).mean())

        rows.append({
            'mean_cgm':   float(w[:, :, IDX_CGM].mean()),
            'mean_pi':    float(w[:, :, IDX_PI].mean()),
            'mean_ra':    float(w[:, :, IDX_RA].mean()),
            'night_frac': night_frac,
            'bolus_rate': float(w[:, :, IDX_BOLUS].mean()),
            'carb_rate':  float(w[:, :, IDX_CARBS].mean()),
        })

    feat_df = pd.DataFrame(rows, index=patient_ids).dropna()
    enc_clean = enc[[patient_ids.index(p) for p in feat_df.index], :]

    # Column labels matching the 7 model inputs (time cols 3–4 summarised as night fraction)
    feat_df.rename(columns={
        'mean_cgm':   'Mean CGM (col 0)',
        'mean_pi':    'Mean PI (col 1)',
        'mean_ra':    'Mean RA (col 2)',
        'night_frac': 'Night frac\n(cols 3–4)',
        'bolus_rate': 'Bolus rate\n(col 5)',
        'carb_rate':  'Carb rate\n(col 6)',
    }, inplace=True)

    print('PCA...')
    pca    = PCA(n_components=enc_clean.shape[1])
    scores = pca.fit_transform(enc_clean)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_pcs  = int(np.searchsorted(cumvar, 0.90)) + 1
    print(f'  PCs for 90% variance: {n_pcs}')

    scores_top = scores[:, :n_pcs]
    feat_cols  = list(feat_df.columns)

    rho  = np.zeros((n_pcs, len(feat_cols)))
    pval = np.zeros((n_pcs, len(feat_cols)))
    for i in range(n_pcs):
        for j, col in enumerate(feat_cols):
            r, p = spearmanr(scores_top[:, i], feat_df[col].values)
            rho[i, j]  = r
            pval[i, j] = p

    pct   = [f'{v:.1%}' for v in pca.explained_variance_ratio_[:n_pcs]]
    rows_ = [f'PC{i+1} ({pct[i]})' for i in range(n_pcs)]
    rho_df  = pd.DataFrame(rho,  index=rows_, columns=feat_cols)
    pval_df = pd.DataFrame(pval, index=rows_, columns=feat_cols)

    # Heatmap
    fig, ax = plt.subplots(figsize=(len(feat_cols) * 1.0 + 1.5, n_pcs * 0.65 + 1.8))
    mask_ns = pval_df.values > 0.05

    sns.heatmap(rho_df, ax=ax, cmap='RdBu_r', vmin=-1, vmax=1, center=0,
                annot=True, fmt='.2f', annot_kws={'size': 9},
                linewidths=0.4, linecolor='#cccccc',
                mask=mask_ns,
                cbar_kws={'label': 'Spearman ρ', 'shrink': 0.7})
    sns.heatmap(rho_df, ax=ax, cmap=['#eeeeee'], vmin=-1, vmax=1,
                annot=True, fmt='.2f', annot_kws={'size': 9, 'color': '#aaaaaa'},
                linewidths=0.4, linecolor='#cccccc',
                mask=~mask_ns, cbar=False)

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
    ax.set_title('Spearman ρ: encoder PCs vs per-patient summaries of the 7 model inputs\n'
                 '(time cols 3–4 as night/dawn fraction; greyed = p > 0.05)', fontsize=9, pad=10)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'pc_feature_heatmap.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved {path}')
    plt.close(fig)

    # Print table
    print('\nSpearman ρ (significant only, p<0.05):')
    print(rho_df.round(3).to_string())


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=== Feature importance plots ===\n')
    plot_ablation()
    print()
    plot_pc_heatmap()
    print('\nDone.')
