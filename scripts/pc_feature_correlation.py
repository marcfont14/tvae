"""
PC–feature correlation analysis.

1. Load cached encoder h_cls embeddings (1037, 128).
2. Load per-patient features: mean CGM, CV(CGM), mean PI, mean RA,
   bolus rate, carb rate (from patient .npz windows), plus GRI, TIR, TBR,
   TAR, CV-clinical, therapy modality (from clinical_vars cache).
3. PCA on h_cls → keep PCs explaining ≥90 % variance.
4. Spearman correlation between PC scores and features.
5. Save heatmap as results/pc_feature_correlation/pc_heatmap.pdf + .png.

Run from /mnt/workspace/tvae:
  python -u scripts/pc_feature_correlation.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EMBED_DIR  = 'results/embedding_study_global_norm'
DATA_DIR   = 'data/processed/adults_global_norm'
OUT_DIR    = 'results/pc_feature_correlation'
os.makedirs(OUT_DIR, exist_ok=True)

# Feature indices in windows array
IDX_CGM   = 0
IDX_PI    = 1
IDX_RA    = 2
IDX_BOLUS = 5
IDX_CARBS = 6


def load_per_patient_features(patient_ids):
    """Return DataFrame with per-patient statistics for all relevant input channels.

    Covered features (from the 10-channel input):
      0  CGM      → mean, std (glycaemic level + variability)
      1  PI       → mean (average plasma insulin)
      2  RA       → mean (average carb absorption)
      5  bolus    → rate (fraction of 5-min steps with a bolus event)
      6  carbs    → rate
      7  AID      → binary 1/0 (patient therapy modality)
      8  SAP      → binary 1/0
      9  MDI      → binary 1/0

    Time features (3,4 = hour_sin/cos) are per-window context variables whose
    per-patient mean is near zero for any patient with uniform day coverage;
    they are excluded from this patient-level analysis.
    """
    files = {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in glob.glob(os.path.join(DATA_DIR, '*.npz'))
    }

    rows = []
    for pid in patient_ids:
        path = files.get(pid)
        if path is None:
            rows.append({'patient_id': pid,
                         'mean_cgm': np.nan, 'cv_cgm': np.nan,
                         'mean_pi': np.nan, 'mean_ra': np.nan,
                         'bolus_rate': np.nan, 'carb_rate': np.nan,
                         'is_AID': np.nan, 'is_SAP': np.nan, 'is_MDI': np.nan})
            continue
        d = np.load(path, allow_pickle=True)
        w = d['windows']            # (N_windows, 288, 11)
        cgm = w[:, :, IDX_CGM].ravel()
        # Therapy modality: constant across all windows for a patient
        # Take the modal value of each one-hot column (should be identical everywhere)
        is_aid = float(w[0, 0, 7] > 0.5)
        is_sap = float(w[0, 0, 8] > 0.5)
        is_mdi = float(w[0, 0, 9] > 0.5)
        rows.append({
            'patient_id':  pid,
            'mean_cgm':    cgm.mean(),
            'cv_cgm':      cgm.std(),
            'mean_pi':     w[:, :, IDX_PI].mean(),
            'mean_ra':     w[:, :, IDX_RA].mean(),
            'bolus_rate':  w[:, :, IDX_BOLUS].mean(),
            'carb_rate':   w[:, :, IDX_CARBS].mean(),
            'is_AID':      is_aid,
            'is_SAP':      is_sap,
            'is_MDI':      is_mdi,
        })
    return pd.DataFrame(rows).set_index('patient_id')


def main():
    print('Loading embeddings and clinical vars...')
    enc = np.load(os.path.join(EMBED_DIR, 'encoder_embeddings.npy'))   # (1037, 128)
    cv  = np.load(os.path.join(EMBED_DIR, 'clinical_vars.npz'), allow_pickle=True)
    patient_ids = cv['patient_ids'].tolist()
    n = len(patient_ids)
    print(f'  {n} patients, embedding dim={enc.shape[1]}')

    print('Loading per-patient physiological features...')
    phys = load_per_patient_features(patient_ids)

    # Build full feature matrix (rows = patients, cols = features)
    # Input feature statistics (all 10 channels covered)
    df = phys[['mean_cgm', 'cv_cgm', 'mean_pi', 'mean_ra',
                'bolus_rate', 'carb_rate',
                'is_AID', 'is_SAP', 'is_MDI']].copy()
    df.rename(columns={
        'mean_cgm':   'Mean CGM (z)',
        'cv_cgm':     'Std CGM (z)',
        'mean_pi':    'Mean PI (z)',
        'mean_ra':    'Mean RA (z)',
        'bolus_rate': 'Bolus rate',
        'carb_rate':  'Carb rate',
        'is_AID':     'AID',
        'is_SAP':     'SAP',
        'is_MDI':     'MDI',
    }, inplace=True)
    # Also include glycaemic outcome targets for reference
    df['GRI']   = cv['gri']
    df['TIR']   = cv['tir']
    df['TBR']   = cv['tbr']

    df.dropna(inplace=True)
    enc_clean = enc[[patient_ids.index(p) for p in df.index], :]

    print(f'  {len(df)} patients after NA drop')

    # PCA
    print('Running PCA on encoder h_cls...')
    pca = PCA(n_components=enc_clean.shape[1])
    scores = pca.fit_transform(enc_clean)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_pcs  = int(np.searchsorted(cumvar, 0.90)) + 1
    print(f'  PCs to explain 90% variance: {n_pcs}')
    print(f'  Top-10 explained variance: ' +
          ', '.join(f'{v:.1%}' for v in pca.explained_variance_ratio_[:10]))

    scores_top = scores[:, :n_pcs]

    # Spearman correlation
    feat_cols = list(df.columns)
    rho = np.zeros((n_pcs, len(feat_cols)))
    pval = np.zeros((n_pcs, len(feat_cols)))
    for i in range(n_pcs):
        for j, col in enumerate(feat_cols):
            r, p = spearmanr(scores_top[:, i], df[col].values)
            rho[i, j] = r
            pval[i, j] = p

    rho_df  = pd.DataFrame(rho,  index=[f'PC{i+1}' for i in range(n_pcs)], columns=feat_cols)
    pval_df = pd.DataFrame(pval, index=rho_df.index, columns=feat_cols)

    print('\nSpearman ρ (PCs × features):')
    print(rho_df.round(3).to_string())

    # ── Therapy classification probe ──────────────────────────────────────────
    # Question: does the encoder already encode therapy modality implicitly
    # (inferred from PI/RA shape), making the explicit one-hot redundant?
    # Test: train a logistic regression on h_cls to predict therapy modality
    # without ever seeing the one-hot at train time.
    print('\n--- Therapy modality probe ---')
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    therapy_int = (df['AID'].values * 0 + df['SAP'].values * 1 + df['MDI'].values * 2).astype(int)
    # Remove patients with no clear modality (shouldn't happen)
    valid = therapy_int >= 0
    X_probe = enc_clean[valid]
    y_probe = therapy_int[valid]
    print(f'  Patients: {valid.sum()}  '
          f'AID={int((y_probe==0).sum())} SAP={int((y_probe==1).sum())} MDI={int((y_probe==2).sum())}')

    skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr    = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    accs  = cross_val_score(lr, X_probe, y_probe, cv=skf, scoring='balanced_accuracy')
    print(f'  5-fold balanced accuracy from h_cls: {accs.mean():.3f} ± {accs.std():.3f}')
    print(f'  (Chance = {1/len(np.unique(y_probe)):.3f})')
    print('  → If accuracy >> chance, encoder infers therapy from PI/RA pattern')
    print('    (explicit one-hot is then a shortcut, not strictly new information)\n')

    # Save CSV
    rho_df.to_csv(os.path.join(OUT_DIR, 'spearman_rho.csv'))
    pval_df.to_csv(os.path.join(OUT_DIR, 'spearman_pval.csv'))

    # Heatmap
    fig, ax = plt.subplots(figsize=(len(feat_cols) * 0.85 + 1.5, n_pcs * 0.6 + 1.5))
    mask_insig = pval_df.values > 0.05

    sns.heatmap(
        rho_df,
        ax=ax,
        cmap='RdBu_r',
        vmin=-1, vmax=1,
        center=0,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 8},
        linewidths=0.4,
        linecolor='#cccccc',
        mask=mask_insig,
        cbar_kws={'label': 'Spearman ρ', 'shrink': 0.7},
    )
    # Grey out non-significant cells
    sns.heatmap(
        rho_df,
        ax=ax,
        cmap=['#eeeeee'],
        vmin=-1, vmax=1,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 8, 'color': '#aaaaaa'},
        linewidths=0.4,
        linecolor='#cccccc',
        mask=~mask_insig,
        cbar=False,
    )

    pct = [f'{v:.1%}' for v in pca.explained_variance_ratio_[:n_pcs]]
    ax.set_yticklabels(
        [f'PC{i+1} ({pct[i]})' for i in range(n_pcs)],
        rotation=0, fontsize=9
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=9)
    ax.set_title(
        f'Spearman ρ: encoder h_cls PCs vs input features\n'
        f'(greyed = p > 0.05; top {n_pcs} PCs explain 90% variance)',
        fontsize=10, pad=10
    )

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'pc_heatmap.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'  Saved {path}')
    plt.close(fig)

    # Variance explained bar
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    xs = np.arange(1, n_pcs + 1)
    ax2.bar(xs, pca.explained_variance_ratio_[:n_pcs] * 100, color='steelblue', alpha=0.8)
    ax2.plot(xs, cumvar[:n_pcs] * 100, 'o-', color='firebrick', ms=4, label='Cumulative')
    ax2.axhline(90, color='firebrick', ls='--', lw=0.8)
    ax2.set_xlabel('Principal component')
    ax2.set_ylabel('Explained variance (%)')
    ax2.set_title('PCA scree plot — encoder h_cls')
    ax2.legend()
    ax2.set_xticks(xs)
    fig2.tight_layout()
    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'scree.{ext}')
        fig2.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'  Saved scree plots')


if __name__ == '__main__':
    main()
