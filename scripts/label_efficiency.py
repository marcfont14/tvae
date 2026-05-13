"""
label_efficiency.py
===================
HbA1c label efficiency curve: how many labelled patients does each
feature set need to match (or beat) CGM statistics?

Uses cached encoder embeddings from results/embedding_study/.
HbA1c labels loaded from T1DEXI LB.csv (fast CSV read, no parquet).
CGM stats recomputed from .npz windows.

Outputs: results/patient_level/label_efficiency_hba1c.png

Run inside docker from /mnt/workspace/tvae:
  python -u scripts/label_efficiency.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

# ── Paths ──────────────────────────────────────────────────────────────────────
EMB_DIR   = 'results/embedding_study'
OUT_DIR   = 'results/patient_level'
DATA_DIR  = 'data/processed/adults'
T1DEXI_LB = 'data/raw/T1DEXI/LB.csv'

GLOBAL_CGM_MEAN = 144.40
GLOBAL_CGM_STD  = 57.11
CGM_IDX = 0
SEED    = 42

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(SEED)


def load_cached():
    enc  = np.load(os.path.join(EMB_DIR, 'encoder_embeddings.npy'))
    cv   = np.load(os.path.join(EMB_DIR, 'clinical_vars.npz'), allow_pickle=True)
    pids = cv['patient_ids'].astype(str)
    return enc, pids


def load_hba1c():
    lb     = pd.read_csv(T1DEXI_LB)
    hba1c  = lb[lb['LBTESTCD'] == 'HBA1C'][['USUBJID', 'LBSTRESN']].copy()
    hba1c['USUBJID'] = hba1c['USUBJID'].apply(lambda x: f'T_{int(x)}')
    result = hba1c.groupby('USUBJID')['LBSTRESN'].mean().to_dict()
    print(f'  HbA1c: {len(result)} T1DEXI patients, '
          f'range {min(result.values()):.1f}–{max(result.values()):.1f}%')
    return result


def compute_cgm_stats(pids):
    stats = {}
    for pid in pids:
        fpath = os.path.join(DATA_DIR, f'{pid}.npz')
        if not os.path.exists(fpath):
            continue
        try:
            d      = np.load(fpath)
            cgm_z  = d['windows'][:, :, CGM_IDX].ravel()
            cgm_mg = cgm_z * GLOBAL_CGM_STD + GLOBAL_CGM_MEAN
            tir = float(((cgm_mg >= 70) & (cgm_mg <= 180)).mean())
            tar = float((cgm_mg > 180).mean())
            tbr = float((cgm_mg < 70).mean())
            cv  = float(cgm_mg.std() / (cgm_mg.mean() + 1e-8))
            stats[pid] = np.array([cgm_mg.mean(), cgm_mg.std(), tir, tar, tbr, cv],
                                   dtype=np.float32)
        except Exception:
            pass
    return stats


def _make_pipe(n_pca, n_train):
    steps = [StandardScaler()]
    if n_pca is not None:
        n_pca_actual = min(n_pca, n_train - 1)
        if n_pca_actual >= 1:
            steps.append(PCA(n_components=n_pca_actual, random_state=SEED))
    steps.append(RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
                         cv=min(5, n_train)))
    return make_pipeline(*steps)


def run_experiment(enc_embs, cgm_stats, hba1c_dict, pids):
    common      = [p for p in pids if p in hba1c_dict and p in cgm_stats]
    N_TOTAL     = len(common)
    N_TEST      = min(100, N_TOTAL // 4)
    N_TRAIN_MAX = N_TOTAL - N_TEST
    print(f'  Aligned: {N_TOTAL}  |  test holdout: {N_TEST}  |  max train: {N_TRAIN_MAX}')

    idx_map    = {p: i for i, p in enumerate(pids)}
    idx        = [idx_map[p] for p in common]
    y_all      = np.array([hba1c_dict[p] for p in common])
    X_enc      = enc_embs[idx]
    X_cgm      = np.stack([cgm_stats[p] for p in common])

    rng_test   = np.random.RandomState(SEED)
    test_idx   = rng_test.choice(N_TOTAL, N_TEST, replace=False)
    train_pool = np.array([i for i in range(N_TOTAL) if i not in set(test_idx)])

    N_VALUES = sorted(set(
        n for n in [5, 10, 20, 50, 100, 200, 300, N_TRAIN_MAX]
        if n <= len(train_pool)
    ))
    K = 50

    features = [
        ('Encoder h_cls',  X_enc, None),
        ('Encoder+PCA-10', X_enc,   10),
        ('CGM stats',      X_cgm, None),
    ]
    curves = {name: {'n': [], 'mean': [], 'std': []} for name, _, _ in features}

    for n in N_VALUES:
        per_feat = {name: [] for name, _, _ in features}
        for k in range(K):
            rng_k     = np.random.RandomState(SEED * 1000 + k)
            train_idx = rng_k.choice(train_pool, n, replace=False)

            for feat_name, X_feat, n_pca in features:
                X_tr = X_feat[train_idx];  y_tr = y_all[train_idx]
                X_te = X_feat[test_idx];   y_te = y_all[test_idx]
                try:
                    pipe = _make_pipe(n_pca, n)
                    pipe.fit(X_tr, y_tr)
                    y_pred = pipe.predict(X_te)
                    ss_res = ((y_te - y_pred) ** 2).sum()
                    ss_tot = ((y_te - y_te.mean()) ** 2).sum()
                    per_feat[feat_name].append(1.0 - ss_res / (ss_tot + 1e-8))
                except Exception:
                    pass

        line = f'  N={n:4d}:'
        for feat_name, _, _ in features:
            vals = per_feat[feat_name]
            if vals:
                m, s = float(np.mean(vals)), float(np.std(vals))
                curves[feat_name]['n'].append(n)
                curves[feat_name]['mean'].append(m)
                curves[feat_name]['std'].append(s)
                line += f'  {feat_name.split()[0]}={m:.3f}'
        print(line)

    return curves, N_TEST, K


def plot_curves(curves, N_TEST, K):
    colors = {'Encoder h_cls': '#2196F3', 'Encoder+PCA-10': '#9C27B0', 'CGM stats': '#4CAF50'}
    fig, ax = plt.subplots(figsize=(8, 5))
    for feat_name, c in curves.items():
        if not c['n']:
            continue
        ns   = np.array(c['n'])
        mean = np.array(c['mean'])
        std  = np.array(c['std'])
        col  = colors.get(feat_name, 'gray')
        ax.plot(ns, mean, marker='o', label=feat_name, color=col)
        ax.fill_between(ns, mean - std, mean + std, alpha=0.15, color=col)

    # Annotate first crossover: encoder+PCA-10 beats CGM stats
    enc_pca = curves.get('Encoder+PCA-10', {})
    cgm_c   = curves.get('CGM stats', {})
    if enc_pca.get('n') and cgm_c.get('n') and enc_pca['n'] == cgm_c['n']:
        for n_val, e, c_ in zip(enc_pca['n'], enc_pca['mean'], cgm_c['mean']):
            if e > c_:
                ax.axvline(n_val, color='#9C27B0', lw=0.8, linestyle='--', alpha=0.6)
                ax.text(n_val * 1.05, ax.get_ylim()[0] + 0.01,
                        f'N={n_val}', color='#9C27B0', fontsize=9)
                break

    ax.set_xscale('log')
    ax.set_xlabel('Training patients (N)', fontsize=12)
    ax.set_ylabel('R² on held-out test', fontsize=12)
    ax.set_title(f'HbA1c Label Efficiency  (K={K} reps, N_test={N_TEST})',
                 fontweight='bold')
    ax.legend()
    ax.axhline(0, color='black', lw=0.5, linestyle='--')
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(OUT_DIR, 'label_efficiency_hba1c.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  Saved: {out_path}')


def main():
    print('\n=== HbA1c Label Efficiency ===\n')

    print('Loading cached encoder embeddings...')
    enc_embs, pids = load_cached()
    print(f'  Encoder: {enc_embs.shape}, Patients: {len(pids)}')

    print('\nLoading HbA1c labels...')
    hba1c_dict = load_hba1c()

    print('\nComputing CGM stats...')
    cgm_stats = compute_cgm_stats(pids)
    print(f'  CGM stats: {len(cgm_stats)} patients')

    print('\nRunning experiment...')
    curves, N_TEST, K = run_experiment(enc_embs, cgm_stats, hba1c_dict, pids)

    plot_curves(curves, N_TEST, K)
    print('\nDone.')


if __name__ == '__main__':
    main()
