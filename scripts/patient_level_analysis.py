"""
patient_level_analysis.py
=========================
Patient-level probe study: do encoder embeddings capture patient-specific
physiological parameters better than raw CGM statistics?

Three targets:
  HbA1c  — lab measurement, 154 T1DEXI patients (ground-truth label)
  CR     — derived: carbs/bolus at meal events, METABONET (interpretability probe)
  ISF    — derived: ΔBG/bolus at correction events, METABONET (interpretability probe)

For each target, compares three feature sets via 5-fold Ridge CV:
  A: encoder h_cls (128-d, mean-pooled across patient windows)
  B: decoder mean-pool H (128-d)
  C: raw CGM stats (mean, std, TIR, TAR, TBR, CV)

Uses cached embeddings from results/embedding_study/.

Run inside docker from /mnt/workspace/tvae:
  python -u scripts/patient_level_analysis.py 2>&1 | tee results/patient_level/log.txt
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
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline

# ── Paths ──────────────────────────────────────────────────────────────────────
EMB_DIR       = 'results/embedding_study_global_norm'
OUT_DIR       = 'results/patient_level_global_norm'
DATA_DIR      = 'data/processed/adults_global_norm'

METABONET_TRAIN = 'data/raw/metabonet_train_filtered.parquet'
METABONET_TEST  = 'data/raw/metabonet_test_filtered.parquet'
T1DEXI_PARQUET  = 'data/raw/t1dexi_parsed.parquet'
T1DEXI_LB       = 'data/raw/T1DEXI/LB.csv'

GLOBAL_CGM_MEAN = 144.40
GLOBAL_CGM_STD  = 57.11
CGM_IDX = 0
PI_IDX  = 1
RA_IDX  = 2

SEED = 42
np.random.seed(SEED)

os.makedirs(OUT_DIR, exist_ok=True)


# ── Load cached embeddings ─────────────────────────────────────────────────────

def load_cached():
    enc = np.load(os.path.join(EMB_DIR, 'encoder_embeddings.npy'))   # (1037, 128)
    dec = np.load(os.path.join(EMB_DIR, 'decoder_embeddings.npy'))   # (1037, 128)
    cv  = np.load(os.path.join(EMB_DIR, 'clinical_vars.npz'), allow_pickle=True)
    pids = cv['patient_ids'].astype(str)   # (1037,)
    return enc, dec, pids, cv


# ── Raw CGM stats per patient (from .npz windows) ─────────────────────────────

def compute_cgm_stats(pids):
    """Return dict pid -> (mean_mg, std_mg, tir, tar, tbr, cv) from .npz files."""
    stats = {}
    for pid in pids:
        fpath = os.path.join(DATA_DIR, f'{pid}.npz')
        if not os.path.exists(fpath):
            continue
        try:
            d = np.load(fpath)
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


def compute_cgm_pi_ra_stats(pids):
    """
    Return dict pid -> feature vector:
      CGM stats (6) + mean PI curve (288,) + mean RA curve (288,)  =  582-d.
    Mean curves are averaged across all windows, giving the patient's typical
    24h PI/RA profile — same temporal information the transformer sees.
    Ridge+PCA is applied downstream to handle the dimensionality.
    """
    stats = {}
    for pid in pids:
        fpath = os.path.join(DATA_DIR, f'{pid}.npz')
        if not os.path.exists(fpath):
            continue
        try:
            d = np.load(fpath)
            windows = d['windows']                        # (N, 288, 11)
            cgm_z  = windows[:, :, CGM_IDX].ravel()
            cgm_mg = cgm_z * GLOBAL_CGM_STD + GLOBAL_CGM_MEAN
            tir = float(((cgm_mg >= 70) & (cgm_mg <= 180)).mean())
            tar = float((cgm_mg > 180).mean())
            tbr = float((cgm_mg < 70).mean())
            cv  = float(cgm_mg.std() / (cgm_mg.mean() + 1e-8))
            cgm_feats = np.array([cgm_mg.mean(), cgm_mg.std(), tir, tar, tbr, cv],
                                  dtype=np.float32)
            mean_pi = windows[:, :, PI_IDX].mean(axis=0).astype(np.float32)  # (288,)
            mean_ra = windows[:, :, RA_IDX].mean(axis=0).astype(np.float32)  # (288,)
            stats[pid] = np.concatenate([cgm_feats, mean_pi, mean_ra])        # (582,)
        except Exception:
            pass
    return stats


# ── Derive CR from METABONET ───────────────────────────────────────────────────

def derive_cr():
    """
    Per-patient median CR (g/U) from co-occurring bolus+carbs events.
    Only METABONET (T1DEXI carb units are inconsistent).
    """
    print('  Deriving CR from METABONET...')
    dfs = []
    for path in [METABONET_TRAIN, METABONET_TEST]:
        df = pd.read_parquet(path, columns=['id', 'bolus', 'carbs'])
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    meal = df[(df['bolus'] > 0) & (df['carbs'] > 0)].copy()
    meal['cr'] = meal['carbs'] / meal['bolus']

    # remove physiologically impossible values (< 2 or > 50 g/U)
    meal = meal[(meal['cr'] >= 2) & (meal['cr'] <= 50)]

    # require at least 5 meal events per patient
    counts = meal.groupby('id')['cr'].count()
    valid_ids = counts[counts >= 5].index
    meal = meal[meal['id'].isin(valid_ids)]

    cr_per_patient = meal.groupby('id')['cr'].median().to_dict()
    print(f'  CR: {len(cr_per_patient)} patients, '
          f'range {min(cr_per_patient.values()):.1f}–{max(cr_per_patient.values()):.1f} g/U')
    return {str(k): v for k, v in cr_per_patient.items()}


# ── Derive ISF from METABONET ──────────────────────────────────────────────────

def derive_isf():
    """
    Per-patient median ISF (mg/dL per U) from correction boluses.
    Correction bolus: bolus > 0, carbs == 0.
    ISF estimated as (CGM_at_bolus - CGM_min_next90min) / bolus.
    Requires CGM > 150 at correction time, clear drop observed.
    """
    print('  Deriving ISF from METABONET (iter_batches to avoid OOM)...')
    import pyarrow.parquet as pq
    isf_records = []
    STEPS_90MIN = 18
    COLS = ['id', 'date', 'CGM', 'bolus', 'carbs']

    # Accumulate per-patient rows across batches, then process
    patient_rows = {}   # pid -> list of (date, CGM, bolus, carbs)

    for path in [METABONET_TRAIN, METABONET_TEST]:
        print(f'    Reading {path}...', flush=True)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=100_000, columns=COLS):
            df = batch.to_pandas()
            for pid, grp in df.groupby('id'):
                pid = str(pid)
                if pid not in patient_rows:
                    patient_rows[pid] = []
                patient_rows[pid].append(grp)
            del df
        print(f'    Done. {len(patient_rows)} patients accumulated.', flush=True)

    print(f'  Processing {len(patient_rows)} patients...', flush=True)
    for pid, chunks in patient_rows.items():
        grp = pd.concat(chunks).sort_values('date').reset_index(drop=True)
        cgm   = grp['CGM'].values
        bolus = grp['bolus'].values
        carbs = grp['carbs'].fillna(0).values
        for i in range(len(grp) - STEPS_90MIN):
            if bolus[i] <= 0 or carbs[i] > 0:
                continue
            if np.isnan(cgm[i]) or cgm[i] < 150:
                continue
            future_cgm = cgm[i+1 : i+1+STEPS_90MIN]
            valid = future_cgm[~np.isnan(future_cgm)]
            if len(valid) < 6:
                continue
            cgm_min = valid.min()
            if cgm_min >= cgm[i] - 10:
                continue
            isf_est = (cgm[i] - cgm_min) / bolus[i]
            if 5 <= isf_est <= 200:
                isf_records.append({'id': pid, 'isf': isf_est})
    del patient_rows

    if not isf_records:
        print('  ISF: no valid records found.')
        return {}

    isf_df = pd.DataFrame(isf_records)
    counts  = isf_df.groupby('id')['isf'].count()
    valid   = counts[counts >= 5].index
    isf_df  = isf_df[isf_df['id'].isin(valid)]
    isf_per_patient = isf_df.groupby('id')['isf'].median().to_dict()
    print(f'  ISF: {len(isf_per_patient)} patients, '
          f'range {min(isf_per_patient.values()):.1f}–{max(isf_per_patient.values()):.1f} mg/dL/U')
    return isf_per_patient


# ── Load HbA1c ─────────────────────────────────────────────────────────────────

def load_hba1c():
    """Returns dict pid -> HbA1c (%) for T1DEXI patients in processed dataset."""
    print('  Loading HbA1c from T1DEXI...')
    lb = pd.read_csv(T1DEXI_LB)
    hba1c = lb[lb['LBTESTCD'] == 'HBA1C'][['USUBJID', 'LBSTRESN']].copy()
    # LB.csv USUBJID is numeric (e.g. 1000); our cache uses 'T_1000' prefix
    hba1c['USUBJID'] = hba1c['USUBJID'].apply(lambda x: f'T_{int(x)}')
    hba1c = hba1c.groupby('USUBJID')['LBSTRESN'].mean()
    result = hba1c.to_dict()
    print(f'  HbA1c: {len(result)} T1DEXI patients, '
          f'range {min(result.values()):.1f}–{max(result.values()):.1f}%')
    return result


# ── Ridge probe ────────────────────────────────────────────────────────────────

def ridge_probe(X, y, n_splits=5, n_pca=None):
    """5-fold CV Ridge regression, optionally with PCA preprocessing. Returns mean R² and std."""
    steps = [StandardScaler()]
    if n_pca is not None:
        n_pca_actual = min(n_pca, X.shape[0] - 1, X.shape[1])
        if n_pca_actual >= 1:
            steps.append(PCA(n_components=n_pca_actual, random_state=SEED))
    steps.append(RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5))
    pipe = make_pipeline(*steps)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipe, X, y, cv=kf, scoring='r2')
    return float(scores.mean()), float(scores.std())


# ── Run one target ─────────────────────────────────────────────────────────────

def run_target(name, label_dict, enc_embs, dec_embs, cgm_stats, full_stats, pids, axes):
    """
    Align patients, run probes for encoder / decoder / cgm_stats / full_stats,
    print results, fill axes[0..3] with scatter plots.
    """
    # align — require both stat dicts to be present
    common = [p for p in pids
              if p in label_dict and p in cgm_stats and p in full_stats]
    if len(common) < 20:
        print(f'  {name}: only {len(common)} patients — skipping.')
        return None

    idx_map  = {p: i for i, p in enumerate(pids)}
    idx      = [idx_map[p] for p in common]
    y        = np.array([label_dict[p] for p in common])
    X_enc    = enc_embs[idx]
    X_dec    = dec_embs[idx]
    X_cgm    = np.stack([cgm_stats[p] for p in common])
    X_full   = np.stack([full_stats[p] for p in common])

    results = {}
    for feat_name, X, n_pca in [('Encoder h_cls',      X_enc,  None),
                                  ('Encoder+PCA-10',     X_enc,    10),
                                  ('Decoder H',          X_dec,  None),
                                  ('Decoder+PCA-10',     X_dec,    10),
                                  ('CGM stats',          X_cgm,  None),
                                  ('CGM+PI+RA curves',   X_full,   32)]:
        r2, std = ridge_probe(X, y, n_pca=n_pca)
        results[feat_name] = (r2, std)
        print(f'    {feat_name:22s}  R²={r2:.3f} ± {std:.3f}')

    # scatter: full fit for visual (4 panels: encoder, decoder, CGM, CGM+PI+RA)
    for ax, (feat_name, X, n_pca) in zip(axes, [('Encoder h_cls',     X_enc,  None),
                                                  ('Decoder H',         X_dec,  None),
                                                  ('CGM stats',         X_cgm,  None),
                                                  ('CGM+PI+RA curves',  X_full,   32)]):
        steps = [StandardScaler()]
        if n_pca is not None:
            steps.append(PCA(n_components=min(n_pca, X.shape[0]-1, X.shape[1]),
                             random_state=SEED))
        steps.append(RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5))
        pipe = make_pipeline(*steps)
        pipe.fit(X, y)
        y_pred = pipe.predict(X)
        r2 = results[feat_name][0]
        ax.scatter(y, y_pred, alpha=0.5, s=20, edgecolors='none')
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=1)
        ax.set_xlabel(f'Actual {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{feat_name}\nR²={r2:.3f} (CV)  n={len(common)}')

    return results


# ── Summary bar chart ──────────────────────────────────────────────────────────

def plot_summary(all_results, out_path):
    targets   = list(all_results.keys())
    feat_names = ['Encoder h_cls', 'Decoder H', 'CGM stats', 'CGM+PI+RA curves']
    colors     = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']

    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, tgt in zip(axes, targets):
        res = all_results[tgt]
        names  = [f for f in feat_names if f in res]
        r2s    = [res[f][0] for f in names]
        stds   = [res[f][1] for f in names]
        cols   = [colors[feat_names.index(f)] for f in names]
        bars   = ax.bar(names, r2s, yerr=stds, color=cols, capsize=4,
                        alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylim(min(0, min(r2s) - 0.1), min(1.05, max(r2s) + 0.15))
        ax.set_title(tgt, fontweight='bold')
        ax.set_ylabel('Cross-validated R²')
        ax.tick_params(axis='x', rotation=15)
        for bar, r2 in zip(bars, r2s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{r2:.3f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle('Patient-level prediction: embedding vs CGM statistics', fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print('\n=== Patient-level analysis ===\n')

    print('Loading cached embeddings...')
    enc_embs, dec_embs, pids, _ = load_cached()
    print(f'  Encoder: {enc_embs.shape}, Decoder: {dec_embs.shape}, Patients: {len(pids)}')

    print('\nComputing stats from .npz...')
    cgm_stats  = compute_cgm_stats(pids)
    full_stats = compute_cgm_pi_ra_stats(pids)
    print(f'  CGM stats: {len(cgm_stats)} patients, CGM+PI+RA curves: {len(full_stats)} patients')

    print('\nDeriving labels...')
    hba1c_dict = load_hba1c()
    cr_dict    = derive_cr()
    try:
        isf_dict = derive_isf()
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f'  ISF derivation failed: {e} — skipping.')
        isf_dict = {}

    targets = {
        'HbA1c (%)'  : hba1c_dict,
        'CR (g/U)'   : cr_dict,
        'ISF (mg/dL/U)': isf_dict,
    }

    all_results = {}
    fig_scatter, ax_scatter = plt.subplots(
        len(targets), 4,
        figsize=(20, 5 * len(targets))
    )

    for row_i, (tgt_name, label_dict) in enumerate(targets.items()):
        print(f'\n── {tgt_name} ──')
        axes_row = ax_scatter[row_i] if len(targets) > 1 else ax_scatter
        res = run_target(tgt_name, label_dict, enc_embs, dec_embs,
                         cgm_stats, full_stats, pids, axes_row)
        if res is not None:
            all_results[tgt_name] = res

    scatter_path = os.path.join(OUT_DIR, 'scatter_all.png')
    fig_scatter.suptitle('Predicted vs Actual — patient-level targets', fontweight='bold')
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved scatter: {scatter_path}')

    if all_results:
        plot_summary(all_results, os.path.join(OUT_DIR, 'r2_summary.png'))

    # Print summary table
    print('\n── Summary ──')
    print(f'{"Target":<20} {"Encoder R²":>12} {"Decoder R²":>12} {"CGM stats R²":>14} {"CGM+PI+RA R²":>14}')
    print('-' * 76)
    for tgt, res in all_results.items():
        enc_r2  = f"{res.get('Encoder h_cls',    (float('nan'),))[0]:.3f}"
        dec_r2  = f"{res.get('Decoder H',        (float('nan'),))[0]:.3f}"
        cgm_r2  = f"{res.get('CGM stats',        (float('nan'),))[0]:.3f}"
        full_r2 = f"{res.get('CGM+PI+RA curves',  (float('nan'),))[0]:.3f}"
        print(f'{tgt:<20} {enc_r2:>12} {dec_r2:>12} {cgm_r2:>14} {full_r2:>14}')

    print('\nDone.')


if __name__ == '__main__':
    main()
