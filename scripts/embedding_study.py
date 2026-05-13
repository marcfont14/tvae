"""
embedding_study.py
==================
Patient-level embedding study comparing encoder_clean (h_cls, BERT-style)
and decoder_clean (mean-pool H, GPT-style).

Run inside docker from /mnt/workspace/tvae:
  python -u scripts/embedding_study.py 2>&1 | tee results/embedding_study/log.txt

Outputs (results/embedding_study/):
  encoder_embeddings.npy   (N, 128)  h_cls averaged per patient
  decoder_embeddings.npy   (N, 128)  mean-pool(H) averaged per patient
  clinical_vars.npz        GRI, TIR, TBR, TAR, CV, therapy, patient_ids
  plots/                   UMAP, geometry metrics, probes
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

from src.encoder import (
    build_encoder, build_decoder,
    WINDOW_LEN, N_FEATURES, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR          = 'data/processed/adults'
ENCODER_WEIGHTS   = 'results/mtsm/encoder_clean/encoder_weights.weights.h5'
DECODER_WEIGHTS   = 'results/mtsm/decoder_clean/encoder_weights.weights.h5'
OUT_DIR           = 'results/embedding_study'
PLOT_DIR          = os.path.join(OUT_DIR, 'plots')

# ── Constants ──────────────────────────────────────────────────────────────────
GLOBAL_CGM_MEAN = 144.40
GLOBAL_CGM_STD  = 57.11
CGM_IDX         = 0
BOLUS_IDX       = 5
CARBS_IDX       = 6
THERAPY_SLICE   = slice(7, 10)   # AID, SAP, MDI one-hot
BATCH_SIZE      = 256            # windows per forward pass
SEED            = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(PLOT_DIR, exist_ok=True)

# ── GPU setup ──────────────────────────────────────────────────────────────────
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass


# ── Clinical variables (from globally normalised CGM z-scores) ─────────────────

def compute_clinical_vars(windows_all: np.ndarray) -> dict:
    """
    windows_all: (N_windows, 288, 11) — all windows for one patient.
    Returns per-patient scalar clinical variables computed from ALL windows.
    CGM channel is globally z-scored; reconstruct to mg/dL for thresholds.
    """
    cgm_z   = windows_all[:, :, CGM_IDX]                         # (N, 288)
    cgm_mg  = cgm_z * GLOBAL_CGM_STD + GLOBAL_CGM_MEAN           # mg/dL

    tir = ((cgm_mg >= 70) & (cgm_mg <= 180)).mean()
    tbr = (cgm_mg < 70).mean()
    tar = (cgm_mg > 180).mean()

    # TBR sub-ranges for GRI
    tbr_vlow = (cgm_mg < 54).mean()
    tbr_low  = ((cgm_mg >= 54) & (cgm_mg < 70)).mean()
    tar_vhigh = (cgm_mg > 250).mean()
    tar_high  = ((cgm_mg >= 180) & (cgm_mg <= 250)).mean()

    gri = min(100.0, 3.0 * tbr_vlow * 100 + 2.4 * tbr_low * 100
                     + 1.6 * tar_vhigh * 100 + 0.8 * tar_high * 100)

    cv  = cgm_mg.std() / (cgm_mg.mean() + 1e-8)

    therapy_votes = windows_all[:, 0, THERAPY_SLICE].mean(axis=0)  # (3,)
    therapy = int(np.argmax(therapy_votes))   # 0=AID, 1=SAP, 2=MDI

    return dict(tir=float(tir), tbr=float(tbr), tar=float(tar),
                cv=float(cv), gri=float(gri), therapy=therapy)


# ── Forward-pass helpers ───────────────────────────────────────────────────────

def embed_patient_encoder(encoder, windows: np.ndarray) -> np.ndarray:
    """Average h_cls over all windows for one patient. (128,)"""
    h_cls_list = []
    for start in range(0, len(windows), BATCH_SIZE):
        batch = windows[start:start + BATCH_SIZE, :, :N_FEATURES].astype(np.float32)
        _, h_cls = encoder(batch, training=False)
        h_cls_list.append(h_cls.numpy())
    return np.concatenate(h_cls_list, axis=0).mean(axis=0)   # (128,)


def embed_patient_decoder(decoder, windows: np.ndarray) -> np.ndarray:
    """Average mean-pool(H) over all windows for one patient. (128,)"""
    pool_list = []
    for start in range(0, len(windows), BATCH_SIZE):
        batch = windows[start:start + BATCH_SIZE, :, :N_FEATURES].astype(np.float32)
        H, _ = decoder(batch, training=False)
        pool_list.append(H.numpy().mean(axis=1))   # (B, 128)
    return np.concatenate(pool_list, axis=0).mean(axis=0)    # (128,)


# ── Main forward pass ──────────────────────────────────────────────────────────

def build_embeddings():
    print('\n  Loading models...')
    encoder = build_encoder()
    encoder(tf.zeros((1, WINDOW_LEN, N_FEATURES)))
    encoder.load_weights(ENCODER_WEIGHTS)
    encoder.trainable = False

    decoder = build_decoder()
    decoder(tf.zeros((1, WINDOW_LEN, N_FEATURES)))
    decoder.load_weights(DECODER_WEIGHTS)
    decoder.trainable = False

    npz_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.npz')])
    print(f'  Patients found: {len(npz_files)}')

    enc_embs, dec_embs, clin_vars, patient_ids = [], [], [], []
    skipped = 0

    for i, fname in enumerate(npz_files):
        pid = fname.replace('.npz', '')
        fpath = os.path.join(DATA_DIR, fname)
        try:
            data = np.load(fpath, allow_pickle=True)
            wins = data['windows'].astype(np.float32)   # (N, 288, 11)
        except Exception as e:
            print(f'  [WARN] {fname}: {e}')
            skipped += 1
            continue

        if len(wins) == 0:
            skipped += 1
            continue

        enc_embs.append(embed_patient_encoder(encoder, wins))
        dec_embs.append(embed_patient_decoder(decoder, wins))
        clin_vars.append(compute_clinical_vars(wins))
        patient_ids.append(pid)

        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{len(npz_files)}] {fname}  GRI={clin_vars[-1]["gri"]:.1f}')

    print(f'  Done. Patients embedded: {len(enc_embs)}  Skipped: {skipped}')

    enc_embs = np.stack(enc_embs)    # (N, 128)
    dec_embs = np.stack(dec_embs)    # (N, 128)

    gri      = np.array([v['gri']     for v in clin_vars])
    tir      = np.array([v['tir']     for v in clin_vars])
    tbr      = np.array([v['tbr']     for v in clin_vars])
    tar      = np.array([v['tar']     for v in clin_vars])
    cv       = np.array([v['cv']      for v in clin_vars])
    therapy  = np.array([v['therapy'] for v in clin_vars])

    np.save(os.path.join(OUT_DIR, 'encoder_embeddings.npy'), enc_embs)
    np.save(os.path.join(OUT_DIR, 'decoder_embeddings.npy'), dec_embs)
    np.savez(os.path.join(OUT_DIR, 'clinical_vars.npz'),
             gri=gri, tir=tir, tbr=tbr, tar=tar, cv=cv, therapy=therapy,
             patient_ids=np.array(patient_ids))
    print(f'  Saved embeddings and clinical vars to {OUT_DIR}/')
    return enc_embs, dec_embs, gri, tir, therapy, cv


# ── Geometry metrics ───────────────────────────────────────────────────────────

def pca_effective_dims(emb: np.ndarray) -> dict:
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(emb)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    return {
        'n80': int(np.searchsorted(cumvar, 0.80) + 1),
        'n90': int(np.searchsorted(cumvar, 0.90) + 1),
        'n95': int(np.searchsorted(cumvar, 0.95) + 1),
        'explained_variance_ratio': pca.explained_variance_ratio_,
    }


def isotropy(emb: np.ndarray, n_pairs: int = 10_000) -> dict:
    from sklearn.preprocessing import normalize
    emb_l2 = normalize(emb)
    idx     = np.random.randint(0, len(emb_l2), size=(n_pairs, 2))
    sims    = (emb_l2[idx[:, 0]] * emb_l2[idx[:, 1]]).sum(axis=1)
    return {'mean_cos': float(sims.mean()), 'std_cos': float(sims.std())}


def lid_mle(emb: np.ndarray, k: int = 20) -> float:
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(emb)
    dists, _ = nbrs.kneighbors(emb)
    dists    = dists[:, 1:]    # exclude self
    r_k      = dists[:, -1:]
    ratios   = np.clip(dists / (r_k + 1e-10), 1e-10, 1.0)
    lid      = -1.0 / (np.log(ratios).mean(axis=1) + 1e-10)
    return float(lid.mean())


def kmeans_silhouette(emb: np.ndarray, k_range=range(2, 11)) -> tuple:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    scores, labels_best, k_best = [], None, 2
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(emb)
        sil    = silhouette_score(emb, labels, metric='cosine')
        scores.append(sil)
        if sil == max(scores):
            labels_best, k_best = labels, k
    return list(k_range), scores, k_best, labels_best


def knn_consistency(emb: np.ndarray, gri: np.ndarray, k: int = 10) -> float:
    """Fraction of K nearest neighbours sharing the same GRI quartile."""
    from sklearn.neighbors import NearestNeighbors
    quartiles = np.digitize(gri, np.percentile(gri, [25, 50, 75]))
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    _, idx = nbrs.kneighbors(emb)
    idx  = idx[:, 1:]   # exclude self
    consistency = []
    for i, neighbours in enumerate(idx):
        frac = (quartiles[neighbours] == quartiles[i]).mean()
        consistency.append(frac)
    return float(np.mean(consistency))


def linear_probe(emb: np.ndarray, gri: np.ndarray, tir: np.ndarray,
                 cv: np.ndarray, name: str) -> dict:
    """Ridge regression (5-fold CV) predicting GRI from embeddings vs raw stats."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    pipe_emb = make_pipeline(StandardScaler(), Ridge())
    r2_emb   = cross_val_score(pipe_emb, emb, gri, cv=5, scoring='r2').mean()

    # Raw CGM statistics baseline
    raw_feat = np.column_stack([tir, cv, 1 - tir - cv])   # TIR, CV, TBR+TAR
    pipe_raw = make_pipeline(StandardScaler(), Ridge())
    r2_raw   = cross_val_score(pipe_raw, raw_feat, gri, cv=5, scoring='r2').mean()

    print(f'  [{name}] Linear probe GRI — Embedding R²={r2_emb:.3f}  Raw stats R²={r2_raw:.3f}')
    return {'r2_emb': float(r2_emb), 'r2_raw': float(r2_raw)}


def geometry_summary(name: str, emb: np.ndarray, gri: np.ndarray,
                     tir: np.ndarray, cv: np.ndarray) -> dict:
    print(f'\n  === Geometry: {name} ===')
    pca  = pca_effective_dims(emb)
    iso  = isotropy(emb)
    lid  = lid_mle(emb)
    knn  = knn_consistency(emb, gri)
    probe = linear_probe(emb, gri, tir, cv, name)
    k_vals, sil_scores, k_best, _ = kmeans_silhouette(emb)

    print(f'  PCA dims (80/90/95%): {pca["n80"]} / {pca["n90"]} / {pca["n95"]}')
    print(f'  Isotropy mean cos:    {iso["mean_cos"]:.4f}  std: {iso["std_cos"]:.4f}')
    print(f'  LID (k=20):           {lid:.2f}')
    print(f'  KNN consistency:      {knn:.3f}  (k=10, GRI quartile)')
    print(f'  Best K (silhouette):  {k_best}  score={max(sil_scores):.4f}')

    return dict(name=name, pca=pca, iso=iso, lid=lid, knn=knn,
                probe=probe, k_vals=k_vals, sil_scores=sil_scores, k_best=k_best)


# ── UMAP ───────────────────────────────────────────────────────────────────────

def umap_plots(emb: np.ndarray, gri: np.ndarray, tir: np.ndarray,
               therapy: np.ndarray, name: str):
    try:
        import umap
    except ImportError:
        print('  [WARN] umap-learn not installed — skipping UMAP plots')
        return

    print(f'  Running UMAP for {name}...')
    reducer   = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.1)
    coords    = reducer.fit_transform(emb)   # (N, 2)

    gri_q     = np.digitize(gri, np.percentile(gri, [25, 50, 75]))
    therapy_labels = ['AID', 'SAP', 'MDI']
    therapy_colors = ['#2196F3', '#FF9800', '#4CAF50']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'UMAP — {name}', fontsize=13)

    # Panel 1: GRI quartile
    cmap4 = ['#27ae60', '#f39c12', '#e74c3c', '#8e44ad']
    for q, label, c in zip([0,1,2,3], ['Q1 (best)','Q2','Q3','Q4 (worst)'], cmap4):
        mask = gri_q == q
        axes[0].scatter(coords[mask, 0], coords[mask, 1], c=c, label=label, s=8, alpha=0.7)
    axes[0].set_title('GRI quartile')
    axes[0].legend(markerscale=2, fontsize=8)
    axes[0].set_xlabel('UMAP-1')
    axes[0].set_ylabel('UMAP-2')

    # Panel 2: TIR (continuous)
    sc = axes[1].scatter(coords[:, 0], coords[:, 1], c=tir * 100, cmap='RdYlGn',
                         s=8, alpha=0.7, vmin=20, vmax=90)
    plt.colorbar(sc, ax=axes[1], label='TIR (%)')
    axes[1].set_title('Time In Range (%)')
    axes[1].set_xlabel('UMAP-1')

    # Panel 3: Therapy modality
    for t, label, c in zip([0,1,2], therapy_labels, therapy_colors):
        mask = therapy == t
        if mask.sum() > 0:
            axes[2].scatter(coords[mask, 0], coords[mask, 1], c=c, label=label, s=8, alpha=0.7)
    axes[2].set_title('Therapy modality')
    axes[2].legend(markerscale=2, fontsize=8)
    axes[2].set_xlabel('UMAP-1')

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f'umap_{name.lower().replace(" ", "_")}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Summary plots ──────────────────────────────────────────────────────────────

def plot_silhouette(results: list):
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in results:
        ax.plot(r['k_vals'], r['sil_scores'], marker='o', label=r['name'])
        best_idx = np.argmax(r['sil_scores'])
        ax.axvline(r['k_vals'][best_idx], linestyle='--', alpha=0.4)
    ax.set_xlabel('Number of clusters K')
    ax.set_ylabel('Silhouette score (cosine)')
    ax.set_title('K-means silhouette — encoder vs decoder')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'silhouette.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


def plot_linear_probe(results: list):
    names   = [r['name'] for r in results]
    r2_emb  = [r['probe']['r2_emb'] for r in results]
    r2_raw  = [r['probe']['r2_raw'] for r in results]
    x       = np.arange(len(names))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width/2, r2_emb, width, label='Embedding (Ridge)', color='#2196F3')
    ax.bar(x + width/2, r2_raw, width, label='Raw CGM stats (Ridge)', color='#9E9E9E')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('R² (5-fold CV)')
    ax.set_title('Linear probe: h → GRI')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'linear_probe.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


def plot_pca_variance(results: list):
    fig, ax = plt.subplots(figsize=(8, 4))
    for r in results:
        cumvar = np.cumsum(r['pca']['explained_variance_ratio']) * 100
        ax.plot(np.arange(1, len(cumvar) + 1), cumvar, label=r['name'])
    for thresh, ls in [(80, ':'), (90, '--'), (95, '-.')]:
        ax.axhline(thresh, color='gray', linestyle=ls, alpha=0.5, linewidth=0.8)
        ax.text(128, thresh + 0.5, f'{thresh}%', color='gray', fontsize=8)
    ax.set_xlabel('Number of PCA components')
    ax.set_ylabel('Cumulative explained variance (%)')
    ax.set_title('PCA effective dimensionality')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 128)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'pca_variance.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


def plot_geometry_summary(results: list):
    metrics = ['n90', 'iso_mean', 'lid', 'knn', 'r2_emb']
    labels  = ['PCA dims\n(90% var)', 'Isotropy\n(mean cos)', 'LID\n(k=20)', 'KNN\nconsistency', 'Probe R²\n(GRI)']
    names   = [r['name'] for r in results]
    colors  = ['#2196F3', '#FF9800']

    values = []
    for r in results:
        values.append([
            r['pca']['n90'],
            r['iso']['mean_cos'],
            r['lid'],
            r['knn'],
            r['probe']['r2_emb'],
        ])

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    fig.suptitle('Geometry metrics — encoder vs decoder', fontsize=12)
    for j, (m, lbl) in enumerate(zip(metrics, labels)):
        for i, (name, vals) in enumerate(zip(names, values)):
            axes[j].bar(i, vals[j], color=colors[i], label=name if j == 0 else None)
        axes[j].set_title(lbl, fontsize=9)
        axes[j].set_xticks(range(len(names)))
        axes[j].set_xticklabels([n.split()[0] for n in names], fontsize=8)
        axes[j].grid(axis='y', alpha=0.3)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'geometry_summary.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Consistency scores ─────────────────────────────────────────────────────────

def build_consistency_scores(patient_ids: np.ndarray) -> np.ndarray:
    """
    Per-patient intra-patient spread: mean cosine distance of each window's
    h_cls from that patient's mean h_cls.
    Low = consistent day-to-day dynamics. High = erratic / unpredictable.
    Saved to results/embedding_study/consistency_scores.npy.
    """
    print('\n  Building consistency scores (encoder forward pass)...')
    encoder = build_encoder()
    encoder(tf.zeros((1, WINDOW_LEN, N_FEATURES)))
    encoder.load_weights(ENCODER_WEIGHTS)
    encoder.trainable = False

    spreads = []
    for i, pid in enumerate(patient_ids):
        fpath = os.path.join(DATA_DIR, f'{pid}.npz')
        try:
            data  = np.load(fpath, allow_pickle=True)
            wins  = data['windows'].astype(np.float32)
        except Exception as e:
            print(f'  [WARN] {pid}: {e}')
            spreads.append(np.nan)
            continue

        h_list = []
        for start in range(0, len(wins), BATCH_SIZE):
            batch = wins[start:start + BATCH_SIZE, :, :N_FEATURES]
            _, h_cls = encoder(batch, training=False)
            h_list.append(h_cls.numpy())
        h_all = np.concatenate(h_list, axis=0)          # (N_wins, 128)

        h_mean      = h_all.mean(axis=0)
        h_mean_norm = h_mean / (np.linalg.norm(h_mean) + 1e-8)
        norms       = np.linalg.norm(h_all, axis=1, keepdims=True)
        h_norm      = h_all / (norms + 1e-8)
        cos_sims    = h_norm @ h_mean_norm               # (N_wins,)
        spread      = float((1.0 - cos_sims).mean())
        spreads.append(spread)

        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{len(patient_ids)}] spread={spread:.4f}')

    spreads = np.array(spreads, dtype=np.float32)
    out_path = os.path.join(OUT_DIR, 'consistency_scores.npy')
    np.save(out_path, spreads)
    print(f'  Saved: {out_path}  (mean={np.nanmean(spreads):.4f})')
    return spreads


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    enc_path = os.path.join(OUT_DIR, 'encoder_embeddings.npy')
    dec_path = os.path.join(OUT_DIR, 'decoder_embeddings.npy')
    clin_path = os.path.join(OUT_DIR, 'clinical_vars.npz')

    if os.path.exists(enc_path) and os.path.exists(clin_path):
        print('  Loading cached embeddings...')
        enc_embs = np.load(enc_path)
        dec_embs = np.load(dec_path)
        clin     = np.load(clin_path, allow_pickle=True)
        gri, tir, therapy, cv = clin['gri'], clin['tir'], clin['therapy'], clin['cv']
        patient_ids = clin['patient_ids']
    else:
        enc_embs, dec_embs, gri, tir, therapy, cv = build_embeddings()
        clin        = np.load(clin_path, allow_pickle=True)
        patient_ids = clin['patient_ids']

    consist_path = os.path.join(OUT_DIR, 'consistency_scores.npy')
    if os.path.exists(consist_path):
        print('  Loading cached consistency scores...')
    else:
        build_consistency_scores(patient_ids)

    print(f'\n  Patients: {len(enc_embs)}')
    print(f'  GRI: mean={gri.mean():.1f}  std={gri.std():.1f}  range=[{gri.min():.1f}, {gri.max():.1f}]')
    print(f'  TIR: mean={tir.mean()*100:.1f}%')
    therapy_counts = {0: (therapy==0).sum(), 1: (therapy==1).sum(), 2: (therapy==2).sum()}
    print(f'  Therapy: AID={therapy_counts[0]}  SAP={therapy_counts[1]}  MDI={therapy_counts[2]}')

    # UMAP
    umap_plots(enc_embs, gri, tir, therapy, 'Encoder (h_cls)')
    umap_plots(dec_embs, gri, tir, therapy, 'Decoder (mean-pool H)')

    # Geometry metrics
    results = []
    for name, emb in [('Encoder (h_cls)', enc_embs), ('Decoder (mean-pool H)', dec_embs)]:
        r = geometry_summary(name, emb, gri, tir, cv)
        results.append(r)

    # Summary plots
    plot_pca_variance(results)
    plot_silhouette(results)
    plot_linear_probe(results)
    plot_geometry_summary(results)

    # Print final summary
    print('\n' + '='*60)
    print('  SUMMARY')
    print('='*60)
    for r in results:
        print(f"\n  {r['name']}")
        print(f"    PCA 90% var:     {r['pca']['n90']} components")
        print(f"    Isotropy:        mean_cos={r['iso']['mean_cos']:.4f}  std={r['iso']['std_cos']:.4f}")
        print(f"    LID (k=20):      {r['lid']:.2f}")
        print(f"    KNN consistency: {r['knn']:.3f}")
        print(f"    Probe R² (GRI):  emb={r['probe']['r2_emb']:.3f}  raw={r['probe']['r2_raw']:.3f}")
        print(f"    Best K:          {r['k_best']}  silhouette={max(r['sil_scores']):.4f}")

    print(f'\n  All plots saved to {PLOT_DIR}/')


if __name__ == '__main__':
    main()
