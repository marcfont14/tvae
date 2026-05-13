"""
analyze_encoder.py
==================
Embedding space analysis for the MTSM encoder.

Metrics:
  1. Isotropy   — avg cosine similarity between random h_cls pairs
                  (near 0 = well-distributed, near 1 = collapsed)
  2. Alignment  — intra-patient vs inter-patient cosine similarity gap
                  (positive gap = encoder groups same-patient windows together)
  3. UMAP/t-SNE — 2D projection of h_cls coloured by therapy type,
                  mean CGM, and time of day
                  (2 rows: trained encoder vs random init)

Usage (from repo root):
  python scripts/analyze_encoder.py \\
    --data data/processed/adults \\
    --weights results/mtsm/encoder2/encoder_weights.weights.h5 \\
    --out results/mtsm/encoder2
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from src.encoder import build_encoder, load_encoder, build_encoder3, load_encoder3

SEED     = 42
IDX_CGM  = 0
IDX_HSIN = 3
IDX_HCOS = 4

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_windows(processed_dir, max_patients, max_per_patient, seed):
    rng       = np.random.default_rng(seed)
    npz_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('.npz')])
    if max_patients:
        sel       = rng.choice(len(npz_files), min(max_patients, len(npz_files)), replace=False)
        npz_files = [npz_files[i] for i in sorted(sel)]

    windows_list, pids, scalers = [], [], []
    for pid, fname in enumerate(npz_files):
        try:
            d    = np.load(os.path.join(processed_dir, fname), allow_pickle=True)
            wins = d['windows'].astype(np.float32)
            mean = float(d['scaler_mean'][0])
            std  = float(d['scaler_std'][0])
        except Exception:
            continue

        cgm        = wins[:, :, IDX_CGM]
        cgm_std    = cgm.std(axis=1)
        has_driver = ((wins[:, :, 5] + wins[:, :, 6]) > 0).any(axis=1)
        keep       = has_driver & (cgm_std > 0.3) & (cgm_std < 4.0)
        wins       = wins[keep, :, :10]  # drop age_norm
        if len(wins) == 0:
            continue

        if len(wins) > max_per_patient:
            idx  = rng.choice(len(wins), max_per_patient, replace=False)
            wins = wins[idx]

        windows_list.append(wins)
        pids.extend([pid] * len(wins))
        scalers.extend([(mean, std)] * len(wins))

    return (np.concatenate(windows_list, axis=0).astype(np.float32),
            np.array(pids, dtype=np.int32),
            np.array(scalers, dtype=np.float32))


# ── Clinical feature extraction ───────────────────────────────────────────────

def extract_features(windows, scalers):
    """Return therapy (0/1/2 = AID/SAP/MDI), mean CGM (mg/dL), hour (0–24)."""
    therapy  = np.argmax(windows[:, :, 7:10].mean(axis=1), axis=1)
    cgm_mg   = windows[:, :, IDX_CGM] * scalers[:, 1:2] + scalers[:, 0:1]
    mean_cgm = cgm_mg.mean(axis=1)
    hsin     = windows[:, 0, IDX_HSIN]
    hcos     = windows[:, 0, IDX_HCOS]
    hour     = np.arctan2(hsin, hcos) / (2 * np.pi) * 24 % 24
    return therapy, mean_cgm, hour


# ── Metrics ───────────────────────────────────────────────────────────────────

def isotropy(h, n_pairs=5000):
    """Average cosine similarity between random pairs. Near 0 = isotropic."""
    rng    = np.random.default_rng(SEED)
    N      = len(h)
    hn     = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
    i      = rng.integers(0, N, n_pairs)
    j      = (i + rng.integers(1, N, n_pairs)) % N
    sims   = (hn[i] * hn[j]).sum(axis=1)
    return float(sims.mean()), float(sims.std())


def alignment(h, patient_ids, n_pairs=2000):
    """Intra-patient vs inter-patient cosine similarity."""
    rng    = np.random.default_rng(SEED)
    hn     = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
    by_pat = defaultdict(list)
    for i, pid in enumerate(patient_ids):
        by_pat[pid].append(i)
    patients = [p for p in by_pat if len(by_pat[p]) >= 2]
    if len(patients) < 2:
        return 0.0, 0.0

    intra, inter = [], []
    for _ in range(n_pairs):
        pid  = rng.choice(patients)
        i, j = rng.choice(by_pat[pid], 2, replace=False)
        intra.append(float((hn[i] * hn[j]).sum()))
        p1, p2 = rng.choice(patients, 2, replace=False)
        i = rng.choice(by_pat[p1])
        j = rng.choice(by_pat[p2])
        inter.append(float((hn[i] * hn[j]).sum()))

    return float(np.mean(intra)), float(np.mean(inter))


# ── global summary extraction ─────────────────────────────────────────────────

def get_h_cls(encoder, windows, batch=128):
    """encoder returns [H, h_cls/h_last]; return the global summary vector."""
    return encoder.predict(windows, batch_size=batch, verbose=0)[1]


# ── PCA analysis ─────────────────────────────────────────────────────────────

def pca_analysis(h_trained, h_random, therapy, mean_cgm, hour, path, json_out,
                 enc_tag='Trained encoder2'):
    """
    3-row figure:
      Row 0 — trained encoder PC1 vs PC2 (3 colorings)
      Row 1 — random encoder  PC1 vs PC2 (3 colorings)
      Row 2 — explained variance comparison + PC-clinical correlations
    Also prints / saves PC-clinical correlation table and variance summary.
    """
    from sklearn.decomposition import PCA
    from scipy.stats import pearsonr

    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']
    LABELS = ['AID', 'SAP', 'MDI']
    S = 4

    # Fit PCA on all dims, then project
    pca_t = PCA(random_state=SEED).fit(h_trained)
    pca_r = PCA(random_state=SEED).fit(h_random)
    pc_t  = pca_t.transform(h_trained)
    pc_r  = pca_r.transform(h_random)

    ev_t = pca_t.explained_variance_ratio_
    ev_r = pca_r.explained_variance_ratio_

    def pcs_for_90(ev):
        return int(np.searchsorted(np.cumsum(ev), 0.90)) + 1

    n90_t = pcs_for_90(ev_t)
    n90_r = pcs_for_90(ev_r)

    # Pearson |r| of top-5 PCs with each clinical variable
    clin_vars  = {'mean_cgm': mean_cgm, 'therapy': therapy.astype(float), 'hour': hour}
    n_pcs_corr = 5

    def pc_cors(pc):
        table = {}
        for name, v in clin_vars.items():
            table[name] = [abs(pearsonr(pc[:, i], v)[0]) for i in range(n_pcs_corr)]
        return table

    cors_t = pc_cors(pc_t)
    cors_r = pc_cors(pc_r)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f'\n  PCA — PCs needed for 90% variance:')
    print(f'    Trained : {n90_t}   (PC1 ev={ev_t[0]:.3f})')
    print(f'    Random  : {n90_r}   (PC1 ev={ev_r[0]:.3f})')
    print(f'\n  |Pearson r| of top-{n_pcs_corr} PCs with clinical vars:')
    header = f'  {"":15}' + ''.join(f'{"PC"+str(i+1):>8}' for i in range(n_pcs_corr))
    print(header)
    for name in clin_vars:
        for tag, cors in [('Trained', cors_t), ('Random ', cors_r)]:
            row = f'  {tag+" "+name:<15}' + ''.join(f'{cors[name][i]:>8.3f}' for i in range(n_pcs_corr))
            print(row)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14))
    gs  = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    for row_idx, (pc, tag, ev) in enumerate([(pc_t, enc_tag, ev_t),
                                              (pc_r, 'Random init', ev_r)]):
        # Therapy type
        ax = fig.add_subplot(gs[row_idx, 0])
        for t, (col, lbl) in enumerate(zip(COLORS, LABELS)):
            m = therapy == t
            ax.scatter(pc[m, 0], pc[m, 1], c=col, s=S, alpha=0.5,
                       label=lbl, linewidths=0, rasterized=True)
        ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontsize=8)
        ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontsize=8)
        ax.set_title(f'{tag}\nTherapy type', fontsize=9)
        if row_idx == 0:
            ax.legend(fontsize=8, markerscale=3, loc='best')

        # Mean CGM
        ax = fig.add_subplot(gs[row_idx, 1])
        vlo, vhi = np.percentile(mean_cgm, [5, 95])
        sc = ax.scatter(pc[:, 0], pc[:, 1], c=mean_cgm, cmap='RdYlGn_r',
                        s=S, alpha=0.5, vmin=vlo, vmax=vhi,
                        linewidths=0, rasterized=True)
        fig.colorbar(sc, ax=ax, fraction=0.03, label='mg/dL')
        ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontsize=8)
        ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontsize=8)
        ax.set_title(f'{tag}\nMean CGM', fontsize=9)

        # Hour
        ax = fig.add_subplot(gs[row_idx, 2])
        sc = ax.scatter(pc[:, 0], pc[:, 1], c=hour, cmap='twilight',
                        s=S, alpha=0.5, vmin=0, vmax=24,
                        linewidths=0, rasterized=True)
        cb = fig.colorbar(sc, ax=ax, fraction=0.03, label='Hour')
        cb.set_ticks([0, 6, 12, 18, 24])
        ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontsize=8)
        ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontsize=8)
        ax.set_title(f'{tag}\nTime of day', fontsize=9)

    # Row 2 — explained variance + PC-clinical correlation bars
    n_show = min(20, len(ev_t))
    xs     = np.arange(n_show)
    width  = 0.35

    ax_ev = fig.add_subplot(gs[2, 0])
    ax_ev.bar(xs - width/2, ev_t[:n_show] * 100, width, label='Trained', alpha=0.8)
    ax_ev.bar(xs + width/2, ev_r[:n_show] * 100, width, label='Random',  alpha=0.8)
    ax_ev.set_xlabel('PC index', fontsize=8)
    ax_ev.set_ylabel('Explained variance (%)', fontsize=8)
    ax_ev.set_title(f'Explained variance\n(90% at T={n90_t}, R={n90_r} PCs)', fontsize=9)
    ax_ev.legend(fontsize=8)
    ax_ev.set_xticks(xs[::2])
    ax_ev.set_xticklabels([str(i+1) for i in xs[::2]], fontsize=7)

    # PC-clinical |r| heat for top-5 PCs
    for col_idx, (name, cmap_) in enumerate([('mean_cgm', 'Blues'),
                                              ('therapy',  'Oranges')], start=1):
        ax_c = fig.add_subplot(gs[2, col_idx])
        mat  = np.array([cors_t[name], cors_r[name]])   # (2, n_pcs_corr)
        im   = ax_c.imshow(mat, aspect='auto', cmap=cmap_, vmin=0, vmax=1)
        fig.colorbar(im, ax=ax_c, fraction=0.04, label='|r|')
        ax_c.set_yticks([0, 1]); ax_c.set_yticklabels(['Trained', 'Random'], fontsize=8)
        ax_c.set_xticks(range(n_pcs_corr))
        ax_c.set_xticklabels([f'PC{i+1}' for i in range(n_pcs_corr)], fontsize=8)
        ax_c.set_title(f'|r| with {name}', fontsize=9)
        for i in range(2):
            for j in range(n_pcs_corr):
                ax_c.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                          fontsize=7, color='white' if mat[i,j] > 0.5 else 'black')

    fig.suptitle(f'{enc_tag} — PCA', fontsize=13)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')

    # Save JSON
    result = {
        'pcs_for_90pct': {'trained': n90_t, 'random': n90_r},
        'pc1_explained_var': {'trained': round(float(ev_t[0]), 4),
                              'random':  round(float(ev_r[0]), 4)},
        'pc_clinical_cors': {
            'trained': {k: [round(v, 4) for v in cors_t[k]] for k in cors_t},
            'random':  {k: [round(v, 4) for v in cors_r[k]] for k in cors_r},
        },
        'note': ('Higher |r| of top PCs with clinical vars in the random encoder '
                 'means it is preserving raw input feature correlations, not learning '
                 'abstract representations.'),
    }
    with open(json_out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  Saved: {json_out}')
    return result


# ── Dimensionality reduction ──────────────────────────────────────────────────

def reduce_2d(h):
    try:
        import umap
        emb   = umap.UMAP(n_components=2, random_state=SEED,
                          n_neighbors=15, min_dist=0.1).fit_transform(h)
        label = 'UMAP'
    except ImportError:
        from sklearn.manifold import TSNE
        emb   = TSNE(n_components=2, random_state=SEED,
                     perplexity=min(30, len(h) - 1)).fit_transform(h)
        label = 't-SNE'
    return emb, label


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_embeddings(emb_t, emb_r, therapy, mean_cgm, hour, method, path,
                    enc_tag='Trained encoder2'):
    """2 rows (trained / random) × 3 cols (therapy / mean CGM / time of day)."""
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']
    LABELS = ['AID', 'SAP', 'MDI']
    S      = 4

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'{enc_tag} — {method} projection', fontsize=13)

    for row, (emb, tag) in enumerate([(emb_t, enc_tag),
                                       (emb_r, 'Random init')]):
        # Therapy type
        ax = axes[row, 0]
        for t, (col, lbl) in enumerate(zip(COLORS, LABELS)):
            m = therapy == t
            ax.scatter(emb[m, 0], emb[m, 1], c=col, s=S, alpha=0.5,
                       label=lbl, linewidths=0, rasterized=True)
        ax.set_title(f'{tag}\nTherapy type', fontsize=9)
        if row == 0:
            ax.legend(fontsize=8, markerscale=3, loc='best')
        ax.set_xticks([]); ax.set_yticks([])

        # Mean CGM
        ax  = axes[row, 1]
        vlo, vhi = np.percentile(mean_cgm, [5, 95])
        sc  = ax.scatter(emb[:, 0], emb[:, 1], c=mean_cgm, cmap='RdYlGn_r',
                         s=S, alpha=0.5, vmin=vlo, vmax=vhi,
                         linewidths=0, rasterized=True)
        fig.colorbar(sc, ax=ax, fraction=0.03, label='mg/dL')
        ax.set_title(f'{tag}\nMean CGM', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        # Time of day
        ax  = axes[row, 2]
        sc  = ax.scatter(emb[:, 0], emb[:, 1], c=hour, cmap='twilight',
                         s=S, alpha=0.5, vmin=0, vmax=24,
                         linewidths=0, rasterized=True)
        cb  = fig.colorbar(sc, ax=ax, fraction=0.03, label='Hour')
        cb.set_ticks([0, 6, 12, 18, 24])
        ax.set_title(f'{tag}\nTime of day', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.out, exist_ok=True)

    print(f'\n{"="*52}')
    print(f'  Encoder Embedding Analysis')
    print(f'{"="*52}')

    # 1. Load windows
    print(f'\n  Loading windows from {args.data} ...')
    windows, pids, scalers = load_windows(
        args.data, args.max_patients, args.max_per_patient, SEED
    )
    print(f'  Windows: {len(windows):,}   Patients: {pids.max()+1}')

    # 2. global summary — trained
    if args.encoder3:
        enc_tag = 'Trained encoder3 (h_last)'
        print('\n  Extracting h_last — trained encoder3 ...')
        enc_trained = load_encoder3(weights_path=args.weights, trainable=False)
        print('  Building random encoder3 for baseline ...')
        enc_random  = build_encoder3()
        enc_random(tf.zeros((1, 288, 10)))
    else:
        enc_tag = 'Trained encoder2 (h_cls)'
        print('\n  Extracting h_cls — trained encoder2 ...')
        enc_trained = load_encoder(weights_path=args.weights, trainable=False)
        print('  Building random encoder2 for baseline ...')
        enc_random  = build_encoder()
        enc_random(tf.zeros((1, 288, 10)))

    h_trained = get_h_cls(enc_trained, windows)

    # 3. global summary — random init
    print('  Extracting global summary — random encoder ...')
    enc_random.trainable = False
    h_random = get_h_cls(enc_random, windows)

    # 4. Clinical features
    therapy, mean_cgm, hour = extract_features(windows, scalers)

    # 5. Isotropy
    print('\n  Isotropy (avg cosine similarity — lower is better):')
    iso_t_m, iso_t_s = isotropy(h_trained)
    iso_r_m, iso_r_s = isotropy(h_random)
    print(f'    Trained : {iso_t_m:.4f} ± {iso_t_s:.4f}')
    print(f'    Random  : {iso_r_m:.4f} ± {iso_r_s:.4f}')

    # 6. Alignment
    print('\n  Alignment (intra-patient vs inter-patient cosine sim):')
    ali_t_in, ali_t_out = alignment(h_trained, pids)
    ali_r_in, ali_r_out = alignment(h_random,  pids)
    print(f'    Trained : intra={ali_t_in:.4f}  inter={ali_t_out:.4f}  '
          f'gap={ali_t_in - ali_t_out:+.4f}')
    print(f'    Random  : intra={ali_r_in:.4f}  inter={ali_r_out:.4f}  '
          f'gap={ali_r_in - ali_r_out:+.4f}')

    # 7. PCA analysis (uses all windows — fast, linear)
    print('\n  PCA analysis ...')
    pca_result = pca_analysis(
        h_trained, h_random, therapy, mean_cgm, hour,
        os.path.join(args.out, 'pca_analysis.png'),
        os.path.join(args.out, 'pca_analysis.json'),
        enc_tag=enc_tag,
    )

    # 8. 2D projection (subsample for speed)
    print(f'\n  2D projection (n={min(args.n_plot, len(windows))}) ...')
    rng = np.random.default_rng(SEED)
    sel = rng.choice(len(windows), min(args.n_plot, len(windows)), replace=False)
    emb_t, method = reduce_2d(h_trained[sel])
    emb_r, _      = reduce_2d(h_random[sel])
    plot_embeddings(emb_t, emb_r, therapy[sel], mean_cgm[sel], hour[sel],
                    method, os.path.join(args.out, 'embedding_analysis.png'),
                    enc_tag=enc_tag)

    # 9. Save JSON
    summary = {
        'n_windows':  int(len(windows)),
        'n_patients': int(pids.max() + 1),
        'isotropy': {
            'trained': round(iso_t_m, 5),
            'random':  round(iso_r_m, 5),
            'note':    'lower = more isotropic (better distribution)',
        },
        'alignment': {
            'trained_intra': round(ali_t_in,  5),
            'trained_inter': round(ali_t_out, 5),
            'trained_gap':   round(ali_t_in - ali_t_out, 5),
            'random_intra':  round(ali_r_in,  5),
            'random_inter':  round(ali_r_out, 5),
            'random_gap':    round(ali_r_in - ali_r_out, 5),
            'note':          'positive gap = same-patient windows cluster together',
        },
        'pca': pca_result,
    }
    json_path = os.path.join(args.out, 'embedding_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  Saved: {json_path}')

    print(f'\n{"="*52}')
    print(f'  Done.')
    print(f'{"="*52}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',            type=str,
                        default='data/processed/adults')
    parser.add_argument('--weights',         type=str,
                        default='results/mtsm/encoder2/encoder_weights.weights.h5')
    parser.add_argument('--out',             type=str,
                        default='results/mtsm/encoder2')
    parser.add_argument('--max_patients',    type=int, default=200,
                        help='Patient subset to analyse (speed vs coverage)')
    parser.add_argument('--max_per_patient', type=int, default=25,
                        help='Max windows per patient')
    parser.add_argument('--n_plot',          type=int, default=2000,
                        help='Windows to subsample for 2D projection')
    parser.add_argument('--encoder3',        action='store_true', default=False,
                        help='Use encoder3 (PrefixLM, h_last) instead of encoder2')
    args = parser.parse_args()
    main(args)
