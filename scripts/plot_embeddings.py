"""
plot_embeddings.py
==================
Paper-ready plots for the patient embedding study.
Loads cached embeddings from results/embedding_study/ — no forward pass needed.

Run inside docker from /mnt/workspace/tvae:
  python -u scripts/plot_embeddings.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'legend.frameon':     False,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
})

# ── Palette ────────────────────────────────────────────────────────────────────
C_ENC = '#1F77B4'    # encoder blue
C_DEC = '#D62728'    # decoder red
C_RAW = '#7F7F7F'    # raw stats grey

# GRI quartile: green → red (colorblind-safe, RdYlGn-inspired)
GRI_COLORS  = ['#1A9641', '#A6D96A', '#FDAE61', '#D7191C']
GRI_LABELS  = ['Q1 — low risk', 'Q2', 'Q3', 'Q4 — high risk']

# Therapy
THERAPY_COLORS = ['#2166AC', '#4DAC26', '#D01C8B']
THERAPY_LABELS = ['AID', 'SAP', 'MDI']

SEED = 42

# ── Paths ──────────────────────────────────────────────────────────────────────
EMB_DIR  = 'results/embedding_study'
OUT_DIR  = os.path.join(EMB_DIR, 'plots_paper')
os.makedirs(OUT_DIR, exist_ok=True)


# ── Load cached data ───────────────────────────────────────────────────────────

def load_data():
    enc_embs = np.load(os.path.join(EMB_DIR, 'encoder_embeddings.npy'))
    dec_embs = np.load(os.path.join(EMB_DIR, 'decoder_embeddings.npy'))
    clin     = np.load(os.path.join(EMB_DIR, 'clinical_vars.npz'), allow_pickle=True)
    gri      = clin['gri']
    tir      = clin['tir'] * 100   # → percentage
    therapy  = clin['therapy']
    cv       = clin['cv']
    print(f'  Loaded: {len(enc_embs)} patients')
    print(f'  GRI: mean={gri.mean():.1f}  range=[{gri.min():.1f}, {gri.max():.1f}]')
    print(f'  TIR: mean={tir.mean():.1f}%')
    return enc_embs, dec_embs, gri, tir, therapy, cv


# ── UMAP helpers ───────────────────────────────────────────────────────────────

def run_umap_2d(emb, seed=SEED):
    import umap
    return umap.UMAP(n_components=2, random_state=seed,
                     n_neighbors=15, min_dist=0.1).fit_transform(emb)


def run_umap_3d(emb, seed=SEED):
    import umap
    return umap.UMAP(n_components=3, random_state=seed,
                     n_neighbors=15, min_dist=0.1).fit_transform(emb)


def gri_quartile(gri):
    """0-indexed quartile labels (0=best, 3=worst)."""
    bins = np.percentile(gri, [25, 50, 75])
    return np.digitize(gri, bins)


# ── Figure 1: 2D UMAP — encoder + decoder (2 rows × 3 cols) ───────────────────

def fig_umap_2d(enc_coords, dec_coords, gri, tir, therapy, n):
    gri_q = gri_quartile(gri)

    fig = plt.figure(figsize=(13, 8))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.28)

    row_labels = ['Encoder ($h_{cls}$, BERT-style)', 'Decoder (mean-pool $H$, GPT-style)']
    coords_list = [enc_coords, dec_coords]

    for row, (coords, rlbl) in enumerate(zip(coords_list, row_labels)):
        ax0 = fig.add_subplot(gs[row, 0])
        ax1 = fig.add_subplot(gs[row, 1])
        ax2 = fig.add_subplot(gs[row, 2])

        # Panel A: GRI quartile
        for q, color, label in zip(range(4), GRI_COLORS, GRI_LABELS):
            mask = gri_q == q
            ax0.scatter(coords[mask, 0], coords[mask, 1],
                        c=color, s=8, alpha=0.65, linewidths=0, label=label)
        ax0.legend(loc='best', markerscale=1.5,
                   handletextpad=0.4, borderpad=0.5, labelspacing=0.3)
        ax0.set_title('GRI quartile')

        # Panel B: TIR continuous
        sc = ax1.scatter(coords[:, 0], coords[:, 1],
                         c=tir, cmap='RdYlGn', s=8, alpha=0.65,
                         linewidths=0, vmin=20, vmax=90)
        cb = plt.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
        cb.set_label('TIR (%)', fontsize=10)
        cb.ax.tick_params(labelsize=9)
        ax1.set_title('Time In Range')

        # Panel C: therapy modality
        for t, color, label in zip(range(3), THERAPY_COLORS, THERAPY_LABELS):
            mask = therapy == t
            if mask.sum() > 0:
                ax2.scatter(coords[mask, 0], coords[mask, 1],
                            c=color, s=8, alpha=0.65, linewidths=0, label=label)
        ax2.legend(loc='best', markerscale=1.5,
                   handletextpad=0.4, borderpad=0.5, labelspacing=0.3)
        ax2.set_title('Therapy modality')

        for ax in [ax0, ax1, ax2]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('UMAP-1', labelpad=2)
        ax0.set_ylabel('UMAP-2', labelpad=2)
        ax0.text(-0.18, 0.5, rlbl, transform=ax0.transAxes,
                 fontsize=11, fontweight='bold', va='center',
                 rotation=90, ha='center')

    fig.text(0.02, 0.98, f'n = {n} patients', fontsize=9,
             ha='left', va='top', color='#555555')

    path = os.path.join(OUT_DIR, 'umap_2d.png')
    fig.savefig(path)
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 2: 3D UMAP — interactive plotly (encoder only) ─────────────────────

def fig_umap_3d_interactive(coords_3d, gri, tir, n):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print('  [WARN] plotly not installed — skipping interactive 3D UMAP')
        return

    gri_q      = gri_quartile(gri)
    quartile_colors = {0: '#1A9641', 1: '#A6D96A', 2: '#FDAE61', 3: '#D7191C'}
    quartile_names  = {0: 'Q1 — low risk', 1: 'Q2', 2: 'Q3', 3: 'Q4 — high risk'}

    x, y, z = coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2]
    hover   = [f'GRI: {g:.1f}<br>TIR: {t:.1f}%<br>Quartile: {quartile_names[q]}'
               for g, t, q in zip(gri, tir, gri_q)]

    # ── Traces: one per GRI quartile (for categorical view) ──────────────────
    gri_traces = []
    for q in range(4):
        mask = gri_q == q
        gri_traces.append(go.Scatter3d(
            x=x[mask], y=y[mask], z=z[mask],
            mode='markers',
            marker=dict(size=4, color=quartile_colors[q], opacity=0.8),
            name=quartile_names[q],
            text=[hover[i] for i in np.where(mask)[0]],
            hovertemplate='%{text}<extra></extra>',
            visible=True,
        ))

    # ── Trace: continuous TIR coloring (for TIR view) ────────────────────────
    tir_trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=4,
            color=tir,
            colorscale='RdYlGn',
            cmin=20, cmax=90,
            opacity=0.8,
            colorbar=dict(title='TIR (%)', thickness=15, len=0.6),
        ),
        name='TIR (%)',
        text=hover,
        hovertemplate='%{text}<extra></extra>',
        visible=False,
        showlegend=False,
    )

    all_traces = gri_traces + [tir_trace]
    n_gri = len(gri_traces)   # 4

    # ── Toggle buttons ────────────────────────────────────────────────────────
    buttons = [
        dict(
            label='GRI Quartile',
            method='update',
            args=[
                {'visible': [True] * n_gri + [False]},
                {'title': f'3D UMAP — Encoder (h_cls) · GRI Quartile · n={n}'},
            ],
        ),
        dict(
            label='Time In Range (%)',
            method='update',
            args=[
                {'visible': [False] * n_gri + [True]},
                {'title': f'3D UMAP — Encoder (h_cls) · Time In Range · n={n}'},
            ],
        ),
    ]

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title=dict(text=f'3D UMAP — Encoder (h_cls) · GRI Quartile · n={n}',
                   font=dict(size=14)),
        scene=dict(
            xaxis=dict(title='UMAP-1', showticklabels=False, backgroundcolor='white'),
            yaxis=dict(title='UMAP-2', showticklabels=False, backgroundcolor='white'),
            zaxis=dict(title='UMAP-3', showticklabels=False, backgroundcolor='white'),
            bgcolor='white',
        ),
        updatemenus=[dict(
            type='buttons',
            direction='left',
            x=0.0, y=1.08, xanchor='left',
            buttons=buttons,
            bgcolor='#f0f0f0',
            bordercolor='#cccccc',
            font=dict(size=12),
            showactive=True,
            active=0,
        )],
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#cccccc', borderwidth=1),
        margin=dict(l=0, r=0, t=80, b=0),
        height=700,
        paper_bgcolor='white',
    )

    path = os.path.join(OUT_DIR, 'umap_3d_encoder.html')
    fig.write_html(path, include_plotlyjs=True)
    print(f'  Saved: {path}')



# ── Figure 3: PCA effective dimensionality ─────────────────────────────────────

def fig_pca(enc_embs, dec_embs):
    pca_enc = PCA().fit(enc_embs)
    pca_dec = PCA().fit(dec_embs)

    cumvar_enc = np.cumsum(pca_enc.explained_variance_ratio_) * 100
    cumvar_dec = np.cumsum(pca_dec.explained_variance_ratio_) * 100

    # Find 90% threshold components
    n90_enc = int(np.searchsorted(cumvar_enc, 90) + 1)
    n90_dec = int(np.searchsorted(cumvar_dec, 90) + 1)

    MAX_K = 35   # only show first 35 components (story is here)
    x = np.arange(1, MAX_K + 1)

    fig, ax = plt.subplots(figsize=(6.5, 4))

    ax.plot(x, cumvar_enc[:MAX_K], color=C_ENC, lw=2,
            label=f'Encoder ($h_{{cls}}$)')
    ax.plot(x, cumvar_dec[:MAX_K], color=C_DEC, lw=2,
            label=f'Decoder (mean-pool $H$)')

    ax.axhline(90, color='#AAAAAA', lw=1, ls='--')
    ax.text(MAX_K + 0.3, 90, '90%', va='center', ha='left',
            fontsize=9, color='#888888')

    # Annotate threshold crossings
    ax.axvline(n90_enc, color=C_ENC, lw=1, ls=':', alpha=0.7)
    ax.axvline(n90_dec, color=C_DEC, lw=1, ls=':', alpha=0.7)
    ax.text(n90_enc, 30, f' {n90_enc} dims', color=C_ENC,
            fontsize=9, va='bottom', ha='left')
    ax.text(n90_dec, 30, f' {n90_dec} dims', color=C_DEC,
            fontsize=9, va='bottom', ha='left')

    ax.set_xlabel('Number of PCA components')
    ax.set_ylabel('Cumulative explained variance (%)')
    ax.set_xlim(1, MAX_K)
    ax.set_ylim(35, 101)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.25)

    path = os.path.join(OUT_DIR, 'pca_variance.png')
    fig.savefig(path)
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 4: Linear probe ─────────────────────────────────────────────────────

def fig_linear_probe(enc_embs, dec_embs, gri, tir, cv):
    raw_feat = np.column_stack([tir, cv, 1 - tir/100 - cv])

    def r2_cv(X, y):
        pipe = make_pipeline(StandardScaler(), Ridge())
        return cross_val_score(pipe, X, y, cv=5, scoring='r2').mean()

    r2_enc     = r2_cv(enc_embs, gri)
    r2_dec     = r2_cv(dec_embs, gri)
    r2_raw     = r2_cv(raw_feat,  gri)

    labels = ['Encoder\n($h_{cls}$)', 'Decoder\n(mean-pool $H$)', 'Raw CGM\nstatistics']
    values = [r2_enc, r2_dec, r2_raw]
    colors = [C_ENC, C_DEC, C_RAW]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(labels, values, color=colors, width=0.5, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11,
                fontweight='bold')

    ax.set_ylabel('$R^2$ (5-fold CV)')
    ax.set_ylim(0.88, 1.01)
    ax.set_title('Linear probe: embedding → GRI')
    ax.grid(axis='y', alpha=0.25, zorder=0)
    ax.axhline(r2_raw, color=C_RAW, lw=1, ls='--', alpha=0.6, zorder=2)
    ax.text(2.52, r2_raw + 0.001, 'raw baseline', color=C_RAW,
            fontsize=9, va='bottom', ha='right')

    path = os.path.join(OUT_DIR, 'linear_probe.png')
    fig.savefig(path)
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 5: KNN consistency ──────────────────────────────────────────────────

def knn_consistency_score(emb, gri, k=10):
    quartiles = np.digitize(gri, np.percentile(gri, [25, 50, 75]))
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    _, idx = nbrs.kneighbors(emb)
    idx = idx[:, 1:]
    return float(np.mean([(quartiles[idx[i]] == quartiles[i]).mean()
                          for i in range(len(emb))]))


def fig_knn(enc_embs, dec_embs, gri):
    k_vals = [5, 10, 15, 20, 30]
    random_baseline = 0.25   # 4 quartiles, uniform

    enc_scores = [knn_consistency_score(enc_embs, gri, k) for k in k_vals]
    dec_scores = [knn_consistency_score(dec_embs, gri, k) for k in k_vals]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(k_vals, enc_scores, 'o-', color=C_ENC, lw=2,
            label='Encoder ($h_{cls}$)', markersize=6)
    ax.plot(k_vals, dec_scores, 's-', color=C_DEC, lw=2,
            label='Decoder (mean-pool $H$)', markersize=6)
    ax.axhline(random_baseline, color='#AAAAAA', lw=1.5, ls='--',
               label='Random baseline (0.25)')

    ax.set_xlabel('K (number of neighbours)')
    ax.set_ylabel('GRI quartile consistency')
    ax.set_ylim(0.15, 0.80)
    ax.set_xticks(k_vals)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.25)

    path = os.path.join(OUT_DIR, 'knn_consistency.png')
    fig.savefig(path)
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 6: Geometry summary table ──────────────────────────────────────────

def fig_geometry(enc_embs, dec_embs, gri, tir, cv):
    def isotropy(emb, n_pairs=10_000):
        from sklearn.preprocessing import normalize
        e = normalize(emb)
        idx = np.random.randint(0, len(e), size=(n_pairs, 2))
        sims = (e[idx[:, 0]] * e[idx[:, 1]]).sum(axis=1)
        return float(sims.mean()), float(sims.std())

    def lid_mle(emb, k=20):
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(emb)
        d, _ = nbrs.kneighbors(emb)
        d = d[:, 1:]
        r_k = d[:, -1:]
        ratios = np.clip(d / (r_k + 1e-10), 1e-10, 1.0)
        return float((-1.0 / (np.log(ratios).mean(axis=1) + 1e-10)).mean())

    def n90(emb):
        pca = PCA().fit(emb)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        return int(np.searchsorted(cumvar, 0.90) + 1)

    iso_enc_mean, iso_enc_std = isotropy(enc_embs)
    iso_dec_mean, iso_dec_std = isotropy(dec_embs)

    metrics = {
        'PCA dims\n(90% var)': (n90(enc_embs), n90(dec_embs)),
        'Isotropy\n(mean cos↓)': (iso_enc_mean, iso_dec_mean),
        'LID\n(k=20, ↑=richer)': (lid_mle(enc_embs), lid_mle(dec_embs)),
        'KNN consist.\n(k=10, ↑=better)': (
            knn_consistency_score(enc_embs, gri),
            knn_consistency_score(dec_embs, gri)),
    }

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.8))
    fig.suptitle('Embedding space geometry', fontsize=12, y=1.02)

    fmt_map = {
        'PCA dims\n(90% var)': '{:.0f}',
        'Isotropy\n(mean cos↓)': '{:.4f}',
        'LID\n(k=20, ↑=richer)': '{:.2f}',
        'KNN consist.\n(k=10, ↑=better)': '{:.3f}',
    }
    ylim_map = {
        'PCA dims\n(90% var)':        (0, 12),
        'Isotropy\n(mean cos↓)':       (0.88, 0.95),
        'LID\n(k=20, ↑=richer)':      (0, 9),
        'KNN consist.\n(k=10, ↑=better)': (0, 0.80),
    }

    for ax, (metric, (enc_val, dec_val)) in zip(axes, metrics.items()):
        bars = ax.bar(['Encoder', 'Decoder'], [enc_val, dec_val],
                      color=[C_ENC, C_DEC], width=0.5, zorder=3)
        fmt = fmt_map[metric]
        for bar, val in zip(bars, [enc_val, dec_val]):
            offset = (ylim_map[metric][1] - ylim_map[metric][0]) * 0.02
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + offset,
                    fmt.format(val),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_title(metric, fontsize=10)
        ax.set_ylim(*ylim_map[metric])
        ax.grid(axis='y', alpha=0.25, zorder=0)
        ax.tick_params(axis='x', labelsize=10)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'geometry_summary.png')
    fig.savefig(path)
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 7: Consistency score analysis ──────────────────────────────────────

def fig_consistency(enc_coords, gri, tir, therapy, spread):
    from scipy import stats

    valid = ~np.isnan(spread)
    s     = spread[valid]
    g     = gri[valid]
    t     = therapy[valid]
    ec    = enc_coords[valid]

    therapy_labels = ['AID', 'SAP', 'MDI']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle('Intra-patient glycaemic consistency  —  Encoder ($h_{cls}$)',
                 fontsize=12, y=1.02)

    # Panel 1: scatter spread vs GRI, coloured by therapy
    for ti, (label, color) in enumerate(zip(therapy_labels, THERAPY_COLORS)):
        mask = t == ti
        axes[0].scatter(s[mask], g[mask], c=color, s=10, alpha=0.55,
                        linewidths=0, label=label)
    r, p = stats.pearsonr(s, g)
    pstr  = f'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
    axes[0].text(0.97, 0.05, f'r = {r:.2f}\n{pstr}',
                 transform=axes[0].transAxes, ha='right', va='bottom',
                 fontsize=10, bbox=dict(boxstyle='round,pad=0.3',
                                        fc='white', ec='#cccccc', alpha=0.8))
    axes[0].set_xlabel('Intra-patient spread (mean cosine dist.)')
    axes[0].set_ylabel('GRI')
    axes[0].set_title('Spread vs GRI')
    axes[0].legend(markerscale=2, loc='upper left')
    axes[0].grid(alpha=0.2)

    # Panel 2: violin plot by therapy
    data_by_therapy = [s[t == ti] for ti in range(3)]
    parts = axes[1].violinplot(data_by_therapy, positions=range(3),
                               showmedians=True, showextrema=False)
    for pc, color in zip(parts['bodies'], THERAPY_COLORS):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)

    # Overlay individual points
    for ti, color in enumerate(THERAPY_COLORS):
        jitter = np.random.normal(0, 0.06, size=(t == ti).sum())
        axes[1].scatter(ti + jitter, s[t == ti], c=color, s=5,
                        alpha=0.3, linewidths=0)

    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(therapy_labels)
    axes[1].set_ylabel('Intra-patient spread')
    axes[1].set_title('Spread by therapy')
    axes[1].grid(axis='y', alpha=0.2)

    # Annotate medians
    for ti, d in enumerate(data_by_therapy):
        axes[1].text(ti, np.median(d) + 0.0005, f'{np.median(d):.4f}',
                     ha='center', va='bottom', fontsize=8)

    # Panel 3: UMAP coloured by spread
    sc = axes[2].scatter(ec[:, 0], ec[:, 1], c=s,
                         cmap='YlOrRd', s=8, alpha=0.65, linewidths=0)
    cb = plt.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.04)
    cb.set_label('Intra-patient spread', fontsize=10)
    cb.ax.tick_params(labelsize=9)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_xlabel('UMAP-1', labelpad=2)
    axes[2].set_ylabel('UMAP-2', labelpad=2)
    axes[2].set_title('UMAP coloured by spread')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'consistency.png')
    fig.savefig(path)
    plt.close()
    print(f'  Saved: {path}')

    # Print summary stats
    print(f'\n  Consistency summary:')
    for ti, label in enumerate(therapy_labels):
        d = s[t == ti]
        print(f'    {label}: median={np.median(d):.4f}  mean={d.mean():.4f}  n={len(d)}')
    print(f'  Pearson r (spread vs GRI): {r:.3f}  {pstr}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(SEED)

    print('\n  Loading cached embeddings...')
    enc_embs, dec_embs, gri, tir, therapy, cv = load_data()
    n = len(enc_embs)

    print('\n  Computing 2D UMAP...')
    enc_2d = run_umap_2d(enc_embs)
    dec_2d = run_umap_2d(dec_embs)

    print('  Computing 3D UMAP (encoder only)...')
    enc_3d = run_umap_3d(enc_embs)

    consist_path = os.path.join(EMB_DIR, 'consistency_scores.npy')
    spread = np.load(consist_path) if os.path.exists(consist_path) else None

    print('\n  Plotting...')
    fig_umap_2d(enc_2d, dec_2d, gri, tir, therapy, n)
    fig_umap_3d_interactive(enc_3d, gri, tir, n)
    fig_pca(enc_embs, dec_embs)
    fig_linear_probe(enc_embs, dec_embs, gri, tir, cv)
    fig_knn(enc_embs, dec_embs, gri)
    fig_geometry(enc_embs, dec_embs, gri, tir, cv)

    if spread is not None:
        fig_consistency(enc_2d, gri, tir, therapy, spread)
    else:
        print('  [SKIP] consistency_scores.npy not found — run embedding_study.py first')

    print(f'\n  All plots saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
