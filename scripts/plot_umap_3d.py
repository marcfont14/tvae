"""
3D UMAP figure: encoder h_cls and decoder H, coloured by GRI quintile.
2-panel side by side using mplot3d. Approved thesis aesthetic.
"""
import os, sys
sys.path.insert(0, '/mnt/workspace/tvae')
os.chdir('/mnt/workspace/tvae')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
import umap

np.random.seed(42)

enc = np.load('results/embedding_study_global_norm/encoder_embeddings.npy')
dec = np.load('results/embedding_study_global_norm/decoder_embeddings.npy')
cv  = np.load('results/embedding_study_global_norm/clinical_vars.npz')
gri = cv['gri']

# GRI quintiles (1=best, 5=worst)
quintiles = np.zeros(len(gri), dtype=int)
for q, (lo, hi) in enumerate(zip(
        np.percentile(gri, [0, 20, 40, 60, 80]),
        np.percentile(gri, [20, 40, 60, 80, 100])), start=1):
    mask = (gri >= lo) & (gri <= hi)
    quintiles[mask] = q

Q_COLORS = ['#2ca02c', '#98df8a', '#ffbb78', '#ff7f0e', '#d62728']
Q_LABELS = ['Q1 (low risk)', 'Q2', 'Q3', 'Q4', 'Q5 (high risk)']

print('Computing 3D UMAP for encoder...')
emb_enc = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                    random_state=42).fit_transform(enc)

print('Computing 3D UMAP for decoder...')
emb_dec = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                    random_state=42).fit_transform(dec)

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.size':        10,
    'axes.labelsize':   9,
    'axes.titlesize':   10,
    'legend.fontsize':  8.5,
    'figure.facecolor': 'white',
})

fig = plt.figure(figsize=(14, 6))

ELEV, AZIM = 22, 45

for col, (emb, title) in enumerate(zip(
        [emb_enc, emb_dec],
        ['Encoder $h_{\\mathrm{cls}}$', 'Decoder $\\bar{H}$'])):

    ax = fig.add_subplot(1, 2, col + 1, projection='3d')
    ax.set_facecolor('white')

    for q in range(1, 6):
        mask = quintiles == q
        ax.scatter(emb[mask, 0], emb[mask, 1], emb[mask, 2],
                   c=Q_COLORS[q - 1], s=8, alpha=0.75,
                   label=Q_LABELS[q - 1], linewidths=0,
                   depthshade=True)

    ax.set_xlabel('UMAP-1', labelpad=4, fontsize=8.5)
    ax.set_ylabel('UMAP-2', labelpad=4, fontsize=8.5)
    ax.set_zlabel('UMAP-3', labelpad=4, fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.set_title(title, pad=10)
    ax.view_init(elev=ELEV, azim=AZIM)

    # Subtle pane colours
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#cccccc')
    ax.yaxis.pane.set_edgecolor('#cccccc')
    ax.zaxis.pane.set_edgecolor('#cccccc')
    ax.grid(True, color='#dddddd', linewidth=0.4)

    if col == 0:
        ax.legend(loc='upper left', fontsize=7.5, markerscale=1.8,
                  framealpha=0.85, edgecolor='#cccccc',
                  bbox_to_anchor=(-0.05, 1.0))

plt.tight_layout(pad=1.0, w_pad=0.5)
out = 'results/embedding_study_global_norm/umap_3d.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved {out}')
