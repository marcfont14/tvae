"""
Clean UMAP figure: encoder h_cls and decoder H embeddings, coloured by GRI quintile.
2-panel side by side. Approved thesis aesthetic.
"""
import os, sys
sys.path.insert(0, '/mnt/workspace/tvae')
os.chdir('/mnt/workspace/tvae')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import umap

np.random.seed(42)

enc = np.load('results/embedding_study_global_norm/encoder_embeddings.npy')   # (1037,128)
dec = np.load('results/embedding_study_global_norm/decoder_embeddings.npy')   # (1037,128)
cv  = np.load('results/embedding_study_global_norm/clinical_vars.npz')
gri = cv['gri']   # (1037,)

# GRI quintile labels (1=best, 5=worst)
quintiles = np.zeros(len(gri), dtype=int)
for q, (lo, hi) in enumerate(zip(
    np.percentile(gri, [0, 20, 40, 60, 80]),
    np.percentile(gri, [20, 40, 60, 80, 100])
), start=1):
    mask = (gri >= lo) & (gri <= hi)
    quintiles[mask] = q

print('Computing UMAP for encoder...')
reducer_enc = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
emb_enc = reducer_enc.fit_transform(enc)

print('Computing UMAP for decoder...')
reducer_dec = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
emb_dec = reducer_dec.fit_transform(dec)

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.size':        10,
    'axes.labelsize':   10,
    'axes.titlesize':   9.5,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'legend.fontsize':  8.5,
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
})

# GRI quintile colours: green (low risk) → red (high risk)
Q_COLORS = ['#2ca02c', '#98df8a', '#ffbb78', '#ff7f0e', '#d62728']
Q_LABELS = ['Q1 (low risk)', 'Q2', 'Q3', 'Q4', 'Q5 (high risk)']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, emb, title in zip(axes,
                           [emb_enc, emb_dec],
                           ['Encoder $h_{cls}$', 'Decoder $\overline{H}$']):
    for q in range(1, 6):
        mask = quintiles == q
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=Q_COLORS[q-1], s=8, alpha=0.7,
                   label=Q_LABELS[q-1], linewidths=0)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.set_title(title)

axes[0].legend(loc='upper right', fontsize=8, markerscale=2,
               framealpha=0.9, edgecolor='#cccccc')

plt.tight_layout(pad=1.5, w_pad=2.5)
out = 'results/embedding_study_global_norm/umap_clean.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved {out}')
