"""
Final 3D UMAP figure: 2×3 grid.
Rows: encoder h_cls / decoder H.
Columns: 3 viewing angles.
Coords loaded from cached npz.
"""
import os, sys
sys.path.insert(0, '/mnt/workspace/tvae')
os.chdir('/mnt/workspace/tvae')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

np.random.seed(42)

d   = np.load('results/embedding_study_global_norm/umap_3d_coords.npz')
cv  = np.load('results/embedding_study_global_norm/clinical_vars.npz')
emb_enc = d['enc']
emb_dec = d['dec']
gri     = cv['gri']

quintiles = np.zeros(len(gri), dtype=int)
for q, (lo, hi) in enumerate(zip(
        np.percentile(gri, [0, 20, 40, 60, 80]),
        np.percentile(gri, [20, 40, 60, 80, 100])), start=1):
    quintiles[(gri >= lo) & (gri <= hi)] = q

Q_COLORS = ['#2ca02c', '#98df8a', '#ffbb78', '#ff7f0e', '#d62728']
Q_LABELS = ['Q1 (low risk)', 'Q2', 'Q3', 'Q4', 'Q5 (high risk)']

ANGLES = [
    (20,  45),
    (30, -45),
    (15, 135),
]

plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.size':        9,
    'axes.titlesize':   9.5,
    'figure.facecolor': 'white',
})

fig = plt.figure(figsize=(15, 10))

for row, (emb, row_title) in enumerate(zip(
        [emb_enc, emb_dec],
        ['Encoder $h_{\\mathrm{cls}}$', 'Decoder $\\bar{H}$'])):
    for col, (elev, azim) in enumerate(ANGLES):
        ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection='3d')
        ax.set_facecolor('white')

        for q in range(1, 6):
            mask = quintiles == q
            ax.scatter(emb[mask, 0], emb[mask, 1], emb[mask, 2],
                       c=Q_COLORS[q - 1], s=9, alpha=0.75,
                       linewidths=0, depthshade=True,
                       label=Q_LABELS[q - 1] if (row == 0 and col == 0) else None)

        ax.set_xlabel('UMAP-1', labelpad=3, fontsize=8)
        ax.set_ylabel('UMAP-2', labelpad=3, fontsize=8)
        ax.set_zlabel('UMAP-3', labelpad=3, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#cccccc')
        ax.yaxis.pane.set_edgecolor('#cccccc')
        ax.zaxis.pane.set_edgecolor('#cccccc')
        ax.grid(True, color='#dddddd', linewidth=0.35)
        ax.view_init(elev=elev, azim=azim)

        # Row label only on first column
        title = f'{row_title}' if col == 0 else ''
        ax.set_title(title, fontsize=9.5, pad=6)

# Shared legend at bottom
handles = [plt.Line2D([0], [0], marker='o', color='w',
           markerfacecolor=c, markersize=7, label=l)
           for c, l in zip(Q_COLORS, Q_LABELS)]
fig.legend(handles=handles, loc='lower center', ncol=5,
           fontsize=9, framealpha=0.9, edgecolor='#cccccc',
           bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(pad=1.5, w_pad=0.5, h_pad=2.0, rect=[0, 0.06, 1, 1])
out = 'results/embedding_study_global_norm/umap_3d_2x3.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved {out}')
