"""
Save 3D UMAP coords once, then render a 2×4 angle comparison grid.
Rows: encoder / decoder. Columns: 4 candidate viewing angles.
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

COORDS_PATH = 'results/embedding_study_global_norm/umap_3d_coords.npz'

enc = np.load('results/embedding_study_global_norm/encoder_embeddings.npy')
dec = np.load('results/embedding_study_global_norm/decoder_embeddings.npy')
cv  = np.load('results/embedding_study_global_norm/clinical_vars.npz')
gri = cv['gri']

quintiles = np.zeros(len(gri), dtype=int)
for q, (lo, hi) in enumerate(zip(
        np.percentile(gri, [0, 20, 40, 60, 80]),
        np.percentile(gri, [20, 40, 60, 80, 100])), start=1):
    quintiles[(gri >= lo) & (gri <= hi)] = q

Q_COLORS = ['#2ca02c', '#98df8a', '#ffbb78', '#ff7f0e', '#d62728']
Q_LABELS = ['Q1 (low risk)', 'Q2', 'Q3', 'Q4', 'Q5 (high risk)']

if os.path.exists(COORDS_PATH):
    print('Loading cached 3D UMAP coords...')
    d = np.load(COORDS_PATH)
    emb_enc, emb_dec = d['enc'], d['dec']
else:
    print('Computing 3D UMAP for encoder...')
    emb_enc = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                        random_state=42).fit_transform(enc)
    print('Computing 3D UMAP for decoder...')
    emb_dec = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                        random_state=42).fit_transform(dec)
    np.savez(COORDS_PATH, enc=emb_enc, dec=emb_dec)
    print(f'Saved coords to {COORDS_PATH}')

# ── Candidate viewing angles ───────────────────────────────────────────────────
# (elev, azim, label)
ANGLES = [
    (20,  45,  'elev=20  azim=45'),
    (30, -45,  'elev=30  azim=-45'),
    (15, 135,  'elev=15  azim=135'),
    (35, 220,  'elev=35  azim=220'),
]

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8,
                     'figure.facecolor': 'white'})

fig = plt.figure(figsize=(18, 9))

for row, (emb, row_title) in enumerate(zip([emb_enc, emb_dec],
                                            ['Encoder $h_{cls}$', 'Decoder $\\bar{H}$'])):
    for col, (elev, azim, ang_label) in enumerate(ANGLES):
        ax = fig.add_subplot(2, 4, row * 4 + col + 1, projection='3d')
        ax.set_facecolor('white')
        for q in range(1, 6):
            mask = quintiles == q
            ax.scatter(emb[mask, 0], emb[mask, 1], emb[mask, 2],
                       c=Q_COLORS[q - 1], s=6, alpha=0.7,
                       linewidths=0, depthshade=True,
                       label=Q_LABELS[q - 1] if (row == 0 and col == 0) else None)
        ax.set_xlabel('U1', labelpad=2, fontsize=7)
        ax.set_ylabel('U2', labelpad=2, fontsize=7)
        ax.set_zlabel('U3', labelpad=2, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#cccccc')
        ax.yaxis.pane.set_edgecolor('#cccccc')
        ax.zaxis.pane.set_edgecolor('#cccccc')
        ax.grid(True, color='#dddddd', linewidth=0.3)
        ax.view_init(elev=elev, azim=azim)
        title = f'{row_title}  |  {ang_label}' if col == 0 else ang_label
        ax.set_title(title, fontsize=7.5, pad=4)

handles = [plt.Line2D([0],[0], marker='o', color='w',
           markerfacecolor=c, markersize=6, label=l)
           for c, l in zip(Q_COLORS, Q_LABELS)]
fig.legend(handles=handles, loc='lower center', ncol=5,
           fontsize=8, framealpha=0.9, edgecolor='#cccccc',
           bbox_to_anchor=(0.5, 0.0))

plt.tight_layout(pad=1.0, rect=[0, 0.04, 1, 1])
out = 'results/embedding_study_global_norm/umap_3d_angles.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
