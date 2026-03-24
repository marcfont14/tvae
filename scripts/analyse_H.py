"""
analyse_H.py
============
Deep analysis of the Transformer encoder's H representation (Stage 1).

Loads encoder weights from a trained run, reconstructs the exact test split
(same SEED + patient-level split as experiment_mtsm.py), and generates:

  1.  pca_by_modality.png             — mean-pooled H projected to 2D, coloured by AID/SAP/MDI
  2.  pca_by_age.png                  — same projection, coloured by age group
  3.  pca_per_layer.png               — PCA scatter at each of the 5 encoder layers
  4.  tsne_by_modality.png            — t-SNE of mean-pooled H, coloured by modality
  5.  H_norm_vs_drivers.png           — event-triggered average of H_t norm around bolus/carbs
  6.  attention_deep.png              — average attention + per-head heatmaps + entropy
  7.  H_norm_circadian.png            — average H norm by hour of day (circadian pattern)
  8.  H_norm_feature_correlation.png  — Pearson r distributions between H norm and each feature
  9.  layer_attention_matrices.png    — average attention heatmap per encoder layer (1×5 grid)
  10. layer_event_triggered.png       — event-triggered H_t norm per layer (2×5 grid)
  11. layer_reconstruction_probe.png  — linear probe R² and MAE per layer (bar chart)
  12. layer_feature_correlation.png   — median Pearson r heatmap (layers × features)

Usage:
  python scripts/analyse_H.py \\
      --data   data/processed/adults \\
      --run_id run12 \\
      --n_windows 1500   # total windows to sample across test patients
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ── Must match experiment_mtsm.py exactly ─────────────────────────────────────
WINDOW_LEN  = 288
N_FEATURES  = 11
CGM_IDX     = 0
PI_IDX      = 1
RA_IDX      = 2
BOLUS_IDX   = 5
CARBS_IDX   = 6
D_MODEL     = 128
N_HEADS     = 4
N_LAYERS    = 5
D_FF        = 256
DROPOUT     = 0.2
VAL_SPLIT   = 0.1
TEST_SPLIT  = 0.1
SEED        = 42
RESULTS_BASE = 'results/mtsm'

COLORS = {
    'cgm':   '#111827',
    'pi':    '#7C3AED',
    'ra':    '#059669',
    'bolus': '#DC2626',
    'carbs': '#D97706',
    'norm':  '#2563EB',
}

MODALITY_COLORS = {'AID': '#2563EB', 'SAP': '#D97706', 'MDI': '#DC2626', 'unknown': '#9CA3AF'}
AGE_BINS   = [18, 30, 45, 60, 100]
AGE_LABELS = ['18–29', '30–44', '45–59', '60+']
AGE_COLORS = ['#6EE7B7', '#34D399', '#059669', '#065F46']

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── Architecture (identical to experiment_mtsm.py) ────────────────────────────

def get_positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    positions = np.arange(seq_len)[:, np.newaxis]
    dims      = np.arange(d_model)[np.newaxis, :]
    angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)


def build_transformer_encoder(window_len, n_features, d_model, n_heads,
                               n_layers, d_ff, dropout):
    inp = keras.Input(shape=(window_len, n_features), name='input')
    x   = layers.Dense(d_model, name='input_proj')(inp)
    pe  = get_positional_encoding(window_len, d_model)
    x   = x + pe
    for i in range(n_layers):
        attn = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads,
            dropout=dropout, name=f'mhsa_{i}'
        )(x, x)
        attn = layers.Dropout(dropout)(attn)
        x    = layers.LayerNormalization(epsilon=1e-6, name=f'norm1_{i}')(x + attn)
        ffn  = layers.Dense(d_ff, activation='relu', name=f'ffn1_{i}')(x)
        ffn  = layers.Dropout(dropout)(ffn)
        ffn  = layers.Dense(d_model, name=f'ffn2_{i}')(ffn)
        ffn  = layers.Dropout(dropout)(ffn)
        x    = layers.LayerNormalization(epsilon=1e-6, name=f'norm2_{i}')(x + ffn)
    return keras.Model(inp, x, name='TransformerEncoder')


# ── Data loading ──────────────────────────────────────────────────────────────

def index_test_patients(processed_dir: str):
    """
    Reconstruct the exact same patient-level test split as experiment_mtsm.py.
    Returns list of (fpath, modality, age) for test patients only.
    """
    npz_files  = sorted([f for f in os.listdir(processed_dir) if f.endswith('.npz')])
    all_fpaths = [os.path.join(processed_dir, f) for f in npz_files]
    n          = len(all_fpaths)
    perm       = np.random.permutation(n)   # same SEED → same permutation
    n_test     = int(n * TEST_SPLIT)
    test_fpaths = [all_fpaths[i] for i in perm[:n_test]]

    records = []
    for fpath in test_fpaths:
        data     = np.load(fpath, allow_pickle=True)
        modality = str(data['modality'].flat[0])
        age      = float(data['age'].flat[0])
        records.append((fpath, modality, age))

    print(f"  Test patients: {len(records)} / {n} total")
    return records


def sample_windows(test_records, n_windows: int):
    """
    Sample up to n_windows from the test patients, applying the same
    pathological window filter used in training.
    Returns:
        windows   (N, 288, 11)
        modalities list[str]  length N
        ages       list[float] length N
    """
    windows_list   = []
    modalities_list = []
    ages_list      = []

    wins_per_patient = max(1, n_windows // len(test_records))
    rng = np.random.default_rng(SEED)

    for fpath, modality, age in test_records:
        data = np.load(fpath, allow_pickle=True)
        wins = data['windows'].astype(np.float32)   # (N_win, 288, 11)

        bolus = wins[:, :, BOLUS_IDX]
        carbs = wins[:, :, CARBS_IDX]
        cgm   = wins[:, :, CGM_IDX]

        has_driver = ((bolus + carbs) > 0).any(axis=1)
        cgm_std    = cgm.std(axis=1)
        cgm_ok     = (cgm_std > 0.3) & (cgm_std < 4.0)
        valid_idx  = np.where(has_driver & cgm_ok)[0]

        if len(valid_idx) == 0:
            continue

        chosen = rng.choice(valid_idx,
                            size=min(wins_per_patient, len(valid_idx)),
                            replace=False)
        windows_list.append(wins[chosen])
        modalities_list.extend([modality] * len(chosen))
        ages_list.extend([age] * len(chosen))

    windows = np.concatenate(windows_list, axis=0)
    print(f"  Sampled windows: {len(windows)}")
    return windows, modalities_list, ages_list


# ── H extraction ──────────────────────────────────────────────────────────────

def get_H_pooled(encoder, windows: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Run encoder and mean-pool over timesteps → (N, d_model)."""
    H_list = []
    for i in range(0, len(windows), batch_size):
        batch = tf.cast(windows[i:i+batch_size], tf.float32)
        H     = encoder(batch, training=False).numpy()   # (B, 288, d_model)
        H_list.append(H.mean(axis=1))                   # (B, d_model)
    return np.concatenate(H_list, axis=0)


def get_H_per_layer(encoder, window: np.ndarray, n_layers: int) -> list:
    """
    Extract mean-pooled H after each encoder layer for a single window.
    Returns list of (288, d_model) arrays, one per layer.
    """
    layer_outputs = []
    x_in = tf.cast(window[np.newaxis], tf.float32)
    for i in range(n_layers):
        m = keras.Model(encoder.input, encoder.get_layer(f'norm2_{i}').output)
        H = m(x_in, training=False).numpy()[0]   # (288, d_model)
        layer_outputs.append(H)
    return layer_outputs


def get_H_pooled_per_layer(encoder, windows: np.ndarray,
                            n_layers: int, batch_size: int = 64) -> list:
    """
    Mean-pooled H after each layer for all windows.
    Returns list of n_layers arrays, each (N, d_model).
    """
    layer_models = [
        keras.Model(encoder.input, encoder.get_layer(f'norm2_{i}').output)
        for i in range(n_layers)
    ]
    results = [[] for _ in range(n_layers)]
    for i in range(0, len(windows), batch_size):
        batch = tf.cast(windows[i:i+batch_size], tf.float32)
        for li, m in enumerate(layer_models):
            H = m(batch, training=False).numpy()   # (B, 288, d_model)
            results[li].append(H.mean(axis=1))     # (B, d_model)
    return [np.concatenate(r, axis=0) for r in results]


# ── Plot helpers ──────────────────────────────────────────────────────────────

def age_label(age: float) -> str:
    for i, (lo, hi) in enumerate(zip(AGE_BINS[:-1], AGE_BINS[1:])):
        if lo <= age < hi:
            return AGE_LABELS[i]
    return AGE_LABELS[-1]


# ── Plot 1 & 2: PCA by modality / age ─────────────────────────────────────────

def plot_pca_metadata(H_pooled: np.ndarray, modalities: list, ages: list,
                      save_dir: str):
    from sklearn.decomposition import PCA

    pca  = PCA(n_components=2, random_state=SEED)
    H_2d = pca.fit_transform(H_pooled)
    var  = pca.explained_variance_ratio_
    xlabel = f'PC1 ({var[0]*100:.1f}% var)'
    ylabel = f'PC2 ({var[1]*100:.1f}% var)'

    # ── PCA by modality ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    for mod, col in MODALITY_COLORS.items():
        idx = [i for i, m in enumerate(modalities) if m == mod]
        if idx:
            ax.scatter(H_2d[idx, 0], H_2d[idx, 1],
                       c=col, label=f'{mod} (n={len(idx)})',
                       s=12, alpha=0.55, linewidths=0)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title('PCA of mean-pooled H — coloured by therapy modality\n'
                 '(test patients only)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, markerscale=2)
    ax.grid(True, alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)
    path = os.path.join(save_dir, 'pca_by_modality.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ── PCA by age ────────────────────────────────────────────────────────────
    age_labels_per_win = [age_label(a) for a in ages]
    fig, ax = plt.subplots(figsize=(9, 7))
    for lbl, col in zip(AGE_LABELS, AGE_COLORS):
        idx = [i for i, al in enumerate(age_labels_per_win) if al == lbl]
        if idx:
            ax.scatter(H_2d[idx, 0], H_2d[idx, 1],
                       c=col, label=f'{lbl} yr (n={len(idx)})',
                       s=12, alpha=0.55, linewidths=0)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title('PCA of mean-pooled H — coloured by age group\n'
                 '(test patients only)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, markerscale=2)
    ax.grid(True, alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)
    path = os.path.join(save_dir, 'pca_by_age.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 3: per-layer PCA ──────────────────────────────────────────────────────

def plot_pca_per_layer(H_per_layer: list, save_dir: str):
    from sklearn.decomposition import PCA

    n_layers = len(H_per_layer)
    pc1_var  = []

    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 5), sharey=False)
    fig.suptitle('Progressive abstraction — PCA of mean-pooled H per encoder layer\n'
                 'Each point = one window (test set)',
                 fontsize=12, fontweight='bold')

    for li, (H, ax) in enumerate(zip(H_per_layer, axes)):
        pca  = PCA(n_components=2, random_state=SEED)
        H_2d = pca.fit_transform(H)
        var  = pca.explained_variance_ratio_
        pc1_var.append(var[0] * 100)

        ax.scatter(H_2d[:, 0], H_2d[:, 1], s=6, alpha=0.4,
                   c='#2563EB', linewidths=0)
        ax.set_title(f'Layer {li+1}\nPC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%',
                     fontsize=10)
        ax.set_xlabel('PC1', fontsize=9)
        ax.set_ylabel('PC2', fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, 'pca_per_layer.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # Also print the PC1 progression for easy reading
    print("  PC1 % variance per layer: " +
          "  →  ".join(f"L{i+1}: {v:.1f}%" for i, v in enumerate(pc1_var)))


# ── Plot 4: t-SNE by modality ─────────────────────────────────────────────────

def plot_tsne(H_pooled: np.ndarray, modalities: list, save_dir: str):
    from sklearn.manifold import TSNE

    print("  Running t-SNE (this may take ~1 min)...")
    tsne = TSNE(n_components=2, perplexity=40, n_iter=1000,
                random_state=SEED, n_jobs=-1)
    H_2d = tsne.fit_transform(H_pooled)

    fig, ax = plt.subplots(figsize=(9, 7))
    for mod, col in MODALITY_COLORS.items():
        idx = [i for i, m in enumerate(modalities) if m == mod]
        if idx:
            ax.scatter(H_2d[idx, 0], H_2d[idx, 1],
                       c=col, label=f'{mod} (n={len(idx)})',
                       s=12, alpha=0.55, linewidths=0)
    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    ax.set_title('t-SNE of mean-pooled H — coloured by therapy modality\n'
                 '(test patients only)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, markerscale=2)
    ax.grid(True, alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)
    path = os.path.join(save_dir, 'tsne_by_modality.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 5: Event-triggered average of H_t norm ───────────────────────────────

def plot_H_norm_vs_drivers(encoder, windows: np.ndarray, save_dir: str):
    """
    Event-triggered average of H_t norm around bolus and carbs events.

    For every bolus/carbs timestep across all windows, extract H norm in a
    [-PRE, +POST] window centred on the event, then average across all events.
    This collapses single-window noise and gives a clean answer to:
    'Does H norm systematically rise after a physiological event?'

    Layout: 2 rows × 3 cols
      Row 0: bolus-triggered  — H norm | PI | RA
      Row 1: carbs-triggered  — H norm | PI | RA
    """
    PRE  = 24   # 2h before event (steps)
    POST = 48   # 4h after event  (steps)
    WIN  = PRE + POST + 1
    t_axis = (np.arange(WIN) - PRE) * 5   # minutes relative to event

    print("  Computing H for all windows (event-triggered analysis)...")

    # Batch-compute H norm for all windows
    H_norms = []
    for i in range(0, len(windows), 64):
        batch = tf.cast(windows[i:i+64], tf.float32)
        H_batch = encoder(batch, training=False).numpy()   # (B, 288, 128)
        H_norms.append(np.linalg.norm(H_batch, axis=-1))  # (B, 288)
    H_norms = np.concatenate(H_norms, axis=0)             # (N, 288)

    def collect_triggered(event_feat_idx):
        snippets_H, snippets_pi, snippets_ra, snippets_cgm = [], [], [], []
        for wi in range(len(windows)):
            for et in np.where(windows[wi, :, event_feat_idx] > 0)[0]:
                lo, hi = et - PRE, et + POST + 1
                if lo < 0 or hi > WINDOW_LEN:
                    continue
                snippets_H.append(H_norms[wi, lo:hi])
                snippets_pi.append(windows[wi, lo:hi, PI_IDX])
                snippets_ra.append(windows[wi, lo:hi, RA_IDX])
                snippets_cgm.append(windows[wi, lo:hi, CGM_IDX])
        return (np.array(snippets_H), np.array(snippets_pi),
                np.array(snippets_ra), np.array(snippets_cgm))

    bolus_H, bolus_pi, _, bolus_cgm      = collect_triggered(BOLUS_IDX)
    carbs_H, _, carbs_ra, carbs_cgm      = collect_triggered(CARBS_IDX)
    print(f"  Bolus events: {len(bolus_H)}   Carbs events: {len(carbs_H)}")

    def mean_ci(arr):
        m   = arr.mean(axis=0)
        sem = arr.std(axis=0) / np.sqrt(len(arr))
        return m, m - 1.96 * sem, m + 1.96 * sem

    def plot_panel(ax, arr, color, label, ylabel):
        if len(arr) == 0:
            ax.text(0.5, 0.5, 'No events', transform=ax.transAxes, ha='center')
            return
        m, lo, hi = mean_ci(arr)
        ax.axvline(0, color='#111827', lw=1.2, ls='--', alpha=0.6, label='Event (t=0)')
        ax.axvspan(t_axis[0], 0, color='#F3F4F6', zorder=0)
        ax.fill_between(t_axis, lo, hi, color=color, alpha=0.2)
        ax.plot(t_axis, m, color=color, lw=2.0, label=f'{label} (n={len(arr)})')
        ax.set_xlabel('Time relative to event (min)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.15)
        ax.spines[['top', 'right']].set_visible(False)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        'Event-triggered average — H_t norm, PI/RA, CGM\n'
        'Shaded band = 95% CI  |  grey = pre-event  |  t=0 = event timestep',
        fontsize=12, fontweight='bold'
    )

    # Row 0: bolus-triggered — H norm | PI | CGM
    plot_panel(axes[0, 0], bolus_H,   COLORS['norm'], '||H_t||₂',    '||H_t||₂')
    plot_panel(axes[0, 1], bolus_pi,  COLORS['pi'],   'PI (z-score)', 'PI (z-score)')
    plot_panel(axes[0, 2], bolus_cgm, COLORS['cgm'],  'CGM (z-score)','CGM (z-score)')
    axes[0, 0].set_title('Bolus-triggered — H norm', fontsize=10)
    axes[0, 1].set_title('Bolus-triggered — PI',      fontsize=10)
    axes[0, 2].set_title('Bolus-triggered — CGM',     fontsize=10)

    # Row 1: carbs-triggered — H norm | RA | CGM
    plot_panel(axes[1, 0], carbs_H,   COLORS['norm'], '||H_t||₂',    '||H_t||₂')
    plot_panel(axes[1, 1], carbs_ra,  COLORS['ra'],   'RA (z-score)', 'RA (z-score)')
    plot_panel(axes[1, 2], carbs_cgm, COLORS['cgm'],  'CGM (z-score)','CGM (z-score)')
    axes[1, 0].set_title('Carbs-triggered — H norm', fontsize=10)
    axes[1, 1].set_title('Carbs-triggered — RA',      fontsize=10)
    axes[1, 2].set_title('Carbs-triggered — CGM',     fontsize=10)

    plt.tight_layout()
    path = os.path.join(save_dir, 'H_norm_vs_drivers.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 6: Deep attention analysis ──────────────────────────────────────────

def plot_attention_deep(encoder, windows: np.ndarray, n_layers: int,
                        n_heads: int, save_dir: str, n_avg: int = 100):
    """
    Three panels:
      Row 0: Average attention matrix (last layer, mean over heads, N windows)
      Row 1: Per-head attention heatmaps (4 heads, single window)
      Row 2: Attention entropy per timestep (average over N windows)

    Attention entropy H(q_i) = -sum_j a_ij * log(a_ij) measures how focused
    each query timestep is. Low entropy = attends to few keys (specific).
    High entropy = diffuse attention (integrates broadly).
    """
    print(f"  Computing average attention over {n_avg} windows...")

    # Build intermediate model up to the input of the last MHSA layer
    pre_model = keras.Model(
        encoder.input,
        encoder.get_layer(f'norm2_{n_layers - 2}').output
    )
    last_mhsa = encoder.get_layer(f'mhsa_{n_layers - 1}')

    # Accumulate attention across n_avg windows: shape (n_heads, 288, 288)
    attn_sum = None
    sample = windows[:n_avg]
    for i in range(len(sample)):
        x_in  = tf.cast(sample[i:i+1], tf.float32)
        x_pre = pre_model(x_in, training=False)
        _, attn = last_mhsa(x_pre, x_pre,
                            return_attention_scores=True, training=False)
        a = attn[0].numpy()   # (n_heads, 288, 288)
        attn_sum = a if attn_sum is None else attn_sum + a

    attn_avg  = attn_sum / len(sample)              # (n_heads, 288, 288)
    attn_mean = attn_avg.mean(axis=0)               # (288, 288) mean over heads

    # Subsampled versions for display
    sstp  = max(1, WINDOW_LEN // 72)
    sub   = lambda m: m[::sstp, ::sstp]
    n_sub = sub(attn_mean).shape[0]
    tick_every = max(1, n_sub // 12)
    tick_pos   = list(range(0, n_sub, tick_every))
    tick_lbl   = [f'{(t * sstp) * 5 // 60}h' for t in tick_pos]

    def heatmap(ax, mat, title):
        im = ax.imshow(sub(mat), aspect='auto', cmap='Blues',
                       origin='upper', interpolation='nearest')
        plt.colorbar(im, ax=ax, shrink=0.6, label='Attn weight')
        ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lbl, fontsize=7)
        ax.set_yticks(tick_pos); ax.set_yticklabels(tick_lbl, fontsize=7)
        ax.set_xlabel('Key (attended to)', fontsize=8)
        ax.set_ylabel('Query (attending)', fontsize=8)
        ax.set_title(title, fontsize=9)

    # Entropy: -sum_j a_ij * log(a_ij+eps), averaged over windows
    eps     = 1e-10
    entropy = -(attn_mean * np.log(attn_mean + eps)).sum(axis=-1)   # (288,)
    t_axis  = np.arange(WINDOW_LEN) * 5 / 60   # hours

    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(
        f'Deep Attention Analysis — last encoder layer  '
        f'({n_heads} heads, averaged over {len(sample)} windows)',
        fontsize=13, fontweight='bold'
    )
    gs = fig.add_gridspec(3, n_heads, hspace=0.45, wspace=0.35)

    # Row 0: average attention (spans all columns)
    ax_avg = fig.add_subplot(gs[0, :])
    heatmap(ax_avg, attn_mean,
            f'Average attention matrix — {len(sample)} windows, mean over {n_heads} heads\n'
            'Diagonal = local attention  |  Off-diagonal = long-range dependencies')

    # Row 1: per-head attention (single window — most dynamic)
    cgm_std = windows[:, :, CGM_IDX].std(axis=1)
    win_idx = int(np.argmax(cgm_std))
    x_in    = tf.cast(windows[win_idx:win_idx+1], tf.float32)
    x_pre   = pre_model(x_in, training=False)
    _, attn_single = last_mhsa(x_pre, x_pre,
                                return_attention_scores=True, training=False)
    attn_single = attn_single[0].numpy()   # (n_heads, 288, 288)

    for h in range(n_heads):
        ax = fig.add_subplot(gs[1, h])
        heatmap(ax, attn_single[h], f'Head {h+1}')

    # Row 2: attention entropy (spans all columns)
    ax_ent = fig.add_subplot(gs[2, :])
    ax_ent.plot(t_axis, entropy, color='#2563EB', lw=1.5)
    ax_ent.fill_between(t_axis, entropy.min(), entropy, alpha=0.15, color='#2563EB')
    ax_ent.set_xlabel('Timestep (hours)', fontsize=9)
    ax_ent.set_ylabel('Attention entropy H(q)', fontsize=9)
    ax_ent.set_title(
        'Attention entropy per query timestep\n'
        'Low = focused (attends to few keys)  |  High = diffuse (integrates broadly)',
        fontsize=10
    )
    ax_ent.set_xticks(range(0, 25, 2))
    ax_ent.grid(True, alpha=0.2)
    ax_ent.spines[['top', 'right']].set_visible(False)

    path = os.path.join(save_dir, 'attention_deep.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 7: H norm circadian pattern ─────────────────────────────────────────

def plot_H_circadian(H_norms: np.ndarray, windows: np.ndarray, save_dir: str):
    """
    Average H_t norm by hour of day across all windows.
    Hour is recovered from hour_sin (feature 3) and hour_cos (feature 4):
        hour = arctan2(sin, cos) * 24 / 2π  (mod 24)

    Reveals whether H norm has a circadian structure — e.g. peaks around
    meal times (breakfast, lunch, dinner) or during overnight basal periods.
    """
    hour_sin = windows[:, :, 3]   # (N, 288)
    hour_cos = windows[:, :, 4]   # (N, 288)
    hour     = (np.arctan2(hour_sin, hour_cos) * 24 / (2 * np.pi)) % 24  # (N, 288)

    hours_flat  = hour.flatten()               # (N*288,)
    H_norm_flat = H_norms.flatten()            # (N*288,)

    # Bin by integer hour (0–23), compute mean and SEM
    bin_means = []
    bin_sems  = []
    bin_ns    = []
    for h in range(24):
        mask  = (np.floor(hours_flat).astype(int) % 24) == h
        vals  = H_norm_flat[mask]
        bin_means.append(vals.mean() if len(vals) else np.nan)
        bin_sems.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        bin_ns.append(len(vals))

    bin_means = np.array(bin_means)
    bin_sems  = np.array(bin_sems)
    hrs = np.arange(24)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(hrs, bin_means - 1.96 * bin_sems,
                         bin_means + 1.96 * bin_sems,
                    alpha=0.2, color='#2563EB')
    ax.plot(hrs, bin_means, color='#2563EB', lw=2.0, marker='o', ms=4)
    ax.set_xlabel('Hour of day', fontsize=11)
    ax.set_ylabel('Mean ||H_t||₂', fontsize=11)
    ax.set_title(
        'Circadian pattern of H_t norm\n'
        'Mean ± 95% CI across all test windows  |  hour recovered from hour_sin/cos features',
        fontsize=12, fontweight='bold'
    )
    ax.set_xticks(hrs)
    ax.set_xticklabels([f'{h:02d}:00' for h in hrs], rotation=45, fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)

    # Annotate typical meal times
    for meal_h, label in [(7, 'Breakfast'), (13, 'Lunch'), (20, 'Dinner')]:
        ax.axvline(meal_h, color='#D97706', lw=1.0, ls='--', alpha=0.5)
        ax.text(meal_h + 0.1, ax.get_ylim()[1] * 0.99, label,
                fontsize=7, color='#D97706', va='top')

    plt.tight_layout()
    path = os.path.join(save_dir, 'H_norm_circadian.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 8: H norm vs feature correlation distribution ────────────────────────

def plot_H_feature_correlation(H_norms: np.ndarray, windows: np.ndarray,
                                save_dir: str):
    """
    For each window, compute the Pearson r between H_t norm (288,) and each
    input feature (288,). Plots the distribution of r across all windows as
    violin plots.

    Gives a quantitative answer to: which input features drive H norm most?
    Positive r → H norm rises when feature rises.
    Negative r → H norm falls when feature rises.
    Narrow distribution → consistent relationship across patients/windows.
    Wide distribution → relationship is context-dependent.
    """
    feature_names  = ['CGM', 'PI', 'RA', 'hour_sin', 'hour_cos',
                      'bolus', 'carbs']
    feature_idx    = [CGM_IDX, PI_IDX, RA_IDX, 3, 4, BOLUS_IDX, CARBS_IDX]
    feature_colors = [COLORS['cgm'], COLORS['pi'], COLORS['ra'],
                      '#6B7280', '#9CA3AF', COLORS['bolus'], COLORS['carbs']]

    correlations = {name: [] for name in feature_names}

    for i in range(len(windows)):
        h_norm = H_norms[i]              # (288,)
        if h_norm.std() < 1e-8:
            continue
        for name, idx in zip(feature_names, feature_idx):
            feat = windows[i, :, idx]   # (288,)
            if feat.std() < 1e-8:
                continue   # skip windows where feature is flat (e.g. RA between meals)
            r = float(np.corrcoef(h_norm, feat)[0, 1])
            correlations[name].append(r)

    fig, ax = plt.subplots(figsize=(12, 6))

    positions = range(len(feature_names))
    parts = ax.violinplot(
        [correlations[n] for n in feature_names],
        positions=positions,
        showmedians=True, showextrema=True
    )
    for pc, col in zip(parts['bodies'], feature_colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    parts['cmedians'].set_color('#111827')
    parts['cbars'].set_color('#6B7280')
    parts['cmins'].set_color('#6B7280')
    parts['cmaxes'].set_color('#6B7280')

    ax.axhline(0, color='#111827', lw=1.0, ls='--', alpha=0.5)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(feature_names, fontsize=10)
    ax.set_ylabel("Pearson r  (H_t norm vs feature)", fontsize=11)
    ax.set_title(
        'Distribution of per-window Pearson r between H_t norm and each input feature\n'
        'Windows with flat feature excluded (e.g. RA = 0 between meals)  |  test set only',
        fontsize=12, fontweight='bold'
    )
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines[['top', 'right']].set_visible(False)

    # Print median r for reference
    print("  Median Pearson r (H norm vs feature):")
    for name in feature_names:
        vals = correlations[name]
        if vals:
            print(f"    {name:12s}: {np.median(vals):+.3f}  "
                  f"(IQR {np.percentile(vals,25):+.3f} – {np.percentile(vals,75):+.3f})")

    plt.tight_layout()
    path = os.path.join(save_dir, 'H_norm_feature_correlation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Per-layer comparison analyses ─────────────────────────────────────────────

def _forward_with_attention(encoder, batch, n_layers):
    """
    Manual forward pass through every Transformer layer.

    Returns
    -------
    attn_list : list[np.ndarray]  shape (B, n_heads, 288, 288) per layer
    H_list    : list[np.ndarray]  shape (B, 288, d_model) per layer
    """
    pe = get_positional_encoding(WINDOW_LEN, D_MODEL)          # (1, 288, 128)
    x  = tf.cast(encoder.get_layer('input_proj')(batch), tf.float32) + pe

    attn_list, H_list = [], []
    for li in range(n_layers):
        mhsa          = encoder.get_layer(f'mhsa_{li}')
        attn_out, scores = mhsa(x, x, return_attention_scores=True, training=False)
        attn_list.append(scores.numpy())                        # (B, n_heads, T, T)

        x = encoder.get_layer(f'norm1_{li}')(x + attn_out)
        ffn = encoder.get_layer(f'ffn1_{li}')(x)
        ffn = encoder.get_layer(f'ffn2_{li}')(ffn)
        x   = encoder.get_layer(f'norm2_{li}')(x + ffn)
        H_list.append(x.numpy())                               # (B, 288, 128)

    return attn_list, H_list


# ── Plot 9: Per-layer attention matrices ──────────────────────────────────────

def plot_layer_attention_matrices(encoder, windows, n_layers, save_dir, n_avg=100):
    """
    Average attention matrix per encoder layer (mean over heads and windows).
    1 × n_layers grid → layer_attention_matrices.png
    """
    attn_sums = [np.zeros((WINDOW_LEN, WINDOW_LEN)) for _ in range(n_layers)]
    n_seen    = 0
    BATCH     = 16

    rng = np.random.default_rng(SEED + 99)
    idx = rng.choice(len(windows), size=min(n_avg, len(windows)), replace=False)

    for bi in range(0, len(idx), BATCH):
        batch = tf.cast(windows[idx[bi:bi+BATCH]], tf.float32)
        attn_list, _ = _forward_with_attention(encoder, batch, n_layers)
        B = batch.shape[0]
        for li in range(n_layers):
            # mean over heads → (B, T, T), sum over batch → (T, T)
            attn_sums[li] += attn_list[li].mean(axis=1).sum(axis=0)
        n_seen += B

    step     = 4                              # subsample 288→72 for display
    tick_pos = np.arange(0, WINDOW_LEN // step, 18)   # every 6h
    tick_lab = [f'{int(p * step * 5 / 60)}h' for p in tick_pos]

    fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 4.8))
    for li, ax in enumerate(axes):
        mat  = (attn_sums[li] / n_seen)[::step, ::step]
        vmax = np.percentile(mat, 98)
        im   = ax.imshow(mat, aspect='auto', cmap='viridis',
                         vmin=0, vmax=vmax, origin='upper')
        ax.set_title(f'Layer {li + 1}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Key (h)', fontsize=9)
        if li == 0:
            ax.set_ylabel('Query (h)', fontsize=9)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, fontsize=7)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_lab if li == 0 else [], fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f'Average attention matrix per encoder layer  (n={n_seen} windows, '
        f'mean over {N_HEADS} heads)\ntest set only',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(save_dir, 'layer_attention_matrices.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── Plot 10: Per-layer event-triggered H norm ─────────────────────────────────

def plot_layer_event_triggered(encoder, windows, n_layers, save_dir):
    """
    Event-triggered H_t norm (bolus / carbs) at each encoder layer.
    2 × n_layers grid → layer_event_triggered.png
    """
    PRE  = 24
    POST = 48
    WIN  = PRE + POST + 1
    t    = (np.arange(WIN) - PRE) * 5   # minutes

    print("  Computing per-layer H norms...")
    BATCH = 32
    H_norms_layers = [[] for _ in range(n_layers)]

    for bi in range(0, len(windows), BATCH):
        batch = tf.cast(windows[bi:bi+BATCH], tf.float32)
        _, H_list = _forward_with_attention(encoder, batch, n_layers)
        for li in range(n_layers):
            H_norms_layers[li].append(np.linalg.norm(H_list[li], axis=-1))  # (B, 288)

    H_norms_layers = [np.concatenate(r, axis=0) for r in H_norms_layers]  # (N, 288) each

    def collect(event_feat_idx, H_norms):
        snippets = []
        for wi in range(len(windows)):
            for et in np.where(windows[wi, :, event_feat_idx] > 0)[0]:
                lo, hi = et - PRE, et + POST + 1
                if lo < 0 or hi > WINDOW_LEN:
                    continue
                snippets.append(H_norms[wi, lo:hi])
        return np.array(snippets) if snippets else np.empty((0, WIN))

    layer_colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_layers))
    row_cfg = [
        ('Bolus-triggered', BOLUS_IDX),
        ('Carbs-triggered', CARBS_IDX),
    ]

    fig, axes = plt.subplots(2, n_layers, figsize=(3.5 * n_layers, 7), sharex=True)

    for row, (row_label, feat_idx) in enumerate(row_cfg):
        for li in range(n_layers):
            ax    = axes[row, li]
            snips = collect(feat_idx, H_norms_layers[li])
            if len(snips) == 0:
                ax.set_visible(False)
                continue
            mu  = snips.mean(axis=0)
            sem = snips.std(axis=0) / np.sqrt(len(snips))
            col = layer_colors[li]

            ax.plot(t, mu, color=col, lw=2)
            ax.fill_between(t, mu - sem, mu + sem, color=col, alpha=0.25)
            ax.axvline(0, color='#6B7280', lw=1.0, ls='--')

            if row == 0:
                ax.set_title(f'Layer {li + 1}', fontsize=11, fontweight='bold')
            if li == 0:
                ax.set_ylabel(f'{row_label}\nH_t norm', fontsize=9)
            if row == 1:
                ax.set_xlabel('Time (min)', fontsize=9)
            ax.text(0.97, 0.97, f'n={len(snips)}',
                    transform=ax.transAxes, fontsize=7,
                    ha='right', va='top', color='#6B7280')
            ax.grid(True, alpha=0.2)
            ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(
        'Event-triggered H_t norm per encoder layer\n'
        'Mean ± SEM across all events in test set',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(save_dir, 'layer_event_triggered.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── Plot 11: Linear reconstruction probe per layer ────────────────────────────

def plot_layer_reconstruction_probe(encoder, windows, n_layers, save_dir,
                                    n_probe=200):
    """
    Ordinary least-squares linear probe: H_t (128-dim) → CGM_t (z-score).
    Reports R² and MAE at each layer → layer_reconstruction_probe.png

    A jump in R² at the last layer signals head-pressure specialisation
    (the encoder is forced to encode CGM-reconstructable features at L5).
    """
    BATCH = 32
    rng   = np.random.default_rng(SEED + 7)
    idx   = rng.choice(len(windows), size=min(n_probe, len(windows)), replace=False)

    H_layers = [[] for _ in range(n_layers)]
    cgm_all  = []

    for bi in range(0, len(idx), BATCH):
        batch = tf.cast(windows[idx[bi:bi+BATCH]], tf.float32)
        _, H_list = _forward_with_attention(encoder, batch, n_layers)
        for li in range(n_layers):
            H_layers[li].append(H_list[li])   # (B, 288, 128)
        cgm_all.append(batch.numpy()[:, :, CGM_IDX])  # (B, 288)

    cgm_flat = np.concatenate(cgm_all, axis=0).reshape(-1)  # (M,)
    for li in range(n_layers):
        H_layers[li] = np.concatenate(H_layers[li], axis=0).reshape(-1, D_MODEL)

    r2_list, mae_list = [], []
    for li in range(n_layers):
        X  = H_layers[li]
        Xb = np.hstack([X, np.ones((len(X), 1), dtype=np.float32)])
        beta, _, _, _ = np.linalg.lstsq(Xb, cgm_flat, rcond=None)
        y_hat   = Xb @ beta
        ss_tot  = ((cgm_flat - cgm_flat.mean()) ** 2).sum()
        ss_res  = ((cgm_flat - y_hat) ** 2).sum()
        r2      = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        mae     = float(np.abs(cgm_flat - y_hat).mean())
        r2_list.append(r2)
        mae_list.append(mae)
        print(f"  Layer {li + 1}: R²={r2:.4f}  MAE={mae:.4f}")

    layer_labels = [f'L{i+1}' for i in range(n_layers)]
    x = np.arange(n_layers)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    bars = ax1.bar(x, r2_list, width=0.6, color='#2563EB', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_labels, fontsize=11)
    ax1.set_ylabel('R²', fontsize=11)
    ax1.set_title('Linear probe R²  (H_t → CGM_t)', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    for bar, val in zip(bars, r2_list):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.spines[['top', 'right']].set_visible(False)

    bars = ax2.bar(x, mae_list, width=0.6, color='#DC2626', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_labels, fontsize=11)
    ax2.set_ylabel('MAE (z-score)', fontsize=11)
    ax2.set_title('Linear probe MAE  (H_t → CGM_t)', fontsize=11, fontweight='bold')
    for bar, val in zip(bars, mae_list):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.spines[['top', 'right']].set_visible(False)

    N_used = min(n_probe, len(windows))
    fig.suptitle(
        f'Linear reconstruction probe per encoder layer\n'
        f'OLS  H_t (128-dim) → CGM_t (z-scored)  |  n={N_used} windows × 288 steps  |  test set only',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(save_dir, 'layer_reconstruction_probe.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# ── Plot 12: Per-layer feature correlation heatmap ────────────────────────────

def plot_layer_feature_correlation(encoder, windows, n_layers, save_dir,
                                   n_sample=300):
    """
    Heatmap: Pearson r between H_t norm and each input feature, per layer.
    Rows = encoder layers, columns = features → layer_feature_correlation.png

    Correlation is computed over ALL (window × timestep) pairs pooled together,
    not per-window. This handles sparse features like RA (near-zero for most of
    a window) that would produce NaN with a per-window approach.
    """
    feature_names = ['CGM', 'PI', 'RA', 'hour_sin', 'hour_cos', 'bolus', 'carbs']
    feature_idx   = [CGM_IDX, PI_IDX, RA_IDX, 3, 4, BOLUS_IDX, CARBS_IDX]
    BATCH         = 32

    rng = np.random.default_rng(SEED + 13)
    idx = rng.choice(len(windows), size=min(n_sample, len(windows)), replace=False)

    H_norms_layers = [[] for _ in range(n_layers)]
    for bi in range(0, len(idx), BATCH):
        batch = tf.cast(windows[idx[bi:bi+BATCH]], tf.float32)
        _, H_list = _forward_with_attention(encoder, batch, n_layers)
        for li in range(n_layers):
            H_norms_layers[li].append(np.linalg.norm(H_list[li], axis=-1))  # (B, 288)

    # Flatten to (N*288,) for pooled correlation
    H_norms_layers = [np.concatenate(r, axis=0).reshape(-1)
                      for r in H_norms_layers]
    sample_wins    = windows[idx]                                # (N, 288, 11)
    feat_flat      = {fidx: sample_wins[:, :, fidx].reshape(-1)
                      for fidx in feature_idx}                  # (N*288,) per feature

    pooled_r = np.zeros((n_layers, len(feature_names)))
    for li in range(n_layers):
        hn = H_norms_layers[li]
        for fi, fidx in enumerate(feature_idx):
            ft = feat_flat[fidx]
            if hn.std() < 1e-8 or ft.std() < 1e-8:
                pooled_r[li, fi] = float('nan')
            else:
                pooled_r[li, fi] = float(np.corrcoef(hn, ft)[0, 1])

    N_used = min(n_sample, len(windows))
    vmax = float(np.nanmax(np.abs(pooled_r)))
    vmax = max(0.05, vmax)

    fig, ax = plt.subplots(figsize=(len(feature_names) * 1.5 + 1, n_layers * 0.9 + 2))
    # Use masked array so NaN cells render as grey
    masked = np.ma.masked_invalid(pooled_r)
    cmap   = plt.cm.RdBu_r
    cmap.set_bad('#CCCCCC')
    im = ax.imshow(masked, aspect='auto', cmap=cmap,
                   vmin=-vmax, vmax=vmax, origin='upper')

    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, fontsize=10, rotation=30, ha='right')
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f'Layer {i + 1}' for i in range(n_layers)], fontsize=10)

    for li in range(n_layers):
        for fi in range(len(feature_names)):
            val = pooled_r[li, fi]
            if np.isnan(val):
                ax.text(fi, li, 'n/a', ha='center', va='center',
                        fontsize=8, color='#555555')
            else:
                color = 'white' if abs(val) > vmax * 0.6 else '#111827'
                ax.text(fi, li, f'{val:+.2f}', ha='center', va='center',
                        fontsize=9, color=color)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label='Pearson r')
    ax.set_title(
        f'Pooled Pearson r: H_t norm vs input features, per encoder layer\n'
        f'n={N_used} windows × 288 timesteps pooled  |  test set only',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(save_dir, 'layer_feature_correlation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')

    print('  Pooled Pearson r (layer × feature):')
    header = '         ' + ''.join(f'{n:>10}' for n in feature_names)
    print(header)
    for li in range(n_layers):
        row = f'  L{li + 1}:    ' + ''.join(
            f'{"nan":>10}' if np.isnan(pooled_r[li, fi]) else f'{pooled_r[li, fi]:+10.3f}'
            for fi in range(len(feature_names)))
        print(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    save_dir = os.path.join(RESULTS_BASE, args.run_id)
    weights_path = os.path.join(save_dir, 'encoder_weights.weights.h5')
    assert os.path.exists(weights_path), f"Weights not found: {weights_path}"

    print(f"\n── analyse_H.py  run={args.run_id} ──────────────────────────────")

    # 1. Build encoder and load weights
    print("\n[1/15] Building encoder and loading weights...")
    encoder = build_transformer_encoder(
        WINDOW_LEN, N_FEATURES, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT
    )
    # Dummy forward pass to initialise weights before loading
    encoder(tf.zeros((1, WINDOW_LEN, N_FEATURES)), training=False)
    encoder.load_weights(weights_path)
    print(f"  Loaded: {weights_path}")

    # 2. Reconstruct test split
    print("\n[2/15] Reconstructing test split...")
    test_records = index_test_patients(args.data)

    # 3. Sample windows with metadata
    print(f"\n[3/15] Sampling {args.n_windows} windows from test patients...")
    windows, modalities, ages = sample_windows(test_records, args.n_windows)

    # 4. Compute mean-pooled H for all windows
    print("\n[4/15] Computing mean-pooled H (final layer)...")
    H_pooled = get_H_pooled(encoder, windows)
    print(f"  H_pooled shape: {H_pooled.shape}")

    # 5. PCA by modality and age
    print("\n[5/15] Plotting PCA by modality and age...")
    plot_pca_metadata(H_pooled, modalities, ages, save_dir)

    # 6. Per-layer PCA
    print("\n[6/15] Computing H per layer (PCA)...")
    H_per_layer = get_H_pooled_per_layer(encoder, windows, N_LAYERS)
    plot_pca_per_layer(H_per_layer, save_dir)

    # 7. t-SNE
    print("\n[7/15] t-SNE...")
    plot_tsne(H_pooled, modalities, save_dir)

    # 8. H norm vs drivers (event-triggered)
    print("\n[8/15] H_t norm vs drivers (event-triggered average)...")
    plot_H_norm_vs_drivers(encoder, windows, save_dir)

    # 9. Deep attention analysis
    print("\n[9/15] Deep attention analysis (average + per-head + entropy)...")
    plot_attention_deep(encoder, windows, N_LAYERS, N_HEADS, save_dir)

    # 10. H norm circadian pattern
    print("\n[10/15] H norm circadian pattern...")
    H_norms = []
    for i in range(0, len(windows), 64):
        batch = tf.cast(windows[i:i+64], tf.float32)
        H_b   = encoder(batch, training=False).numpy()
        H_norms.append(np.linalg.norm(H_b, axis=-1))
    H_norms = np.concatenate(H_norms, axis=0)   # (N, 288)
    plot_H_circadian(H_norms, windows, save_dir)

    # 11. H norm vs feature correlation distribution
    print("\n[11/15] H norm vs feature correlation distributions...")
    plot_H_feature_correlation(H_norms, windows, save_dir)

    # 12. Per-layer attention matrices
    print("\n[12/15] Per-layer attention matrices...")
    plot_layer_attention_matrices(encoder, windows, N_LAYERS, save_dir, n_avg=100)

    # 13. Per-layer event-triggered H norm
    print("\n[13/15] Per-layer event-triggered H norm...")
    plot_layer_event_triggered(encoder, windows, N_LAYERS, save_dir)

    # 14. Linear reconstruction probe per layer
    print("\n[14/15] Linear reconstruction probe per layer...")
    plot_layer_reconstruction_probe(encoder, windows, N_LAYERS, save_dir)

    # 15. Per-layer feature correlation heatmap
    print("\n[15/15] Per-layer feature correlation heatmap...")
    plot_layer_feature_correlation(encoder, windows, N_LAYERS, save_dir)

    print(f"\n  All plots saved to: {save_dir}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep H analysis for MTSM encoder')
    parser.add_argument('--data',      type=str, default='data/processed/adults',
                        help='Path to processed .npz directory (same as training)')
    parser.add_argument('--run_id',    type=str, default='run12',
                        help='Run ID — loads weights from results/mtsm/{run_id}/')
    parser.add_argument('--n_windows', type=int, default=1500,
                        help='Total windows to sample across test patients')
    args = parser.parse_args()
    main(args)
