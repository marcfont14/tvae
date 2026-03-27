"""
analyse_H.py
============
H representation analysis for the Stage 1 MTSM Transformer encoder.

Loads encoder weights from a trained run, reconstructs the exact test split
(same SEED + patient-level split as experiment_mtsm.py), and generates 5 figures:

  1.  transformer_H_analysis.png  — attention heatmap + H_t norm vs CGM + multi-window PCA
  2.  H_norm_vs_drivers.png       — event-triggered H_t norm around bolus/carbs (2x3 grid)
  3.  H_norm_circadian.png        — average H_t norm by hour of day (circadian pattern)
  4.  abstraction_trajectory.png  — r(H_norm, CGM) and r(H_norm, PI) across layers L1 to L5
  5.  L1_vs_L5_scatter.png        — linear probe predicted vs actual CGM at L1 and L5

Usage:
  python scripts/analyse_H.py \\
      --data     data/processed/adults \\
      --run_id   run14 \\
      --no_age                     # must match the flag used during training
      --n_windows 1500             # total windows to sample across test patients
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


# ── Plot 1: Transformer H analysis (3-panel) ──────────────────────────────────

def plot_transformer_H_analysis(encoder, windows: np.ndarray,
                                 n_layers: int, n_heads: int,
                                 save_dir: str, n_attn_avg: int = 100):
    """
    3-panel figure:
      (0) Average attention heatmap — last layer, averaged over n_attn_avg windows
      (1) H_t norm vs CGM for the most variable single window
      (2) Multi-window PCA — each H_t from ~20 sampled windows, coloured by CGM

    Panel (0): Shows global attention structure (diagonal=local, off-diagonal=long-range).
    Panel (1): Shows how H norm tracks metabolic dynamics within one window.
    Panel (2): PCA across timesteps from many windows reveals that H organises by glucose level.
    """
    from sklearn.decomposition import PCA

    print(f"  Computing attention (avg over {n_attn_avg} windows) + H norm + PCA...")

    # Build pre-last-layer model for attention extraction
    pre_model = keras.Model(
        encoder.input,
        encoder.get_layer(f'norm2_{n_layers - 2}').output
    )
    last_mhsa = encoder.get_layer(f'mhsa_{n_layers - 1}')

    # Accumulate attention
    attn_sum   = None
    n_attn_avg = min(n_attn_avg, len(windows))
    for i in range(n_attn_avg):
        x_in  = tf.cast(windows[i:i+1], tf.float32)
        x_pre = pre_model(x_in, training=False)
        _, attn = last_mhsa(x_pre, x_pre,
                            return_attention_scores=True, training=False)
        a = attn[0].numpy()   # (n_heads, 288, 288)
        attn_sum = a if attn_sum is None else attn_sum + a
    attn_avg  = attn_sum / n_attn_avg
    attn_mean = attn_avg.mean(axis=0)   # (288, 288)

    # Most variable window for H norm panel
    cgm_std  = windows[:, :, CGM_IDX].std(axis=1)
    win_idx  = int(np.argmax(cgm_std))
    x_in     = tf.cast(windows[win_idx:win_idx+1], tf.float32)
    H_win    = encoder(x_in, training=False).numpy()[0]   # (288, d_model)
    H_norm   = np.linalg.norm(H_win, axis=1)              # (288,)
    cgm_win  = windows[win_idx, :, CGM_IDX]

    # Multi-window PCA: sample H_t vectors from up to 20 windows
    N_PCA_WIN = min(20, len(windows))
    H_flat    = []
    cgm_flat  = []
    rng_idx   = np.random.choice(len(windows), N_PCA_WIN, replace=False)
    for wi in rng_idx:
        x_in  = tf.cast(windows[wi:wi+1], tf.float32)
        H_w   = encoder(x_in, training=False).numpy()[0]   # (288, 128)
        H_flat.append(H_w)
        cgm_flat.append(windows[wi, :, CGM_IDX])
    H_flat   = np.concatenate(H_flat,   axis=0)   # (N_PCA_WIN*288, 128)
    cgm_flat = np.concatenate(cgm_flat, axis=0)   # (N_PCA_WIN*288,)

    pca  = PCA(n_components=2, random_state=SEED)
    H_2d = pca.fit_transform(H_flat)
    var  = pca.explained_variance_ratio_

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
    fig.suptitle(
        f'H Representation Analysis — Stage 1 Encoder  '
        f'({n_layers} layers, {n_heads} heads, d={D_MODEL})',
        fontsize=13, fontweight='bold'
    )

    # Panel (0): Attention heatmap
    ax0  = fig.add_subplot(gs[0])
    sstp = max(1, WINDOW_LEN // 64)
    asub = attn_mean[::sstp, ::sstp]
    n_sub = asub.shape[0]
    im = ax0.imshow(asub, aspect='auto', cmap='Blues', origin='upper',
                    interpolation='nearest')
    plt.colorbar(im, ax=ax0, shrink=0.8, label='Attention weight')
    tick_every = max(1, n_sub // 8)
    tick_pos   = list(range(0, n_sub, tick_every))
    tick_lbl   = [f'{(t*sstp)*5//60}h' for t in tick_pos]
    ax0.set_xticks(tick_pos); ax0.set_xticklabels(tick_lbl, fontsize=8)
    ax0.set_yticks(tick_pos); ax0.set_yticklabels(tick_lbl, fontsize=8)
    ax0.set_xlabel('Key (attended to)', fontsize=9)
    ax0.set_ylabel('Query (attending)', fontsize=9)
    ax0.set_title(
        f'Attention weights — last layer\nMean over {n_heads} heads, {n_attn_avg} windows',
        fontsize=10
    )

    # Panel (1): H norm vs CGM
    ax1      = fig.add_subplot(gs[1])
    ax1_twin = ax1.twinx()
    t        = np.arange(WINDOW_LEN)
    ax1.plot(t, H_norm, color=COLORS['norm'], lw=1.5, alpha=0.9, label='||H_t||₂')
    ax1_twin.plot(t, cgm_win, color=COLORS['cgm'], lw=1.2, alpha=0.6, label='CGM (z-score)')
    ax1.set_xlabel('Time (5 min steps)', fontsize=9)
    ax1.set_ylabel('||H_t||₂', fontsize=9, color=COLORS['norm'])
    ax1_twin.set_ylabel('CGM (z-score)', fontsize=9, color=COLORS['cgm'])
    ax1.set_title('H_t norm vs CGM — most variable window', fontsize=10)
    lines = ax1.get_lines() + ax1_twin.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=9)
    ax1.grid(True, alpha=0.2)
    ax1.spines[['top']].set_visible(False)
    step_t = 48
    ax1.set_xticks(range(0, WINDOW_LEN + 1, step_t))
    ax1.set_xticklabels([f'{t*5//60}h' for t in range(0, WINDOW_LEN + 1, step_t)],
                        fontsize=8)

    # Panel (2): Multi-window PCA
    ax2 = fig.add_subplot(gs[2])
    sc  = ax2.scatter(H_2d[:, 0], H_2d[:, 1], c=cgm_flat,
                      cmap='RdYlGn_r', s=4, alpha=0.5)
    plt.colorbar(sc, ax=ax2, label='CGM (z-score)')
    ax2.set_xlabel(f'PC1 ({var[0]*100:.1f}% var)', fontsize=9)
    ax2.set_ylabel(f'PC2 ({var[1]*100:.1f}% var)', fontsize=9)
    ax2.set_title(
        f'H_t in PCA space — {N_PCA_WIN} windows\n'
        'Each point = one timestep, colour = CGM',
        fontsize=10
    )
    ax2.grid(True, alpha=0.2)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.savefig(os.path.join(save_dir, 'transformer_H_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(save_dir, 'transformer_H_analysis.png')}")


# ── Plot 2: Event-triggered average of H_t norm ───────────────────────────────

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


# ── Plot 4: Linear probe scatter — L1 vs L5 ──────────────────────────────────

def plot_L1_vs_L5_scatter(encoder, windows: np.ndarray, n_layers: int,
                           save_dir: str, n_sample: int = 3000):
    """
    Scatter predicted vs actual CGM at layer 1 and layer 5 side by side.

    For each layer, we fit a linear OLS probe: H_t → CGM_t using n_sample
    randomly drawn (window, timestep) pairs. Then we plot predicted vs actual.

    Interpretation:
    - L1: tight positive diagonal → H encodes raw CGM value linearly.
    - L5: inverted / scattered diagonal → H has sign-flipped and become abstract.
    The gap between the two panels quantifies how much the encoder transforms
    the representation beyond a simple linear copy of the input.
    """
    from sklearn.linear_model import Ridge

    print("  Fitting linear probes at L1 and L5...")

    layer_models = {
        1: keras.Model(encoder.input, encoder.get_layer('norm2_0').output),
        n_layers: encoder,   # full encoder = layer 5
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Linear Probe: Predicted vs Actual CGM — L1 vs L5\n'
                 'OLS fitted on H_t → CGM_t across sampled (window, timestep) pairs',
                 fontsize=12, fontweight='bold')

    for ax, (layer_num, lm) in zip(axes, layer_models.items()):
        # Collect (H_t, CGM_t) pairs
        H_all   = []
        cgm_all = []
        for i in range(0, len(windows), 64):
            batch = tf.cast(windows[i:i+64], tf.float32)
            H_b   = lm(batch, training=False).numpy()   # (B, 288, d_model)
            H_all.append(H_b.reshape(-1, H_b.shape[-1]))
            cgm_all.append(windows[i:i+64, :, CGM_IDX].reshape(-1))

        H_all   = np.concatenate(H_all,   axis=0)   # (N*288, d_model)
        cgm_all = np.concatenate(cgm_all, axis=0)   # (N*288,)

        # Random subsample
        idx  = np.random.choice(len(H_all), min(n_sample, len(H_all)), replace=False)
        H_s  = H_all[idx]
        y_s  = cgm_all[idx]

        # Fit OLS and predict
        probe = Ridge(alpha=1.0)
        probe.fit(H_s, y_s)
        y_hat = probe.predict(H_s)
        r2    = 1 - ((y_s - y_hat) ** 2).sum() / ((y_s - y_s.mean()) ** 2).sum()

        # Scatter (subsample further for display)
        n_disp = min(2000, len(y_s))
        disp   = np.random.choice(len(y_s), n_disp, replace=False)
        ax.scatter(y_s[disp], y_hat[disp], s=3, alpha=0.3, color='#2563EB',
                   linewidths=0)
        lo = min(y_s.min(), y_hat.min())
        hi = max(y_s.max(), y_hat.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5, label='y = x')
        ax.set_xlabel('Actual CGM (z-score)', fontsize=10)
        ax.set_ylabel('Predicted CGM (z-score)', fontsize=10)
        layer_label = 'L1 (first layer)' if layer_num == 1 else f'L{n_layers} (last layer)'
        ax.set_title(f'{layer_label}\nLinear probe  R² = {r2:.3f}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_aspect('equal', adjustable='box')
        print(f"  Layer {layer_num}: R² = {r2:.3f}")

    plt.tight_layout()
    path = os.path.join(save_dir, 'L1_vs_L5_scatter.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 5: Abstraction trajectory — r(CGM) and r(PI) per layer ───────────────

def plot_abstraction_trajectory(encoder, windows: np.ndarray, n_layers: int,
                                 save_dir: str):
    """
    Line plot showing Pearson r between H_t norm and CGM/PI across encoder layers.

    For each layer L=1..5:
      - Extract H_t for all (window, timestep) pairs at that layer
      - Compute H_t norm (scalar per timestep)
      - Compute Pearson r(H_norm, CGM) and r(H_norm, PI) across all pairs

    The resulting trajectory (two lines from L1 to L5) reveals:
    - Early layers: H norm is positively correlated with CGM (raw encoding)
    - Late layers:  CGM correlation inverts (sign flip) while PI correlation changes
      This is the key finding — the representation progressively transforms from
      a raw glucose tracker to an abstract metabolic complexity signal.
    """
    from scipy.stats import pearsonr

    print("  Computing H_t norm vs CGM/PI Pearson r per layer...")

    r_cgm_per_layer = []
    r_pi_per_layer  = []

    for li in range(n_layers):
        lm = keras.Model(encoder.input, encoder.get_layer(f'norm2_{li}').output)

        H_norms_all = []
        cgm_all     = []
        pi_all      = []

        for i in range(0, len(windows), 64):
            batch = tf.cast(windows[i:i+64], tf.float32)
            H_b   = lm(batch, training=False).numpy()   # (B, 288, d_model)
            H_norm_b = np.linalg.norm(H_b, axis=-1)    # (B, 288)
            H_norms_all.append(H_norm_b.reshape(-1))
            cgm_all.append(windows[i:i+64, :, CGM_IDX].reshape(-1))
            pi_all.append(windows[i:i+64, :, PI_IDX].reshape(-1))

        H_norms_flat = np.concatenate(H_norms_all)
        cgm_flat     = np.concatenate(cgm_all)
        pi_flat      = np.concatenate(pi_all)

        r_cgm, _ = pearsonr(H_norms_flat, cgm_flat)
        r_pi,  _ = pearsonr(H_norms_flat, pi_flat)
        r_cgm_per_layer.append(r_cgm)
        r_pi_per_layer.append(r_pi)
        print(f"  L{li+1}: r(CGM)={r_cgm:+.3f}  r(PI)={r_pi:+.3f}")

    layers = list(range(1, n_layers + 1))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhline(0, color='#9CA3AF', lw=1, ls='-', alpha=0.5)
    ax.plot(layers, r_cgm_per_layer, color=COLORS['cgm'], lw=2.5,
            marker='o', ms=7, label='r(||H_t||, CGM)')
    ax.plot(layers, r_pi_per_layer,  color=COLORS['pi'],  lw=2.5,
            marker='s', ms=7, label='r(||H_t||, PI)')
    for li, (rc, rp) in enumerate(zip(r_cgm_per_layer, r_pi_per_layer)):
        ax.annotate(f'{rc:+.2f}', (li+1, rc),
                    textcoords='offset points', xytext=(6, 4), fontsize=8,
                    color=COLORS['cgm'])
        ax.annotate(f'{rp:+.2f}', (li+1, rp),
                    textcoords='offset points', xytext=(6, -10), fontsize=8,
                    color=COLORS['pi'])

    ax.set_xlabel('Encoder Layer', fontsize=11)
    ax.set_ylabel('Pearson r  (||H_t||₂ vs feature)', fontsize=11)
    ax.set_title(
        'Abstraction Trajectory — H_t norm vs CGM / PI across layers\n'
        'Sign flip from L1→L5 indicates the encoder transforms raw glucose into '
        'an abstract metabolic complexity signal',
        fontsize=11, fontweight='bold'
    )
    ax.set_xticks(layers)
    ax.set_xticklabels([f'L{l}' for l in layers], fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(-1.05, 1.05)

    plt.tight_layout()
    path = os.path.join(save_dir, 'abstraction_trajectory.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    save_dir     = os.path.join(RESULTS_BASE, args.run_id)
    weights_path = os.path.join(save_dir, 'encoder_weights.weights.h5')
    assert os.path.exists(weights_path), f"Weights not found: {weights_path}"

    print(f"\n── analyse_H.py  run={args.run_id} ──────────────────────────────")

    # 1. Build encoder and load weights
    print("\n[1/5] Building encoder and loading weights...")
    n_features_model = 10 if getattr(args, 'no_age', False) else N_FEATURES
    encoder = build_transformer_encoder(
        WINDOW_LEN, n_features_model, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT
    )
    encoder(tf.zeros((1, WINDOW_LEN, n_features_model)), training=False)
    encoder.load_weights(weights_path)
    print(f"  Loaded: {weights_path}")

    # 2. Reconstruct test split and sample windows
    print("\n[2/5] Reconstructing test split and sampling windows...")
    test_records          = index_test_patients(args.data)
    windows, modalities, ages = sample_windows(test_records, args.n_windows)

    if getattr(args, 'no_age', False):
        windows = windows[:, :, :10]

    print(f"  Windows shape: {windows.shape}")

    # 3. Transformer H analysis (attention + H norm + multi-window PCA)
    print("\n[3/5] Transformer H analysis...")
    plot_transformer_H_analysis(encoder, windows, N_LAYERS, N_HEADS, save_dir)

    # 4. Event-triggered H norm vs PI/RA/CGM
    print("\n[4/5] Event-triggered H norm vs drivers...")
    plot_H_norm_vs_drivers(encoder, windows, save_dir)

    # 5a. H norm circadian pattern
    print("\n[5/5] H norm circadian pattern + abstraction trajectory + L1 vs L5 probe...")
    H_norms_list = []
    for i in range(0, len(windows), 64):
        batch = tf.cast(windows[i:i+64], tf.float32)
        H_b   = encoder(batch, training=False).numpy()
        H_norms_list.append(np.linalg.norm(H_b, axis=-1))
    H_norms = np.concatenate(H_norms_list, axis=0)   # (N, 288)
    plot_H_circadian(H_norms, windows, save_dir)

    # 5b. Abstraction trajectory
    plot_abstraction_trajectory(encoder, windows, N_LAYERS, save_dir)

    # 5c. L1 vs L5 linear probe scatter
    plot_L1_vs_L5_scatter(encoder, windows, N_LAYERS, save_dir)

    print(f"\n  All plots saved to: {save_dir}/")
    print(f"  Outputs: transformer_H_analysis.png | H_norm_vs_drivers.png | "
          f"H_norm_circadian.png | abstraction_trajectory.png | L1_vs_L5_scatter.png\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep H analysis for MTSM encoder')
    parser.add_argument('--data',      type=str, default='data/processed/adults',
                        help='Path to processed .npz directory (same as training)')
    parser.add_argument('--run_id',    type=str, default='run14',
                        help='Run ID — loads weights from results/mtsm/{run_id}/')
    parser.add_argument('--n_windows', type=int, default=1500,
                        help='Total windows to sample across test patients')
    parser.add_argument('--no_age',    action='store_true', default=False,
                        help='Drop age_norm (feature 10) from encoder input — '
                             'must match the flag used during training')
    args = parser.parse_args()
    main(args)
