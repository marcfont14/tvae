"""
attention_viz.py
================
Visualise per-window attention patterns for individual physiologically
selected windows from a trained MTSM encoder.

Unlike the averaged attention heatmaps in replot.py (which collapse all
windows and heads into one matrix), this script selects specific windows
by physiological criteria and plots their attention matrices individually.

Usage (inside container):
  python scripts/attention_viz.py --run_id run21 --no_age
  python scripts/attention_viz.py --run_id run21 --no_age --layer 5
  python scripts/attention_viz.py --run_id run21 --no_age --layer all
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ── Constants (must match experiment_mtsm.py) ─────────────────────────────────
WINDOW_LEN   = 288
N_FEATURES   = 11
CGM_IDX      = 0
PI_IDX       = 1
RA_IDX       = 2
BOLUS_IDX    = 5
CARBS_IDX    = 6
D_MODEL      = 128
N_HEADS      = 4
N_LAYERS     = 5
D_FF         = 256
DROPOUT      = 0.2
TEST_SPLIT   = 0.1
VAL_SPLIT    = 0.1
SEED         = 42
RESULTS_BASE = 'results/mtsm'

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── Positional encoding ────────────────────────────────────────────────────────
def get_positional_encoding(seq_len, d_model):
    positions = np.arange(seq_len)[:, np.newaxis]
    dims      = np.arange(d_model)[np.newaxis, :]
    angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)


# ── Encoder build + attention extraction ──────────────────────────────────────
def build_transformer_encoder(window_len, n_features, d_model, n_heads, n_layers, d_ff, dropout):
    """Functional API encoder — identical to experiment_mtsm.py."""
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


def extract_attention(encoder, x_batch, n_layers):
    """
    Run the encoder forward pass layer-by-layer, capturing attention weights
    from each MHA layer using return_attention_scores=True.

    Works on the loaded functional-API encoder by accessing layers by name.
    Returns (H, attn_weights_all) where attn_weights_all[i] has shape
    (batch, n_heads, seq_len, seq_len).
    """
    # Step through the encoder manually using named layers
    x = encoder.get_layer('input_proj')(x_batch, training=False)
    pe = get_positional_encoding(encoder.input_shape[1], x.shape[-1])
    x  = x + pe

    attn_weights_all = []
    for i in range(n_layers):
        mhsa  = encoder.get_layer(f'mhsa_{i}')
        norm1 = encoder.get_layer(f'norm1_{i}')
        ffn1  = encoder.get_layer(f'ffn1_{i}')
        ffn2  = encoder.get_layer(f'ffn2_{i}')
        norm2 = encoder.get_layer(f'norm2_{i}')

        attn_out, attn_w = mhsa(x, x, return_attention_scores=True, training=False)
        x = norm1(x + attn_out, training=False)
        x = norm2(x + ffn2(ffn1(x, training=False), training=False), training=False)
        attn_weights_all.append(attn_w)   # (batch, n_heads, seq, seq)

    return x, attn_weights_all


# ── Data loading ───────────────────────────────────────────────────────────────
def load_test_windows(processed_dir, n_sample=2000, no_age=True):
    npz_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('.npz')])
    rng = np.random.RandomState(SEED)
    rng.shuffle(npz_files)
    n_test = max(1, int(len(npz_files) * TEST_SPLIT))
    test_files = npz_files[:n_test]

    windows = []
    for fname in test_files:
        fpath = os.path.join(processed_dir, fname)
        try:
            data = np.load(fpath, allow_pickle=True)
            wins = data['windows'].astype(np.float32)
        except Exception:
            continue
        bolus = wins[:, :, BOLUS_IDX]
        carbs = wins[:, :, CARBS_IDX]
        cgm   = wins[:, :, CGM_IDX]
        keep  = (((bolus + carbs) > 0).any(axis=1) &
                 (cgm.std(axis=1) > 0.3) & (cgm.std(axis=1) < 4.0))
        windows.extend(wins[keep])

    windows = np.stack(windows, axis=0)
    if len(windows) > n_sample:
        idx = np.random.choice(len(windows), n_sample, replace=False)
        windows = windows[idx]
    if no_age:
        windows = windows[:, :, :10]
    print(f"  Loaded {len(windows)} test windows")
    return windows


# ── Window selection by physiological type ────────────────────────────────────
def select_windows(windows):
    """
    Select one representative window per physiological category.
    Returns dict: {label: window_array (288, n_features)}
    """
    cgm   = windows[:, :, CGM_IDX]        # z-scored
    bolus = windows[:, :, BOLUS_IDX]
    carbs = windows[:, :, CARBS_IDX]

    selected = {}

    # 1. Post-meal: largest carbs event, good CGM variability
    carbs_count = (carbs > 0).sum(axis=1)
    post_meal_idx = np.where(carbs_count >= 2)[0]
    if len(post_meal_idx):
        # Pick the one with highest CGM std among post-meal windows
        best = post_meal_idx[cgm[post_meal_idx].std(axis=1).argmax()]
        selected['Post-meal'] = windows[best]

    # 2. Post-bolus: large bolus activity, no carbs event
    bolus_count = (bolus > 0).sum(axis=1)
    post_bolus_idx = np.where((bolus_count >= 3) & (carbs_count == 0))[0]
    if len(post_bolus_idx):
        best = post_bolus_idx[cgm[post_bolus_idx].std(axis=1).argmax()]
        selected['Post-bolus'] = windows[best]

    # 3. Hypoglycaemia: CGM dips below -1.5 z-score (roughly < 63 mg/dL)
    hypo_idx = np.where((cgm.min(axis=1) < -1.5) & (carbs_count > 0))[0]
    if len(hypo_idx):
        # Pick window with deepest dip
        best = hypo_idx[cgm[hypo_idx].min(axis=1).argmin()]
        selected['Hypoglycaemia'] = windows[best]

    # 4. Stable basal: low CGM std, no events
    basal_idx = np.where(
        (cgm.std(axis=1) < 0.6) & (carbs_count == 0) & (bolus_count == 0)
    )[0]
    if len(basal_idx):
        best = basal_idx[np.abs(cgm[basal_idx].mean(axis=1)).argmin()]
        selected['Stable basal'] = windows[best]

    # 5. High variability: extreme CGM swings
    high_var_idx = np.where(cgm.std(axis=1) > 1.8)[0]
    if len(high_var_idx):
        best = high_var_idx[cgm[high_var_idx].std(axis=1).argmax()]
        selected['High variability'] = windows[best]

    print(f"  Selected window types: {list(selected.keys())}")
    return selected


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_individual_attention(encoder, selected_windows, layers_to_plot,
                               n_heads, results_dir):
    """
    Grid: rows = window types, columns = selected encoder layers.
    Each cell = attention matrix averaged over heads, with CGM overlay.
    """
    window_labels = list(selected_windows.keys())
    n_windows = len(window_labels)
    n_layers  = len(layers_to_plot)
    time_h    = np.linspace(0, 24, WINDOW_LEN)

    fig, axes = plt.subplots(
        n_windows, n_layers + 1,
        figsize=(4 * (n_layers + 1), 3.5 * n_windows),
        gridspec_kw={'width_ratios': [1.2] + [2] * n_layers}
    )
    if n_windows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f'Per-Window Attention — {os.path.basename(results_dir)}\n'
        f'Each cell = mean over {n_heads} heads | layers {layers_to_plot}',
        fontsize=13, fontweight='bold', y=1.01
    )

    for row_i, label in enumerate(window_labels):
        win = selected_windows[label]
        x   = tf.cast(win[np.newaxis], tf.float32)
        _, attn_weights_all = extract_attention(encoder, x, N_LAYERS)

        cgm_vals = win[:, CGM_IDX]

        # Column 0: CGM + driver signal for this window
        ax = axes[row_i, 0]
        ax.plot(time_h, cgm_vals, color='#111827', lw=1.5, label='CGM')
        bolus_times = time_h[win[:, BOLUS_IDX] > 0]
        carbs_times = time_h[win[:, CARBS_IDX] > 0]
        for bt in bolus_times:
            ax.axvline(bt, color='#DC2626', alpha=0.6, lw=0.8)
        for ct in carbs_times:
            ax.axvline(ct, color='#D97706', alpha=0.6, lw=0.8)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('CGM (z)')
        ax.set_xlim(0, 24)

        # Columns 1+: attention matrix per layer
        for col_i, layer_idx in enumerate(layers_to_plot):
            ax = axes[row_i, col_i + 1]
            # attn_weights_all[layer_idx]: (1, n_heads, seq, seq)
            attn = attn_weights_all[layer_idx][0].numpy()  # (n_heads, 288, 288)
            attn_mean = attn.mean(axis=0)                  # (288, 288)

            # Subsample for display (every 4 steps = 20-min resolution)
            step = 4
            attn_sub = attn_mean[::step, ::step]
            time_sub = time_h[::step]

            im = ax.imshow(
                attn_sub, aspect='auto', origin='lower',
                extent=[0, 24, 0, 24],
                cmap='Blues', vmin=0, vmax=np.percentile(attn_mean, 99)
            )
            ax.set_title(f'Layer {layer_idx + 1}', fontsize=9)
            ax.set_xlabel('Key (h)')
            if col_i == 0:
                ax.set_ylabel('Query (h)')
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    plt.tight_layout()
    save_path = os.path.join(results_dir, 'attention_individual_windows.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_head_attention(encoder, selected_windows, layer_idx,
                             n_heads, results_dir):
    """
    For the selected layer, plot each attention head separately for each
    window type. Reveals head specialisation (local, global, etc.).
    """
    window_labels = list(selected_windows.keys())
    n_windows = len(window_labels)

    fig, axes = plt.subplots(
        n_windows, n_heads,
        figsize=(4 * n_heads, 3.5 * n_windows)
    )
    if n_windows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f'Per-Head Attention — Layer {layer_idx + 1} — {os.path.basename(results_dir)}',
        fontsize=13, fontweight='bold'
    )

    for row_i, label in enumerate(window_labels):
        win = selected_windows[label]
        x   = tf.cast(win[np.newaxis], tf.float32)
        _, attn_weights_all = extract_attention(encoder, x, N_LAYERS)

        attn = attn_weights_all[layer_idx][0].numpy()  # (n_heads, 288, 288)

        for head_i in range(n_heads):
            ax = axes[row_i, head_i]
            attn_h = attn[head_i]
            step   = 4
            attn_sub = attn_h[::step, ::step]

            im = ax.imshow(
                attn_sub, aspect='auto', origin='lower',
                extent=[0, 24, 0, 24],
                cmap='Blues', vmin=0, vmax=np.percentile(attn_h, 99)
            )
            if row_i == 0:
                ax.set_title(f'Head {head_i + 1}', fontsize=10)
            if head_i == 0:
                ax.set_ylabel(f'{label}\nQuery (h)', fontsize=8)
            ax.set_xlabel('Key (h)', fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    plt.tight_layout()
    save_path = os.path.join(results_dir, f'attention_per_head_L{layer_idx+1}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    results_dir = os.path.join(RESULTS_BASE, args.run_id)
    enc_path    = os.path.join(results_dir, 'encoder_weights.weights.h5')
    assert os.path.exists(enc_path), f"Encoder weights not found: {enc_path}"

    n_features_model = 10 if args.no_age else N_FEATURES

    # Parse layers_to_plot
    if args.layer == 'all':
        layers_to_plot = list(range(N_LAYERS))
    else:
        layers_to_plot = [int(args.layer) - 1]   # 1-indexed CLI → 0-indexed

    print(f"\n── attention_viz.py  run={args.run_id} ──────────────────────────")

    # Build and load
    print("\n[1/4] Building encoder and loading weights...")
    encoder = build_transformer_encoder(
        WINDOW_LEN, n_features_model, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT
    )
    encoder(tf.zeros((1, WINDOW_LEN, n_features_model)), training=False)
    encoder.load_weights(enc_path)
    print(f"  Weights loaded from {enc_path}")

    # Load test windows
    print("\n[2/4] Loading test windows...")
    windows = load_test_windows(args.data, n_sample=2000, no_age=args.no_age)

    # Select physiological windows
    print("\n[3/4] Selecting representative windows...")
    selected = select_windows(windows)

    # Plot
    print("\n[4/4] Plotting...")
    plot_individual_attention(encoder, selected, layers_to_plot, N_HEADS, results_dir)
    # Per-head breakdown for last layer
    plot_per_head_attention(encoder, selected, layer_idx=N_LAYERS - 1, n_heads=N_HEADS,
                             results_dir=results_dir)

    print(f"\n  Done. Plots saved to: {results_dir}/\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualise attention patterns for individual windows'
    )
    parser.add_argument('--run_id',  type=str, required=True)
    parser.add_argument('--data',    type=str, default='data/processed/adults')
    parser.add_argument('--no_age',  action='store_true', default=False)
    parser.add_argument('--layer',   type=str, default='all',
                        help='Layer to visualise: 1-5 or "all" (default: all)')
    args = parser.parse_args()
    main(args)
