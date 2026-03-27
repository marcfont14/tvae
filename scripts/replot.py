"""
replot.py
=========
Regenerate evaluation plots for any completed MTSM run without retraining.

Loads saved weights and reconstructs the test split, then runs the same
plot functions used in experiment_mtsm.py and analyse_H.py.

Requirements per run:
  - encoder_weights.weights.h5  → required for H analysis plots (runs 10+)
  - model_weights.weights.h5    → required for reconstruction plots (runs saved after replot.py added)

If model_weights.weights.h5 is absent, only the H analysis plots are generated.

Usage:
  python scripts/replot.py --run_id run14 --no_age
  python scripts/replot.py --run_id run14 --no_age --plots h    # H analysis only
  python scripts/replot.py --run_id run14 --no_age --plots recon # reconstruction only
  python scripts/replot.py --run_id run14 --no_age --plots all   # both (default)
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ── Constants (must match experiment_mtsm.py exactly) ─────────────────────────

WINDOW_LEN  = 288
N_FEATURES  = 11
CGM_IDX     = 0
PI_IDX      = 1
RA_IDX      = 2
BOLUS_IDX   = 5
CARBS_IDX   = 6
MASK_RATIO   = 0.35
MASK_MIN_LEN = 60
MASK_MAX_LEN = 96
MASK_TOKEN   = 0.0
D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 5
D_FF     = 256
DROPOUT  = 0.2
BATCH_SIZE  = 128
VAL_SPLIT   = 0.1
TEST_SPLIT  = 0.1
DRIVER_EFFECT_STEPS = 24
RESULTS_BASE = 'results/mtsm'
SEED = 42

COLORS = {
    'cgm':   '#111827',
    'pi':    '#7C3AED',
    'ra':    '#059669',
    'bolus': '#DC2626',
    'carbs': '#D97706',
    'norm':  '#2563EB',
}

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── Architecture ──────────────────────────────────────────────────────────────

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


def build_full_model(window_len, n_features, d_model, n_heads, n_layers, d_ff, dropout):
    """Rebuild MTSM model (encoder + 2-layer MLP reconstruction head)."""
    encoder = build_transformer_encoder(
        window_len, n_features, d_model, n_heads, n_layers, d_ff, dropout
    )
    inp  = keras.Input(shape=(window_len, n_features))
    H    = encoder(inp)
    x    = layers.Dense(64, activation='relu')(H)
    out  = layers.Dense(1)(x)
    out  = tf.squeeze(out, axis=-1)
    model = keras.Model(inp, out)
    return model, encoder


# ── Data loading (same logic as experiment_mtsm.py) ───────────────────────────

def index_dataset(processed_dir: str, max_patients=None):
    npz_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('.npz')])
    if max_patients:
        npz_files = npz_files[:max_patients]

    index    = []
    n_before = 0
    n_filtered = 0

    for fname in npz_files:
        fpath = os.path.join(processed_dir, fname)
        data  = np.load(fpath, allow_pickle=True)
        wins  = data['windows']

        bolus = wins[:, :, BOLUS_IDX]
        carbs = wins[:, :, CARBS_IDX]
        cgm   = wins[:, :, CGM_IDX]

        has_driver = ((bolus + carbs) > 0).any(axis=1)
        cgm_std    = cgm.std(axis=1)
        cgm_ok     = (cgm_std > 0.3) & (cgm_std < 4.0)
        keep       = has_driver & cgm_ok

        n_before   += len(wins)
        n_filtered += (~keep).sum()

        for i in np.where(keep)[0]:
            index.append((fpath, int(i)))

    print(f"  Filtered {n_filtered:,} pathological windows → {len(index):,} remaining")
    return index


def load_windows_from_index(index_sample: list) -> np.ndarray:
    windows = []
    for fpath, win_idx in index_sample:
        data = np.load(fpath, allow_pickle=True)
        windows.append(data['windows'][win_idx].astype(np.float32))
    return np.stack(windows, axis=0)


def load_scalers_from_index(index_sample: list) -> tuple:
    cache = {}
    means, stds = [], []
    for fpath, _ in index_sample:
        if fpath not in cache:
            d = np.load(fpath, allow_pickle=True)
            cache[fpath] = (float(d['scaler_mean'][0]), float(d['scaler_std'][0]))
        m, s = cache[fpath]
        means.append(m)
        stds.append(s)
    return np.array(means, np.float32), np.array(stds, np.float32)


def create_mask(window_len, mask_ratio, mask_min_len, mask_max_len):
    mask = np.zeros(window_len, dtype=np.float32)
    n_mask = int(window_len * mask_ratio)
    while mask.sum() < n_mask:
        length = np.random.randint(mask_min_len, mask_max_len + 1)
        start  = np.random.randint(0, window_len - length + 1)
        mask[start:start + length] = 1.0
    return mask


def get_test_index(processed_dir: str):
    """Reconstruct exact test split (same SEED as training)."""
    index    = index_dataset(processed_dir)
    all_fpaths = sorted(list(set(fp for fp, _ in index)))
    n = len(all_fpaths)
    perm = np.random.permutation(n)
    n_test = int(n * TEST_SPLIT)
    test_set = set(all_fpaths[i] for i in perm[:n_test])
    return [(fp, wi) for fp, wi in index if fp in test_set]


# ── Import plot functions from experiment_mtsm and analyse_H ──────────────────

def _import_plot_fns():
    """Import plot functions without triggering TF re-init."""
    import importlib.util, sys

    # experiment_mtsm
    spec = importlib.util.spec_from_file_location(
        'experiment_mtsm',
        os.path.join(os.path.dirname(__file__), 'experiment_mtsm.py')
    )
    mtsm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mtsm)

    # analyse_H
    spec2 = importlib.util.spec_from_file_location(
        'analyse_H',
        os.path.join(os.path.dirname(__file__), 'analyse_H.py')
    )
    ah = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(ah)

    return mtsm, ah


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    results_dir  = os.path.join(RESULTS_BASE, args.run_id)
    enc_path     = os.path.join(results_dir, 'encoder_weights.weights.h5')
    model_path   = os.path.join(results_dir, 'model_weights.weights.h5')

    assert os.path.exists(enc_path), f"Encoder weights not found: {enc_path}"

    # Override architecture from run config if present
    config_path = os.path.join(results_dir, 'run_config.txt')
    d_model, n_heads, d_ff = D_MODEL, N_HEADS, D_FF
    if os.path.exists(config_path):
        with open(config_path) as f:
            for line in f:
                if line.startswith('D_MODEL:'):
                    d_model = int(line.split(':')[1].strip())
                elif line.startswith('N_HEADS:'):
                    n_heads = int(line.split(':')[1].strip())
                elif line.startswith('D_FF:'):
                    d_ff    = int(line.split(':')[1].strip())
        print(f"  Architecture from config: d_model={d_model} n_heads={n_heads} d_ff={d_ff}")

    n_features_model = 10 if args.no_age else N_FEATURES

    print(f"\n── replot.py  run={args.run_id}  plots={args.plots} ──────────────")

    # ── H analysis plots ──────────────────────────────────────────────────────
    if args.plots in ('h', 'all'):
        print("\n── H Analysis Plots ─────────────────────────────────────────────")
        from analyse_H import (
            build_transformer_encoder as build_enc_ah,
            index_test_patients, sample_windows,
            plot_transformer_H_analysis, plot_H_norm_vs_drivers,
            plot_H_circadian, plot_abstraction_trajectory, plot_L1_vs_L5_scatter
        )

        encoder = build_enc_ah(
            WINDOW_LEN, n_features_model, d_model, n_heads, N_LAYERS, d_ff, DROPOUT
        )
        encoder(tf.zeros((1, WINDOW_LEN, n_features_model)), training=False)
        encoder.load_weights(enc_path)
        print(f"  Encoder weights loaded: {enc_path}")

        test_records             = index_test_patients(args.data)
        windows, modalities, ages = sample_windows(test_records, args.n_windows)
        if args.no_age:
            windows = windows[:, :, :10]

        print("\n[1/5] Transformer H analysis...")
        plot_transformer_H_analysis(encoder, windows, N_LAYERS, n_heads, results_dir)

        print("\n[2/5] Event-triggered H norm vs drivers...")
        plot_H_norm_vs_drivers(encoder, windows, results_dir)

        print("\n[3/5] H norm circadian pattern...")
        H_norms_list = []
        for i in range(0, len(windows), 64):
            batch = tf.cast(windows[i:i+64], tf.float32)
            H_b   = encoder(batch, training=False).numpy()
            H_norms_list.append(np.linalg.norm(H_b, axis=-1))
        H_norms = np.concatenate(H_norms_list, axis=0)
        plot_H_circadian(H_norms, windows, results_dir)

        print("\n[4/5] Abstraction trajectory...")
        plot_abstraction_trajectory(encoder, windows, N_LAYERS, results_dir)

        print("\n[5/5] L1 vs L5 scatter...")
        plot_L1_vs_L5_scatter(encoder, windows, N_LAYERS, results_dir)

    # ── Reconstruction plots ───────────────────────────────────────────────────
    if args.plots in ('recon', 'all'):
        print("\n── Reconstruction Plots ─────────────────────────────────────────")
        if not os.path.exists(model_path):
            print(f"  [SKIP] model_weights.weights.h5 not found in {results_dir}/")
            print(f"  Reconstruction plots require the full model weights saved during training.")
            print(f"  These are saved automatically from this version onwards.")
        else:
            from experiment_mtsm import (
                plot_training_curves, plot_reconstruction_examples,
                plot_reconstruction_quality, plot_reconstruction_timeseries
            )

            model, encoder = build_full_model(
                WINDOW_LEN, n_features_model, d_model, n_heads, N_LAYERS, d_ff, DROPOUT
            )
            model(tf.zeros((1, WINDOW_LEN, n_features_model)), training=False)
            model.load_weights(model_path)
            print(f"  Full model weights loaded: {model_path}")

            print("\n  Indexing test split...")
            test_index = get_test_index(args.data)

            N_PLOT = 500
            if len(test_index) > N_PLOT:
                rng_idx    = np.random.choice(len(test_index), N_PLOT, replace=False)
                plot_index = [test_index[i] for i in rng_idx]
            else:
                plot_index = test_index

            test_windows = load_windows_from_index(plot_index)
            if args.no_age:
                test_windows = test_windows[:, :, :10]

            masks_test = np.stack([
                create_mask(WINDOW_LEN, MASK_RATIO, MASK_MIN_LEN, MASK_MAX_LEN)
                for _ in range(len(test_windows))
            ])

            scaler_mean_cgm, scaler_std_cgm = load_scalers_from_index(plot_index)

            print("\n  Generating reconstruction_examples.png...")
            plot_reconstruction_examples(
                model, test_windows, masks_test,
                mask_ratio=MASK_RATIO,
                mask_min_len=MASK_MIN_LEN,
                mask_max_len=MASK_MAX_LEN,
                save_path=os.path.join(results_dir, 'reconstruction_examples.png')
            )

            print("\n  Generating reconstruction_quality.png...")
            plot_reconstruction_quality(
                model, test_windows, masks_test,
                scaler_mean_cgm, scaler_std_cgm,
                save_path=os.path.join(results_dir, 'reconstruction_quality.png')
            )

            print("\n  Generating reconstruction_timeseries.png...")
            plot_reconstruction_timeseries(
                model, test_windows, masks_test,
                save_path=os.path.join(results_dir, 'reconstruction_timeseries.png')
            )

    print(f"\n  Done. Plots saved to: {results_dir}/\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Regenerate plots for a completed MTSM run without retraining'
    )
    parser.add_argument('--run_id',    type=str, required=True,
                        help='Run ID (loads from results/mtsm/{run_id}/)')
    parser.add_argument('--data',      type=str, default='data/processed/adults',
                        help='Path to processed .npz directory')
    parser.add_argument('--no_age',    action='store_true', default=False,
                        help='Drop age_norm (feature 10) — must match training flag')
    parser.add_argument('--plots',     type=str, default='all',
                        choices=['all', 'h', 'recon'],
                        help='Which plots to generate: all | h (H analysis) | recon (reconstruction)')
    parser.add_argument('--n_windows', type=int, default=1500,
                        help='Windows for H analysis (default 1500)')
    args = parser.parse_args()
    main(args)
