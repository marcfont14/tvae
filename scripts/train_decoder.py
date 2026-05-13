"""
train_decoder.py
================
Stage 1 pre-training for the causal decoder (GPT-style NTP) — decoder2 variant.

Improvements over decoder:
  1. Delta targets: predicts Δ = CGM[t+1] − CGM[t] instead of absolute CGM[t+1].
     Naive baseline is zero (flat), so Huber penalises excursions correctly.
  2. Multi-step auxiliary losses: heads at t+6 (30min), t+12 (1h), t+24 (2h)
     predict cumulative deltas from H[t], aligning pre-training with downstream
     forecasting horizon and reducing AR rollout error compounding.

Architecture:
  PrefixLMBlock(prefix_len=0) × 5 — fully causal lower-triangular attention mask.
  Identical hyperparams to encoder2 (d_model=128, 4 heads, d_ff=256, 5 layers).
  No CLS token. Returns [H (B, 288, 128), h_last (B, 128)].

Outputs (dict):
  ntp_output:   (B, 287)     — 1-step delta predictions, driver-weighted loss
  multi_output: (B, 264, 3)  — [delta_6, delta_12, delta_24] cumulative deltas
  Loss weights: ntp=1.0, multi=0.25

Usage:
  python -u scripts/train_decoder.py \\
      --data data/processed/adults --run_id decoder2 --no_age --epochs 70 \\
      2>&1 | tee results/mtsm/decoder2_log.txt
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.encoder import build_decoder, WINDOW_LEN, N_FEATURES, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass

# ── Constants ─────────────────────────────────────────────────────────────────

CGM_IDX             = 0
BOLUS_IDX           = 5
CARBS_IDX           = 6
DRIVER_LOSS_WEIGHT  = 3.0
DRIVER_EFFECT_STEPS = 24   # 2h @ 5-min steps
T_MS                = WINDOW_LEN - 24  # 264 — positions with valid t+24 target
BATCH_SIZE          = 128
LR                  = 1e-3
VAL_SPLIT           = 0.1
TEST_SPLIT          = 0.1
SEED                = 42
RESULTS_BASE        = 'results/mtsm'

# Cross-entropy binning (decoder_clean)
K_BINS          = 200
CGM_MG_MIN      = 40.0
CGM_MG_MAX      = 400.0
GLOBAL_CGM_MEAN = 144.40
GLOBAL_CGM_STD  = 57.11
BIN_Z_MIN = (CGM_MG_MIN - GLOBAL_CGM_MEAN) / GLOBAL_CGM_STD
BIN_Z_MAX = (CGM_MG_MAX - GLOBAL_CGM_MEAN) / GLOBAL_CGM_STD

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── Dataset helpers ───────────────────────────────────────────────────────────

def index_dataset(processed_dir: str, max_patients: int = None,
                  allowed_files: set = None) -> list:
    npz_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('.npz')])
    if not npz_files:
        raise FileNotFoundError(f'No .npz files found in {processed_dir}')
    if allowed_files is not None:
        npz_files = [f for f in npz_files if f in allowed_files]
        print(f'  (--pretrain_patients: using {len(npz_files)} non-test patients)')
    if max_patients is not None:
        npz_files = npz_files[:max_patients]
        print(f'  (--max_patients {max_patients}: loading subset)')

    index, n_before, n_filtered = [], 0, 0
    for fname in npz_files:
        fpath = os.path.join(processed_dir, fname)
        data  = np.load(fpath, allow_pickle=True)
        wins  = data['windows'].astype(np.float32)
        bolus = wins[:, :, BOLUS_IDX]
        carbs = wins[:, :, CARBS_IDX]
        cgm   = wins[:, :, CGM_IDX]
        keep  = ((bolus + carbs) > 0).any(axis=1) & (cgm.std(axis=1) > 0.3) & (cgm.std(axis=1) < 4.0)
        n_before   += len(wins)
        n_filtered += (~keep).sum()
        for i in np.where(keep)[0]:
            index.append((fpath, int(i)))

    pct = n_filtered / n_before * 100 if n_before else 0
    print(f'  Filtered {n_filtered:,} windows ({pct:.1f}%) → {len(index):,} remaining')
    print(f'  Patients: {len(npz_files)}   Windows: {len(index):,}')
    return index


def apply_decoder_targets(wins: np.ndarray, no_age: bool = False):
    """
    Build delta-NTP and multi-step targets. No input masking.

    Returns:
      x:       (N, 288, 10)   full window, unmasked
      Y_ntp:   (N, 287, 2)    [delta_1, driver_weight] — 1-step delta targets
      Y_multi: (N, T_MS, 3)   [delta_6, delta_12, delta_24] cumulative deltas
    """
    N   = len(wins)
    x   = wins.copy()
    if no_age:
        x = x[:, :, :10]

    bolus        = wins[:, :, BOLUS_IDX]
    carbs        = wins[:, :, CARBS_IDX]
    driver_event = ((bolus + carbs) > 0).astype(np.float32)
    driver_infl  = np.zeros((N, WINDOW_LEN), dtype=np.float32)
    for offset in range(1, DRIVER_EFFECT_STEPS + 1):
        driver_infl[:, offset:] += driver_event[:, :-offset]
    driver_infl   = np.clip(driver_infl, 0, 1)
    driver_weight = np.where(driver_infl > 0, DRIVER_LOSS_WEIGHT, 1.0).astype(np.float32)

    cgm     = wins[:, :, CGM_IDX]
    delta_1 = cgm[:, 1:] - cgm[:, :-1]       # (N, 287)
    dw_next = driver_weight[:, 1:]            # (N, 287)
    Y_ntp   = np.stack([delta_1, dw_next], axis=-1).astype(np.float32)  # (N, 287, 2)

    # Cumulative deltas from H[t]: cgm[t+k] - cgm[t] for t in 0..T_MS-1
    delta_6  = cgm[:, 6:6   + T_MS] - cgm[:, :T_MS]
    delta_12 = cgm[:, 12:12 + T_MS] - cgm[:, :T_MS]
    delta_24 = cgm[:, 24:24 + T_MS] - cgm[:, :T_MS]
    Y_multi  = np.stack([delta_6, delta_12, delta_24], axis=-1).astype(np.float32)  # (N, T_MS, 3)

    return x.astype(np.float32), Y_ntp, Y_multi


def make_window_dataset_decoder(index: list, shuffle: bool, no_age: bool = False):
    from collections import defaultdict
    patient_to_windows = defaultdict(list)
    for fpath, win_idx in index:
        patient_to_windows[fpath].append(win_idx)
    fpaths = list(patient_to_windows.keys())

    def generator():
        order = np.random.permutation(len(fpaths)) if shuffle else range(len(fpaths))
        for pi in order:
            fpath       = fpaths[pi]
            win_indices = np.array(patient_to_windows[fpath], dtype=np.int32)
            try:
                data = np.load(fpath, allow_pickle=True)
                wins = data['windows'][win_indices].astype(np.float32)
            except Exception as e:
                print(f'[WARN] skipping {fpath}: {e}', flush=True)
                continue
            x, Y_ntp, Y_multi = apply_decoder_targets(wins, no_age=no_age)
            if shuffle:
                perm = np.random.permutation(len(wins))
                x, Y_ntp, Y_multi = x[perm], Y_ntp[perm], Y_multi[perm]
            yield x, {'ntp_output': Y_ntp, 'multi_output': Y_multi}

    n_feat = 10 if no_age else N_FEATURES
    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, WINDOW_LEN, n_feat), dtype=tf.float32),
            {
                'ntp_output':   tf.TensorSpec(shape=(None, WINDOW_LEN - 1, 2), dtype=tf.float32),
                'multi_output': tf.TensorSpec(shape=(None, T_MS, 3),           dtype=tf.float32),
            }
        )
    )
    ds = ds.unbatch()
    if shuffle:
        ds = ds.shuffle(buffer_size=5_000, seed=SEED)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ── Losses and metrics ────────────────────────────────────────────────────────

class NTPHuberLoss(keras.losses.Loss):
    """
    1-step delta Huber loss with driver weighting.
    y_true: (B, 287, 2) = [delta_1, driver_weight]
    y_pred: (B, 287)
    """
    def __init__(self, delta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def call(self, y_true, y_pred):
        target        = y_true[:, :, 0]   # delta_1
        driver_weight = y_true[:, :, 1]
        abs_err = tf.abs(target - y_pred)
        huber   = tf.where(abs_err <= self.delta,
                           0.5 * tf.square(abs_err),
                           self.delta * (abs_err - 0.5 * self.delta))
        return tf.reduce_mean(huber * driver_weight)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'delta': self.delta})
        return cfg


class MultiStepHuberLoss(keras.losses.Loss):
    """
    Multi-step cumulative delta Huber loss (unweighted).
    y_true, y_pred: (B, T_MS, 3)
    """
    def __init__(self, delta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def call(self, y_true, y_pred):
        abs_err = tf.abs(y_true - y_pred)
        huber   = tf.where(abs_err <= self.delta,
                           0.5 * tf.square(abs_err),
                           self.delta * (abs_err - 0.5 * self.delta))
        return tf.reduce_mean(huber)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'delta': self.delta})
        return cfg


class NTPMae(keras.metrics.Metric):
    """Unweighted MAE over all 287 1-step delta positions (z-score units)."""
    def __init__(self, name='ntp_mae', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        target = y_true[:, :, 0]
        self.total.assign_add(tf.reduce_sum(tf.abs(target - y_pred)))
        self.count.assign_add(tf.cast(tf.size(target), tf.float32))

    def result(self):
        return self.total / (self.count + 1e-8)

    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)


# ── CE NTP helpers (decoder_clean) ───────────────────────────────────────────

def cgm_z_to_bin(cgm_z: np.ndarray) -> np.ndarray:
    t = (cgm_z - BIN_Z_MIN) / (BIN_Z_MAX - BIN_Z_MIN)
    return np.clip((t * K_BINS).astype(np.int32), 0, K_BINS - 1).astype(np.float32)


class NTPCELoss(keras.losses.Loss):
    """
    Weighted sparse cross-entropy for absolute NTP prediction.
    y_true: (B, 287, 2) = [bin_idx, driver_weight]
    y_pred: (B, 287, K_BINS) — raw logits
    """
    def call(self, y_true, y_pred):
        bin_idx       = tf.cast(y_true[:, :, 0], tf.int32)
        driver_weight = y_true[:, :, 1]
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=bin_idx, logits=y_pred
        )  # (B, 287)
        return tf.reduce_mean(ce * driver_weight)

    def get_config(self):
        return super().get_config()


class NTPCEAccuracy(keras.metrics.Metric):
    """Top-1 bin accuracy for NTP CE head."""
    def __init__(self, name='ntp_acc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.count   = self.add_weight(name='count',   initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        bin_idx  = tf.cast(y_true[:, :, 0], tf.int32)
        pred_bin = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        self.correct.assign_add(tf.reduce_sum(tf.cast(tf.equal(bin_idx, pred_bin), tf.float32)))
        self.count.assign_add(tf.cast(tf.size(bin_idx), tf.float32))

    def result(self):
        return self.correct / (self.count + 1e-8)

    def reset_state(self):
        self.correct.assign(0.)
        self.count.assign(0.)


# ── Model builder ─────────────────────────────────────────────────────────────

def apply_decoder_targets_clean(wins: np.ndarray, no_age: bool = False):
    """
    Build absolute NTP targets as bin indices (decoder_clean).
    Input:  CGM[0..T-1]
    Target: bin_idx(CGM[1..T])  — absolute next-token prediction

    Returns:
      x:     (N, 288, 10)    full window, unmasked
      Y_ntp: (N, 287, 2)     [bin_idx, driver_weight]
    """
    N = len(wins)
    x = wins.copy()
    if no_age:
        x = x[:, :, :10]

    bolus        = wins[:, :, BOLUS_IDX]
    carbs        = wins[:, :, CARBS_IDX]
    driver_event = ((bolus + carbs) > 0).astype(np.float32)
    driver_infl  = np.zeros((N, WINDOW_LEN), dtype=np.float32)
    for offset in range(1, DRIVER_EFFECT_STEPS + 1):
        driver_infl[:, offset:] += driver_event[:, :-offset]
    driver_infl   = np.clip(driver_infl, 0, 1)
    driver_weight = np.where(driver_infl > 0, DRIVER_LOSS_WEIGHT, 1.0).astype(np.float32)

    cgm       = wins[:, :, CGM_IDX]
    next_bins = cgm_z_to_bin(cgm[:, 1:])     # (N, 287) — absolute next value as bin index
    dw_next   = driver_weight[:, 1:]          # (N, 287)
    Y_ntp     = np.stack([next_bins, dw_next], axis=-1).astype(np.float32)  # (N, 287, 2)

    return x.astype(np.float32), Y_ntp


def build_ntp_clean_model(n_features: int, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                          n_layers: int = N_LAYERS, d_ff: int = D_FF,
                          dropout: float = DROPOUT) -> tuple:
    """
    decoder_clean: absolute NTP with cross-entropy head.
    Output: {'ntp_output': (B, 287, K_BINS)} — raw logits.
    """
    decoder   = build_decoder(n_features=n_features, d_model=d_model, n_heads=n_heads,
                               n_layers=n_layers, d_ff=d_ff, dropout=dropout)
    inp       = keras.Input(shape=(WINDOW_LEN, n_features), name='input')
    H, h_last = decoder(inp)

    H_ntp  = layers.Lambda(lambda z: z[:, :-1, :], name='H_ntp')(H)   # (B, 287, d)
    ntp    = layers.Dense(64, activation='relu', name='ntp_hidden')(H_ntp)
    ntp    = layers.Dense(K_BINS, name='ntp_output')(ntp)              # (B, 287, K_BINS)

    model = keras.Model(inp, ntp, name='NTP_Decoder_Clean')
    return model, decoder


def build_ntp_model(n_features: int, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                    n_layers: int = N_LAYERS, d_ff: int = D_FF,
                    dropout: float = DROPOUT) -> tuple:
    """
    Decoder2 NTP model with delta targets and multi-step auxiliary heads.
    Outputs dict: {'ntp_output': (B, 287), 'multi_output': (B, T_MS, 3)}.
    Returns (full_model, decoder).
    """
    decoder   = build_decoder(n_features=n_features, d_model=d_model, n_heads=n_heads,
                               n_layers=n_layers, d_ff=d_ff, dropout=dropout)
    inp       = keras.Input(shape=(WINDOW_LEN, n_features), name='input')
    H, h_last = decoder(inp)

    # 1-step delta head
    H_ntp  = layers.Lambda(lambda z: z[:, :-1, :], name='H_ntp')(H)   # (B, 287, d)
    ntp    = layers.Dense(64, activation='relu', name='ntp_hidden')(H_ntp)
    ntp    = layers.Dense(1,  name='ntp_head')(ntp)
    ntp    = layers.Reshape((WINDOW_LEN - 1,), name='ntp_output')(ntp) # (B, 287)

    # Multi-step cumulative delta head
    H_ms   = layers.Lambda(lambda z: z[:, :-24, :], name='H_ms')(H)   # (B, T_MS, d)
    ms     = layers.Dense(64, activation='relu', name='ms_hidden')(H_ms)
    ms     = layers.Dense(3,  name='multi_output')(ms)                 # (B, T_MS, 3)

    model = keras.Model(inp, {'ntp_output': ntp, 'multi_output': ms}, name='NTP_Decoder2')
    return model, decoder


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['loss'],     label='train loss')
    axes[0].plot(history['val_loss'], label='val loss')
    axes[0].set_title('Combined Loss (NTP×1.0 + Multi×0.25)')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    mae_key     = 'ntp_output_ntp_mae'
    val_mae_key = 'val_ntp_output_ntp_mae'
    if mae_key in history:
        axes[1].plot(history[mae_key],     label='train delta-MAE')
        axes[1].plot(history[val_mae_key], label='val delta-MAE')
        axes[1].set_title('NTP Delta MAE (z-score)')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'  Saved: {save_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Causal decoder NTP pre-training')
    parser.add_argument('--data',              default='data/processed/adults')
    parser.add_argument('--run_id',            default='decoder_clean')
    parser.add_argument('--epochs',            type=int,   default=70)
    parser.add_argument('--no_age',            action='store_true')
    parser.add_argument('--max_patients',      type=int,   default=None)
    parser.add_argument('--lr',                type=float, default=LR)
    parser.add_argument('--ce_head',           action='store_true', default=False,
                        help='Use cross-entropy NTP head (decoder_clean). '
                             'Predicts absolute next CGM as bin index over K_BINS.')
    parser.add_argument('--pretrain_patients', type=str, default=None,
                        help='Path to .txt with allowed patient filenames (exclude test set).')
    return parser.parse_args()


def main():
    args        = parse_args()
    results_dir = os.path.join(RESULTS_BASE, args.run_id)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*52}")
    print(f'  Causal Decoder2 — Delta NTP + Multi-step')
    print(f'  Run ID:    {args.run_id}')
    print(f'  Data:      {args.data}')
    print(f'  Epochs:    {args.epochs}')
    print(f'  no_age:    {args.no_age}')
    print(f'  Arch:      d={D_MODEL}  heads={N_HEADS}  ff={D_FF}  layers={N_LAYERS}')
    print(f'  T_MS:      {T_MS} positions (t+6/12/24 aux losses)')
    print(f"{'='*52}\n")

    print('  Indexing dataset...')
    allowed_files = None
    if args.pretrain_patients:
        with open(args.pretrain_patients) as fh:
            allowed_files = set(line.strip() for line in fh if line.strip())
        print(f'  Pretrain whitelist: {len(allowed_files)} patients')
    index = index_dataset(args.data, args.max_patients, allowed_files=allowed_files)

    all_fpaths = sorted(set(fp for fp, _ in index))
    n          = len(all_fpaths)
    perm       = np.random.permutation(n)
    n_test     = int(n * TEST_SPLIT)
    n_val      = int(n * VAL_SPLIT)
    test_set   = set(all_fpaths[i] for i in perm[:n_test])
    val_set    = set(all_fpaths[i] for i in perm[n_test:n_test + n_val])
    train_set  = set(all_fpaths[i] for i in perm[n_test + n_val:])

    train_index = [(fp, wi) for fp, wi in index if fp in train_set]
    val_index   = [(fp, wi) for fp, wi in index if fp in val_set]

    print(f'  Patient split — Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}')
    print(f'  Window split  — Train: {len(train_index):,}  Val: {len(val_index):,}')

    n_features = 10 if args.no_age else N_FEATURES

    if args.ce_head:
        # ── decoder_clean path: absolute NTP with cross-entropy ───────────────
        print(f'\n  [decoder_clean] CE NTP head (K={K_BINS} bins, absolute targets)')

        def gen_clean(idx_list, shuffle):
            from collections import defaultdict
            p2w = defaultdict(list)
            for fp, wi in idx_list:
                p2w[fp].append(wi)
            fps = list(p2w.keys())
            def generator():
                order = np.random.permutation(len(fps)) if shuffle else range(len(fps))
                for pi in order:
                    fp = fps[pi]
                    try:
                        data = np.load(fp, allow_pickle=True)
                        wins = data['windows'][np.array(p2w[fp])].astype(np.float32)
                    except Exception as e:
                        print(f'[WARN] {fp}: {e}', flush=True)
                        continue
                    x, Y = apply_decoder_targets_clean(wins, no_age=args.no_age)
                    if shuffle:
                        perm = np.random.permutation(len(wins))
                        x, Y = x[perm], Y[perm]
                    yield x, Y
            nf = 10 if args.no_age else N_FEATURES
            ds = tf.data.Dataset.from_generator(
                generator,
                output_signature=(
                    tf.TensorSpec(shape=(None, WINDOW_LEN, nf),    dtype=tf.float32),
                    tf.TensorSpec(shape=(None, WINDOW_LEN - 1, 2), dtype=tf.float32),
                )
            ).unbatch()
            if shuffle:
                ds = ds.shuffle(5_000, seed=SEED)
            return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        train_ds = gen_clean(train_index, shuffle=True).repeat()
        val_ds   = gen_clean(val_index,   shuffle=False)
        model, decoder = build_ntp_clean_model(n_features)
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=1e-4),
            loss=NTPCELoss(),
            metrics=[NTPCEAccuracy()],
        )
    else:
        # ── Original decoder2 path ────────────────────────────────────────────
        train_ds = make_window_dataset_decoder(train_index, shuffle=True,  no_age=args.no_age).repeat()
        val_ds   = make_window_dataset_decoder(val_index,   shuffle=False, no_age=args.no_age)
        model, decoder = build_ntp_model(n_features)
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=1e-4),
            loss={
                'ntp_output':   NTPHuberLoss(delta=1.0),
                'multi_output': MultiStepHuberLoss(delta=1.0),
            },
            loss_weights={'ntp_output': 1.0, 'multi_output': 0.25},
            metrics={'ntp_output': [NTPMae()]}
        )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        ),
    ]
    steps_per_epoch = len(train_index) // BATCH_SIZE
    history_obj = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, steps_per_epoch=steps_per_epoch,
        callbacks=callbacks, verbose=1
    )
    history = history_obj.history

    # Save decoder backbone weights
    encoder_path = os.path.join(results_dir, 'encoder_weights.weights.h5')
    decoder.save_weights(encoder_path)
    print(f'\n  Decoder weights saved: {encoder_path}')

    model.save_weights(os.path.join(results_dir, 'model_weights.weights.h5'))

    if not args.ce_head:
        # Save standalone NTP head for legacy load_ntp_head2
        h_inp    = keras.Input(shape=(None, D_MODEL), name='h_input')
        ntp_h    = model.get_layer('ntp_hidden')(h_inp)
        ntp_h    = model.get_layer('ntp_head')(ntp_h)
        ntp_head = keras.Model(h_inp, ntp_h, name='NTPHeadStandalone')
        ntp_head.save_weights(os.path.join(results_dir, 'ntp_head_weights.weights.h5'))
        print(f'  NTP head weights saved: {results_dir}/ntp_head_weights.weights.h5')

    plot_training_curves(history, os.path.join(results_dir, 'training_curves.png'))

    best_val_loss = min(history['val_loss'])
    mae_key       = 'val_ntp_output_ntp_mae'
    best_val_mae  = min(history.get(mae_key, [float('nan')]))
    n_epochs      = len(history['loss'])
    print(f'\n  Best val_loss:         {best_val_loss:.4f}')
    print(f'  Best val delta-MAE:    {best_val_mae:.4f} (z-score)')
    print(f'  Epochs trained:        {n_epochs}')
    print(f'\n  Results saved to: {results_dir}')


if __name__ == '__main__':
    main()
