"""
experiment_jepa.py
==================
Joint-Embedding Predictive Architecture (JEPA) — Stage 1 H enrichment (run20).

Instead of reconstructing raw CGM signal, the model predicts in representation space:
  - Context encoder (backprop):  processes masked CGM window → H_ctx
  - Target encoder (EMA):        processes original window   → H_target (no gradient)
  - Predictor (2-layer MLP):     H_ctx → H_pred (discarded after training)

Loss: MSE(H_pred, H_target) on masked positions only — no raw signal reconstruction.

Hypothesis: predicting in H-space forces the encoder to learn abstract physiological
representations rather than interpolation-based signal reconstruction. Expected result:
R²_probe_L5 drops significantly vs run14 baseline (0.939) while Σ|r_L5| stays healthy.

The target encoder (EMA-stabilised) is saved as encoder_weights.weights.h5 for Stage 2.
Training uses model.fit() via a Keras subclassed model — XLA compatible.

Usage:
  python -u scripts/experiment_jepa.py --data data/processed/adults --epochs 70 --run_id run20 --no_age | tee results/mtsm/run20_log.txt

Outputs (results/mtsm/{run_id}/):
  encoder_weights.weights.h5   target encoder weights (Stage 2 ready)
  training_curves.png          JEPA loss train/val per epoch
  run_config.txt               exact configuration for reproducibility
  H plots via replot.py:
    python -u scripts/replot.py --run_id run20 --no_age --plots h
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ── Constants (matching experiment_mtsm.py) ───────────────────────────────────
WINDOW_LEN  = 288
N_FEATURES  = 11
CGM_IDX     = 0
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

BATCH_SIZE   = 128
LR           = 1e-3
EMA_MOMENTUM = 0.996
VAL_SPLIT    = 0.1
TEST_SPLIT   = 0.1

RESULTS_BASE = 'results/mtsm'
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ── Positional Encoding ───────────────────────────────────────────────────────
def get_positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    positions = np.arange(seq_len)[:, np.newaxis]
    dims      = np.arange(d_model)[np.newaxis, :]
    angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)


# ── Transformer Encoder (identical to experiment_mtsm.py) ────────────────────
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


# ── Predictor ─────────────────────────────────────────────────────────────────
def build_predictor(window_len, d_model):
    """
    2-layer MLP applied independently per timestep.
    Adapts context H to match target H — discarded after pretraining.
    """
    inp = keras.Input(shape=(window_len, d_model), name='pred_input')
    x   = layers.Dense(d_model, activation='relu', name='pred_hidden')(inp)
    x   = layers.Dense(d_model, name='pred_out')(x)
    return keras.Model(inp, x, name='Predictor')


# ── JEPA Model ────────────────────────────────────────────────────────────────
class JEPAModel(keras.Model):
    """
    Context encoder (backprop) + target encoder (EMA) + predictor.

    train_step is overridden so model.fit() compiles the entire step as a single
    tf.function — XLA compatible, no raw Python training loop.

    After training: save target_encoder.weights (EMA-stable) for Stage 2.
    """
    def __init__(self, context_encoder, target_encoder, predictor, ema_momentum):
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder  = target_encoder
        self.predictor       = predictor
        self.ema_m           = tf.constant(ema_momentum, dtype=tf.float32)
        # Target encoder is never touched by the optimiser
        self.target_encoder.trainable = False

    def call(self, x_masked, training=False):
        H_ctx  = self.context_encoder(x_masked, training=training)
        H_pred = self.predictor(H_ctx, training=training)
        return H_pred

    def _jepa_loss(self, H_pred, H_target, mask):
        # MSE in H-space, averaged over masked timesteps and d_model dimensions
        mask_3d = tf.cast(tf.expand_dims(mask, -1), tf.float32)   # (B, 288, 1)
        sq_err  = tf.square(H_pred - H_target) * mask_3d           # (B, 288, D)
        n_mask  = tf.reduce_sum(mask) + 1e-8
        d       = tf.cast(tf.shape(H_target)[-1], tf.float32)
        return tf.reduce_sum(sq_err) / (n_mask * d)

    def train_step(self, data):
        x_masked, (x_original, mask) = data

        # Target representation — computed outside tape, no gradient
        H_target = self.target_encoder(x_original, training=False)

        with tf.GradientTape() as tape:
            H_ctx  = self.context_encoder(x_masked, training=True)
            H_pred = self.predictor(H_ctx, training=True)
            loss   = self._jepa_loss(H_pred, H_target, mask)

        tvars = (self.context_encoder.trainable_variables +
                 self.predictor.trainable_variables)
        grads = tape.gradient(loss, tvars)
        self.optimizer.apply_gradients(zip(grads, tvars))

        # EMA update: θ_target ← m·θ_target + (1−m)·θ_context
        for c_v, t_v in zip(self.context_encoder.variables,
                             self.target_encoder.variables):
            t_v.assign(self.ema_m * t_v + (1.0 - self.ema_m) * c_v)

        return {'loss': loss}

    def test_step(self, data):
        x_masked, (x_original, mask) = data
        H_target = self.target_encoder(x_original, training=False)
        H_ctx    = self.context_encoder(x_masked,  training=False)
        H_pred   = self.predictor(H_ctx, training=False)
        loss     = self._jepa_loss(H_pred, H_target, mask)
        return {'loss': loss}


# ── Masking ───────────────────────────────────────────────────────────────────
def create_mask(window_len, mask_ratio, min_len, max_len):
    mask          = np.zeros(window_len, dtype=np.float32)
    target_len    = int(window_len * mask_ratio)
    masked_so_far = 0
    attempts      = 0
    while masked_so_far < target_len and attempts < 50:
        span_len = np.random.randint(min_len, max_len + 1)
        start    = np.random.randint(0, window_len - span_len)
        mask[start:start + span_len] = 1
        masked_so_far = mask.sum()
        attempts += 1
    return mask


def apply_masks_jepa(wins, mask_ratio, mask_min_len, mask_max_len, no_age):
    """
    Returns (x_masked, x_original, masks).
      x_masked:   CGM zeroed at mask positions — input to context encoder
      x_original: unmodified — input to target encoder
      masks:      (N, 288) binary array
    """
    N          = len(wins)
    x_original = wins.copy()
    x_masked   = wins.copy()
    masks      = np.zeros((N, WINDOW_LEN), dtype=np.float32)
    for i in range(N):
        m              = create_mask(WINDOW_LEN, mask_ratio, mask_min_len, mask_max_len)
        masks[i]       = m
        x_masked[i, m.astype(bool), CGM_IDX] = MASK_TOKEN
    if no_age:
        x_original = x_original[:, :, :10]
        x_masked   = x_masked[:, :, :10]
    return x_masked.astype(np.float32), x_original.astype(np.float32), masks


# ── Dataset ───────────────────────────────────────────────────────────────────
def make_jepa_dataset(index, shuffle, mask_ratio, mask_min_len, mask_max_len, no_age):
    """
    Yields (x_masked, (x_original, mask)) per batch.
    Groups windows by patient file — one .npz opened once per epoch.
    """
    patient_to_windows = defaultdict(list)
    for fpath, win_idx in index:
        patient_to_windows[fpath].append(win_idx)
    fpaths = list(patient_to_windows.keys())
    n_feat = 10 if no_age else N_FEATURES

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
            x_masked, x_original, masks = apply_masks_jepa(
                wins, mask_ratio, mask_min_len, mask_max_len, no_age
            )
            if shuffle:
                perm = np.random.permutation(len(wins))
                x_masked, x_original, masks = (
                    x_masked[perm], x_original[perm], masks[perm]
                )
            yield x_masked, (x_original, masks)

    feat_spec = tf.TensorSpec(shape=(None, WINDOW_LEN, n_feat), dtype=tf.float32)
    mask_spec = tf.TensorSpec(shape=(None, WINDOW_LEN),         dtype=tf.float32)
    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(feat_spec, (feat_spec, mask_spec))
    )
    ds = ds.unbatch()
    if shuffle:
        ds = ds.shuffle(buffer_size=5_000, seed=SEED)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ── Data indexing (identical to experiment_mtsm.py) ──────────────────────────
def index_dataset(processed_dir, max_patients=None):
    npz_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('.npz')])
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {processed_dir}")
    if max_patients is not None:
        npz_files = npz_files[:max_patients]
        print(f"  (--max_patients {max_patients}: loading subset)")
    index = []
    n_before = n_filtered = 0
    for fname in npz_files:
        fpath = os.path.join(processed_dir, fname)
        data  = np.load(fpath, allow_pickle=True)
        wins  = data['windows'].astype(np.float32)
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
    pct = n_filtered / n_before * 100 if n_before > 0 else 0
    print(f"  Filtered {n_filtered:,} pathological windows ({pct:.1f}%) → {len(index):,} remaining")
    print(f"  Patients: {len(npz_files)}   Windows: {len(index):,}")
    return index


# ── Training curves ───────────────────────────────────────────────────────────
def plot_training_curves(history, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history['loss'],     color='#2563EB', lw=2,   label='train')
    ax.plot(history['val_loss'], color='#2563EB', lw=1.5, ls='--', label='val')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('JEPA Loss (H-space MSE on masked positions)', fontsize=11)
    ax.set_title('JEPA Training Curves', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    results_dir = os.path.join(RESULTS_BASE, args.run_id)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Index + split
    print("\n── Indexing dataset ──────────────────────────────────────────────")
    all_index = index_dataset(args.data, args.max_patients)
    rng = np.random.RandomState(SEED)
    rng.shuffle(all_index)
    n       = len(all_index)
    n_test  = int(n * TEST_SPLIT)
    n_val   = int(n * VAL_SPLIT)
    test_index  = all_index[:n_test]
    val_index   = all_index[n_test:n_test + n_val]
    train_index = all_index[n_test + n_val:]
    print(f"  Train: {len(train_index):,}  Val: {len(val_index):,}  Test: {len(test_index):,}")

    # 2. Build datasets
    n_features_model = 10 if args.no_age else N_FEATURES
    ds_kwargs = dict(mask_ratio=MASK_RATIO, mask_min_len=MASK_MIN_LEN,
                     mask_max_len=MASK_MAX_LEN, no_age=args.no_age)
    train_ds = make_jepa_dataset(train_index, shuffle=True,  **ds_kwargs).repeat()
    val_ds   = make_jepa_dataset(val_index,   shuffle=False, **ds_kwargs)

    # 3. Build model
    print("\n── Building model ────────────────────────────────────────────────")
    context_encoder = build_transformer_encoder(
        WINDOW_LEN, n_features_model, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT
    )
    target_encoder = build_transformer_encoder(
        WINDOW_LEN, n_features_model, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT
    )
    predictor = build_predictor(WINDOW_LEN, D_MODEL)

    # Initialise both encoders, then copy weights so they start identical
    dummy = tf.zeros((1, WINDOW_LEN, n_features_model))
    context_encoder(dummy)
    target_encoder(dummy)
    target_encoder.set_weights(context_encoder.get_weights())

    model = JEPAModel(context_encoder, target_encoder, predictor, EMA_MOMENTUM)
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=LR, weight_decay=1e-4))
    model(dummy)  # build

    context_encoder.summary()
    predictor.summary()

    # 4. Save run config (keys match what replot.py expects)
    config_path = os.path.join(results_dir, 'run_config.txt')
    with open(config_path, 'w') as f:
        f.write("JEPA Run Configuration\n")
        f.write("=" * 40 + "\n")
        f.write(f"data: {args.data}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"max_patients: {args.max_patients}\n")
        f.write(f"run_id: {args.run_id}\n")
        f.write(f"no_age: {args.no_age}\n")
        f.write(f"mask_ratio: {MASK_RATIO}\n")
        f.write(f"mask_max_len: {MASK_MAX_LEN}\n")
        f.write(f"ema_momentum: {EMA_MOMENTUM}\n")
        f.write("\nFixed hyperparameters:\n")
        f.write(f"D_MODEL: {D_MODEL}\n")
        f.write(f"N_HEADS: {N_HEADS}\n")
        f.write(f"N_LAYERS: {N_LAYERS}\n")
        f.write(f"D_FF: {D_FF}\n")
        f.write(f"DROPOUT: {DROPOUT}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"LR: {LR}\n")
        f.write(f"N_FEATURES_MODEL: {n_features_model}\n")

    # 5. Train
    print("\n── Training ──────────────────────────────────────────────────────")
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
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Save target encoder (EMA-stable — this is the Stage 2 encoder)
    encoder_path = os.path.join(results_dir, 'encoder_weights.weights.h5')
    target_encoder.save_weights(encoder_path)
    print(f"\n  Target encoder weights saved: {encoder_path}")

    # 7. Training curves
    plot_training_curves(history_obj.history,
                         os.path.join(results_dir, 'training_curves.png'))

    print(f"\n  Done. Results in: {results_dir}")
    print(f"  Run H analysis with:")
    print(f"  python -u scripts/replot.py --run_id {args.run_id} --no_age --plots h")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JEPA pre-training — Stage 1 H enrichment')
    parser.add_argument('--data',         type=str, default='data/processed/adults')
    parser.add_argument('--epochs',       type=int, default=70)
    parser.add_argument('--run_id',       type=str, required=True)
    parser.add_argument('--max_patients', type=int, default=None)
    parser.add_argument('--no_age',       action='store_true')
    args = parser.parse_args()
    main(args)
