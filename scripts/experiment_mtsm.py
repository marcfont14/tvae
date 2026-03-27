"""
experiment_mtsm.py
==================
Masked Time Series Modelling (MTSM) — Stage 1 pre-training experiment.

El modelo aprende a reconstruir spans contiguos de CGM enmascarados,
usando el contexto completo de la ventana (pasado + futuro + drivers).
Esto obliga al encoder a aprender las dinámicas reales de la glucosa,
no el atajo de "copia el último valor" que aparece en el forecasting.

Diseño del masking:
  - Solo se enmascara CGM (feature 0). PI, RA y el resto siempre visibles.
  - Spans contiguos de longitud aleatoria entre MASK_MIN_LEN y MASK_MAX_LEN.
  - Se aplican múltiples spans hasta cubrir ~MASK_RATIO de la ventana.
  - Los timesteps enmascarados se reemplazan por un token fijo (0.0).
  - Loss calculado SOLO sobre los timesteps enmascarados.

Nuevas ideas implementadas (activables via CLI):
  --mask_ratio 0.45          Idea 1: enmascaramiento agresivo (default 0.25)
  --mask_max_len 120         Idea 1: spans hasta 10h (default 72 = 6h)
  --shape_loss 0.2           Idea 3: penalización de derivada temporal (default 0.0)
  --multimodal_prob 0.2      Idea 2: enmascarar PI o RA en lugar de CGM en
                                     un 20% de las iteraciones (default 0.0)

Arquitectura:
  Input (288, 11) → [masking de CGM] → Transformer Encoder → H (288, d_model)
  → Reconstruction Head: Dense(1) por timestep → ŷ sobre spans enmascarados
  → Loss vs target real en esos timesteps

El Reconstruction Head se desecha después del pre-training.
En Stage 2 se conecta Attention Pooling + VAE en su lugar.

Outputs (en results/mtsm/{run_id}/):
  training_curves.png             loss train/val por epoch
  reconstruction_examples.png     8 windows ordenadas por variabilidad con ŷ superpuesto
  reconstruction_quality.png      6 paneles: KDE, violin, stats, TIR/TBR/TAR, MAE zone, event
  reconstruction_timeseries.png   2 paneles: distribución Pearson r y DTW por ventana
  run_config.txt                  configuración exacta del run para reproducibilidad

Usage:
  python scripts/experiment_mtsm.py [opciones]

  Argumentos principales:
    --data            Directorio con los .npz procesados
                      (default: data/processed — apunta a data/processed/all
                      o data/processed/adults según el experimento)
    --max_patients    Número máximo de pacientes a cargar (default: todos)
    --epochs          Número máximo de épocas (default: 35, early stopping activo)
    --run_id          Nombre del run para la carpeta de resultados
                      (default: timestamp YYYYMMDD_HHMM)

  Argumentos opcionales (desactivados por defecto):
    --mask_ratio      Fracción de timesteps a enmascarar (default: 0.35)
    --mask_max_len    Span máximo en steps, 1 step = 5min (default: 96 = 8h)
    --shape_loss      Lambda de penalización de derivada temporal (default: 0.0)
    --multimodal_prob Probabilidad de enmascarar PI o RA en lugar de CGM
                      (default: 0.0)
    --no_age          Elimina age_norm (feature 10) del input al encoder.
                      Late fusion: age se pasa solo a los heads downstream.

  Ejemplo:
    python scripts/experiment_mtsm.py \\
        --data data/processed/all --max_patients 150 --run_id run9
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

# ── Config (defaults — todos sobreescribibles por CLI) ────────────────────────

WINDOW_LEN  = 288      # steps por ventana (24h a 5min/step)
N_FEATURES  = 11
CGM_IDX     = 0        # feature 0 = CGM
PI_IDX      = 1        # feature 1 = Plasma Insulin
RA_IDX      = 2        # feature 2 = Rate of Appearance
BOLUS_IDX   = 5        # feature 5 = bolus_logged
CARBS_IDX   = 6        # feature 6 = carbs_logged

# Masking defaults
MASK_RATIO   = 0.35
MASK_MIN_LEN = 60       
MASK_MAX_LEN = 96       
MASK_TOKEN   = 0.0

# Transformer
D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 5
D_FF     = 256
DROPOUT  = 0.2

# Training
BATCH_SIZE = 128
EPOCHS     = 35
LR         = 1e-3
VAL_SPLIT  = 0.1
TEST_SPLIT  = 0.1

# Driver weighting
DRIVER_LOSS_WEIGHT  = 3.0
DRIVER_EFFECT_STEPS = 24   # 2h @ 5min/step

# Nuevas ideas (defaults desactivados)
SHAPE_LOSS_LAMBDA = 0.0    # Idea 3: 0.0 = desactivado
MULTIMODAL_PROB   = 0.0    # Idea 2: 0.0 = desactivado

RESULTS_BASE = 'results/mtsm'
SEED         = 42

COLORS = {
    'cgm_real':  '#111827',
    'recon':     '#2563EB',
    'pi':        '#7C3AED',
    'ra':        '#059669',
    'bolus':     '#DC2626',
    'carbs':     '#D97706',
}

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


# ── Transformer Encoder ───────────────────────────────────────────────────────

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

        ffn = layers.Dense(d_ff, activation='relu', name=f'ffn1_{i}')(x)
        ffn = layers.Dropout(dropout)(ffn)
        ffn = layers.Dense(d_model, name=f'ffn2_{i}')(ffn)
        ffn = layers.Dropout(dropout)(ffn)
        x   = layers.LayerNormalization(epsilon=1e-6, name=f'norm2_{i}')(x + ffn)

    return keras.Model(inp, x, name='TransformerEncoder')


# ── MTSM Model ────────────────────────────────────────────────────────────────

def build_mtsm_model(window_len, n_features, d_model, n_heads, n_layers,
                     d_ff, dropout, vicreg_lambda: float = 0.0):
    inp     = keras.Input(shape=(window_len, n_features), name='input')
    encoder = build_transformer_encoder(
        window_len, n_features, d_model, n_heads, n_layers, d_ff, dropout
    )
    H   = encoder(inp)
    out = layers.Dense(64, activation='relu', name='recon_hidden')(H)  # (batch, 288, 64)
    out = layers.Dense(1, name='reconstruction_head')(out)             # (batch, 288, 1)
    out = layers.Reshape((window_len,), name='output')(out)            # (batch, 288)

    if vicreg_lambda > 0.0:
        # Expose mean-pooled H as a second output for VICReg loss.
        # GlobalAveragePooling1D = mean over time axis — equivalent to reduce_mean(H, axis=1).
        # Use a proper Keras layer (not Lambda) to guarantee GPU graph compilation.
        h_pool = layers.GlobalAveragePooling1D(name='h_pool')(H)
        return keras.Model(inp, [out, h_pool], name='MTSM'), encoder

    return keras.Model(inp, out, name='MTSM'), encoder


# ── Masking ───────────────────────────────────────────────────────────────────

def create_mask(window_len: int, mask_ratio: float,
                min_len: int, max_len: int) -> np.ndarray:
    """Genera mascara binaria con spans contiguos."""
    mask       = np.zeros(window_len, dtype=np.float32)
    target_len = int(window_len * mask_ratio)
    masked_so_far = 0
    attempts = 0
    while masked_so_far < target_len and attempts < 50:
        span_len = np.random.randint(min_len, max_len + 1)
        start    = np.random.randint(0, window_len - span_len)
        mask[start:start + span_len] = 1
        masked_so_far = mask.sum()
        attempts += 1
    return mask



# ── Custom Loss ───────────────────────────────────────────────────────────────

class MaskedMSELoss(keras.losses.Loss):
    """
    MSE ponderado sobre timesteps enmascarados + shape loss opcional.

    y_true: (batch, 288, 3) = [target_real, mask, driver_weight]
    y_pred: (batch, 288)

    Idea 3 — Shape loss (shape_loss_lambda > 0):
      Penaliza diferencias en la derivada primera (pendiente) entre la curva
      real y la reconstruida, dentro de la zona enmascarada. Evita que el
      modelo aprenda trayectorias conservadoras (lineas planas) ignorando la
      volatilidad fisiologica real.

      L_total = L_MSE + lambda * L_shape
      L_shape = mean( (d_pred/dt - d_real/dt)^2 ) sobre timesteps enmascarados
    """
    def __init__(self, shape_loss_lambda: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.shape_loss_lambda = shape_loss_lambda

    def call(self, y_true, y_pred, sample_weight=None):
        target_real   = y_true[:, :, 0]
        mask          = y_true[:, :, 1]
        driver_weight = y_true[:, :, 2]

        # MSE ponderado sobre timesteps enmascarados
        sq_err   = tf.square(target_real - y_pred)
        weighted = sq_err * mask * driver_weight
        n_masked = tf.reduce_sum(mask, axis=1, keepdims=True) + 1e-8
        mse_loss = tf.reduce_mean(
            tf.reduce_sum(weighted, axis=1) / tf.squeeze(n_masked, axis=1)
        )

        if self.shape_loss_lambda > 0.0:
            # Idea 3: derivada primera discreta
            # dy[t] = y[t] - y[t-1]  → shape (batch, 287)
            dy_real = target_real[:, 1:] - target_real[:, :-1]
            dy_pred = y_pred[:, 1:]      - y_pred[:, :-1]

            # Activa solo donde t y t-1 estan ambos enmascarados
            mask_deriv = mask[:, 1:] * mask[:, :-1]
            n_deriv    = tf.reduce_sum(mask_deriv, axis=1, keepdims=True) + 1e-8

            shape_err  = tf.square(dy_pred - dy_real) * mask_deriv
            shape_loss = tf.reduce_mean(
                tf.reduce_sum(shape_err, axis=1) / tf.squeeze(n_deriv, axis=1)
            )
            return mse_loss + self.shape_loss_lambda * shape_loss

        return mse_loss

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'shape_loss_lambda': self.shape_loss_lambda})
        return cfg


class MaskedMAE(keras.metrics.Metric):
    """MAE no ponderado sobre timesteps enmascarados."""
    def __init__(self, name='masked_mae', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        target_real = y_true[:, :, 0]
        mask        = y_true[:, :, 1]
        abs_err     = tf.abs(target_real - y_pred) * mask
        self.total.assign_add(tf.reduce_sum(abs_err))
        self.count.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.total / (self.count + 1e-8)

    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)


class VICRegLoss(keras.losses.Loss):
    """
    VICReg regularisation on mean-pooled H (shape: (batch, d_model)).
    y_true is a dummy zero tensor — loss is entirely self-supervised on y_pred.

    variance_term : Σ_d ReLU(1 − std_d(h))  — keeps all dims active (std ≥ 1)
    covariance_term: Σ_{i≠j} cov(h)²_{ij} / D  — decorrelates dimensions

    Reference: Bardes et al., 2022 — VICReg: Variance-Invariance-Covariance Reg.
    Note: invariance term omitted (no augmentation pairs here).
    """
    def call(self, y_true, y_pred):
        h = y_pred  # (batch, d_model)
        N = tf.cast(tf.shape(h)[0], tf.float32)
        D = tf.cast(tf.shape(h)[1], tf.float32)

        # Variance term — penalise dims with std < 1
        std = tf.sqrt(tf.math.reduce_variance(h, axis=0) + 1e-4)
        var_loss = tf.reduce_mean(tf.nn.relu(1.0 - std))

        # Covariance term — penalise off-diagonal correlations
        # Use tf.eye instead of tf.linalg.diag/diag_part — guaranteed GPU kernel.
        h_c = h - tf.reduce_mean(h, axis=0, keepdims=True)
        cov = tf.matmul(h_c, h_c, transpose_a=True) / (N - 1.0)
        off_diag = 1.0 - tf.eye(tf.shape(h)[1], dtype=h.dtype)
        cov_loss = tf.reduce_sum(tf.square(cov) * off_diag) / D

        return var_loss + cov_loss


class InfoNCELoss:
    """
    InfoNCE contrastive loss on mean-pooled H anchor/positive pairs.
    Not a keras.losses.Loss — computed manually in the custom training step.

    For each anchor h_i and its positive h_i+:
      L = -mean_i[ log( exp(sim(h_i, h_i+)/τ) / Σ_j exp(sim(h_i, h_j)/τ) ) ]
    where j ranges over all positives and negatives in the batch.
    """
    def __init__(self, temperature: float = 0.1):
        self.temperature = temperature

    def __call__(self, h_anchor, h_positive):
        # h_anchor, h_positive: (batch, d_model)
        # Normalise to unit sphere
        h_a = tf.math.l2_normalize(h_anchor,   axis=-1)
        h_p = tf.math.l2_normalize(h_positive, axis=-1)

        # Cosine similarity matrix: (batch, batch)
        logits = tf.matmul(h_a, h_p, transpose_b=True) / self.temperature

        # Positive pairs are on the diagonal
        batch_size = tf.shape(logits)[0]
        labels = tf.range(batch_size)
        loss_a = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        loss_b = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels, tf.transpose(logits)
        )
        return tf.reduce_mean((loss_a + loss_b) / 2.0)


# ── Data preparation ──────────────────────────────────────────────────────────

def apply_masks_vectorized(wins: np.ndarray, mask_ratio: float,
                           mask_min_len: int, mask_max_len: int,
                           multimodal_prob: float,
                           no_logged_events: bool = False,
                           no_age: bool = False):
    """
    Applies masking and driver weighting to N windows at once.
    Driver weight accumulation is fully vectorized over the batch axis.
    Returns (x_masked, Y) with Y = stack([target, mask, driver_weight], axis=-1).

    no_logged_events: if True, zero out bolus_logged and carbs_logged features
        so the model must rely on PI and RA alone to detect driver events.
        Driver loss weighting is still computed from the original windows.
    no_age: if True, drop feature index 10 (age_norm) from the model input.
        Driver weighting and masking use original feature indices (unaffected).
        Late fusion: age should be passed to the downstream head, not the encoder.
    """
    N        = len(wins)
    x_masked = wins.copy()
    if no_logged_events:
        x_masked[:, :, BOLUS_IDX] = 0.0
        x_masked[:, :, CARBS_IDX] = 0.0
    masks    = np.zeros((N, WINDOW_LEN), dtype=np.float32)
    targets  = np.zeros((N, WINDOW_LEN), dtype=np.float32)

    for i in range(N):
        ch = CGM_IDX
        if multimodal_prob > 0 and np.random.random() < multimodal_prob:
            # 2:1 PI over RA — PI is richer signal, RA is sparser
            ch = int(np.random.choice([PI_IDX, RA_IDX], p=[2/3, 1/3]))
        m          = create_mask(WINDOW_LEN, mask_ratio, mask_min_len, mask_max_len)
        targets[i] = wins[i, :, ch]
        masks[i]   = m
        x_masked[i, m.astype(bool), ch] = MASK_TOKEN

    # Vectorized driver weights across all N windows
    bolus        = wins[:, :, BOLUS_IDX]
    carbs        = wins[:, :, CARBS_IDX]
    driver_event = ((bolus + carbs) > 0).astype(np.float32)
    driver_infl  = np.zeros((N, WINDOW_LEN), dtype=np.float32)
    for offset in range(1, DRIVER_EFFECT_STEPS + 1):
        driver_infl[:, offset:] += driver_event[:, :-offset]
    driver_infl   = np.clip(driver_infl, 0, 1)
    driver_weight = np.where(
        (driver_infl > 0) & (masks > 0), DRIVER_LOSS_WEIGHT, 1.0
    ).astype(np.float32)

    # Drop age_norm (feature 10) from model input — late fusion design.
    # Feature indices 0-9 are unchanged, so BOLUS_IDX/CARBS_IDX stay valid above.
    if no_age:
        x_masked = x_masked[:, :, :10]

    Y = np.stack([targets, masks, driver_weight], axis=-1)  # (N, 288, 3)
    return x_masked.astype(np.float32), Y


def make_window_dataset(index: list, shuffle: bool, mask_ratio: float,
                        mask_min_len: int, mask_max_len: int,
                        multimodal_prob: float,
                        no_logged_events: bool = False,
                        no_age: bool = False) -> tf.data.Dataset:
    """
    Builds a tf.data.Dataset from an index of (fpath, win_idx) tuples.

    Groups windows by patient file so each .npz is opened ONCE per epoch
    (not once per window). The generator yields one patient's worth of
    pre-masked windows at a time; tf.data.unbatch() splits them back into
    individual samples before batching to BATCH_SIZE.

    Yields (x_masked, Y) where:
        x_masked: (BATCH_SIZE, 288, 11)
        Y:        (BATCH_SIZE, 288, 3)  = [target_real, mask, driver_weight]
    """
    from collections import defaultdict
    patient_to_windows: dict = defaultdict(list)
    for fpath, win_idx in index:
        patient_to_windows[fpath].append(win_idx)
    fpaths = list(patient_to_windows.keys())

    def generator():
        order = np.random.permutation(len(fpaths)) if shuffle else range(len(fpaths))
        for pi in order:
            fpath       = fpaths[pi]
            win_indices = np.array(patient_to_windows[fpath], dtype=np.int32)
            data        = np.load(fpath, allow_pickle=True)
            wins        = data['windows'][win_indices].astype(np.float32)
            x_masked, Y = apply_masks_vectorized(
                wins, mask_ratio, mask_min_len, mask_max_len, multimodal_prob,
                no_logged_events=no_logged_events, no_age=no_age
            )
            if shuffle:
                perm     = np.random.permutation(len(wins))
                x_masked = x_masked[perm]
                Y        = Y[perm]
            yield x_masked, Y  # (N_patient_windows, 288, 11/3)

    n_features_out = 10 if no_age else N_FEATURES
    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, WINDOW_LEN, n_features_out), dtype=tf.float32),
            tf.TensorSpec(shape=(None, WINDOW_LEN, 3),              dtype=tf.float32),
        )
    )
    ds = ds.unbatch()
    if shuffle:
        ds = ds.shuffle(buffer_size=5_000, seed=SEED)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_paired_window_dataset(index: list, shuffle: bool, mask_ratio: float,
                               mask_min_len: int, mask_max_len: int,
                               multimodal_prob: float,
                               no_logged_events: bool = False,
                               no_age: bool = False) -> tf.data.Dataset:
    """
    Paired dataset for InfoNCE contrastive learning (Run 17).
    Yields ((x_anchor, x_positive), Y_anchor) where anchor and positive are
    consecutive windows from the same patient.
    Windows with only one valid window per patient are skipped.
    """
    from collections import defaultdict
    patient_to_windows: dict = defaultdict(list)
    for fpath, win_idx in index:
        patient_to_windows[fpath].append(win_idx)
    # Keep only patients with ≥ 2 windows so pairs exist
    fpaths = [fp for fp, idxs in patient_to_windows.items() if len(idxs) >= 2]

    def generator():
        order = np.random.permutation(len(fpaths)) if shuffle else range(len(fpaths))
        for pi in order:
            fpath       = fpaths[pi]
            win_indices = np.array(sorted(patient_to_windows[fpath]), dtype=np.int32)
            data        = np.load(fpath, allow_pickle=True)
            wins        = data['windows'][win_indices].astype(np.float32)

            # Build consecutive pairs: (i, i+1)
            anchors   = wins[:-1]
            positives = wins[1:]

            x_a, Y_a = apply_masks_vectorized(
                anchors, mask_ratio, mask_min_len, mask_max_len, multimodal_prob,
                no_logged_events=no_logged_events, no_age=no_age
            )
            x_p, _   = apply_masks_vectorized(
                positives, mask_ratio, mask_min_len, mask_max_len, multimodal_prob,
                no_logged_events=no_logged_events, no_age=no_age
            )
            if shuffle:
                perm = np.random.permutation(len(x_a))
                x_a, x_p, Y_a = x_a[perm], x_p[perm], Y_a[perm]
            yield (x_a, x_p), Y_a

    n_features_out = 10 if no_age else N_FEATURES
    feat_spec = tf.TensorSpec(shape=(None, WINDOW_LEN, n_features_out), dtype=tf.float32)
    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            (feat_spec, feat_spec),
            tf.TensorSpec(shape=(None, WINDOW_LEN, 3), dtype=tf.float32),
        )
    )
    ds = ds.unbatch()
    if shuffle:
        ds = ds.shuffle(buffer_size=5_000, seed=SEED)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ── Plot: training curves ─────────────────────────────────────────────────────

def plot_training_curves(history, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('MTSM Training Curves — Masked Reconstruction',
                 fontsize=14, fontweight='bold')

    # Multi-output models (VICReg) name the metric 'output_masked_mae'
    mae_key = 'output_masked_mae' if 'output_masked_mae' in history else 'masked_mae'

    for ax, (metric, label) in zip(axes, [('loss', 'Masked MSE Loss (+ shape)'),
                                           (mae_key, 'Masked MAE')]):
        ax.plot(history[metric],
                color='#2563EB', lw=2, ls='-', label='train')
        ax.plot(history[f'val_{metric}'],
                color='#2563EB', lw=1.5, ls='--', label='val')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {save_path}")


# ── Plot: reconstruction examples ─────────────────────────────────────────────

def plot_reconstruction_examples(model, windows_orig: np.ndarray,
                                  masks: np.ndarray,
                                  mask_ratio: float, mask_min_len: int,
                                  mask_max_len: int, save_path: str):
    N = len(windows_orig)
    cgm   = windows_orig[:, :, CGM_IDX]
    pi    = windows_orig[:, :, PI_IDX]
    ra    = windows_orig[:, :, RA_IDX]
    bolus = windows_orig[:, :, BOLUS_IDX]
    carbs = windows_orig[:, :, CARBS_IDX]

    cgm_range_in_mask = []
    for i in range(N):
        m = masks[i].astype(bool)
        cgm_range_in_mask.append(
            cgm[i][m].max() - cgm[i][m].min() if m.sum() > 0 else 0.0
        )
    cgm_range_in_mask = np.array(cgm_range_in_mask)

    pct_low  = np.percentile(cgm_range_in_mask, 20)
    pct_high = np.percentile(cgm_range_in_mask, 80)

    flat_candidates    = np.where(cgm_range_in_mask <= pct_low)[0]
    dynamic_candidates = np.where(cgm_range_in_mask >= pct_high)[0]
    np.random.shuffle(flat_candidates)
    np.random.shuffle(dynamic_candidates)

    selected = list(flat_candidates[:2]) + list(dynamic_candidates[:6])
    labels   = ['Basal (flat)', 'Basal (flat)',
            'Dynamic (peak)', 'Dynamic (peak)',
            'Dynamic (peak)', 'Dynamic (peak)',
            'Dynamic (peak)', 'Dynamic (peak)']

    x_masked_sel = windows_orig[selected].copy()
    for j, idx in enumerate(selected):
        m = masks[idx].astype(bool)
        x_masked_sel[j, m, CGM_IDX] = MASK_TOKEN

    y_pred = model.predict(x_masked_sel, verbose=0)

    fig = plt.figure(figsize=(18, 5 * 8))
    gs  = gridspec.GridSpec(8, 1, figure=fig, hspace=0.6)
    t   = np.arange(WINDOW_LEN)

    for row, (idx, label) in enumerate(zip(selected, labels)):
        ax     = fig.add_subplot(gs[row])
        ax_drv = ax.twinx()

        m = masks[idx].astype(bool)

        mask_starts = np.where(np.diff(m.astype(int)) == 1)[0] + 1
        mask_ends   = np.where(np.diff(m.astype(int)) == -1)[0] + 1
        if m[0]:  mask_starts = np.insert(mask_starts, 0, 0)
        if m[-1]: mask_ends   = np.append(mask_ends, WINDOW_LEN)
        for s, e in zip(mask_starts, mask_ends):
            ax.axvspan(s, e, alpha=0.12, color='#9CA3AF')

        ax.plot(t, cgm[idx], color=COLORS['cgm_real'], lw=2,
                label='CGM real', zorder=5)

        recon = np.full(WINDOW_LEN, np.nan)
        recon[m] = y_pred[row][m]
        ax.plot(t, recon, color=COLORS['recon'], lw=2.5, ls='--',
                label='Reconstruccion', zorder=6)

        ax_drv.plot(t, pi[idx], color=COLORS['pi'], lw=1.2, alpha=0.7, label='PI')
        ax_drv.plot(t, ra[idx], color=COLORS['ra'], lw=1.2, alpha=0.7, label='RA')

        bolus_t = t[bolus[idx] > 0]
        carbs_t = t[carbs[idx] > 0]
        cgm_min = cgm[idx].min()
        if len(bolus_t) > 0:
            ax.scatter(bolus_t, [cgm_min - 0.15] * len(bolus_t),
                       marker='^', color=COLORS['bolus'], s=40, zorder=7,
                       label='Bolus event')
        if len(carbs_t) > 0:
            ax.scatter(carbs_t, [cgm_min - 0.3] * len(carbs_t),
                       marker='^', color=COLORS['carbs'], s=40, zorder=7,
                       label='Carbs event')

        ax.set_ylabel('CGM (z-score)', fontsize=9)
        ax_drv.set_ylabel('PI / RA (z-score)', fontsize=9, color='#6B7280')
        ax.set_xlabel('Timestep (5 min)', fontsize=9)

        n_masked = m.sum()
        ax.set_title(
            f'{label}  —  window idx={idx}  '
            f'({n_masked} steps = {n_masked*5} min enmascarados  |  '
            f'dCGM zona: {cgm_range_in_mask[idx]:.2f} z-score)',
            fontsize=10
        )

        all_lines = ([l for l in ax.get_lines() if not l.get_label().startswith('_')]
                   + [l for l in ax_drv.get_lines() if not l.get_label().startswith('_')])
        ax.legend(all_lines, [l.get_label() for l in all_lines],
                  fontsize=8, loc='upper right', ncol=3)
        ax.grid(True, alpha=0.2)
        ax.spines[['top']].set_visible(False)
        ax_drv.spines[['top']].set_visible(False)

        step_t = 48
        ticks  = list(range(0, WINDOW_LEN + 1, step_t))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{t*5//60}h' for t in ticks], fontsize=8)

    fig.suptitle(
        f'MTSM Reconstruction Examples  '
        f'(mask_ratio={mask_ratio:.0%}, '
        f'spans={mask_min_len*5//60}h-{mask_max_len*5//60}h)\n'
        f'Gris = zona enmascarada  |  Azul = reconstruccion  |  '
        f'PI y RA siempre visibles',
        fontsize=13, fontweight='bold'
    )
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {save_path}")


# ── Load data ─────────────────────────────────────────────────────────────────

def index_dataset(processed_dir: str, max_patients: int = None) -> list:
    """
    Builds an index of (filepath, window_idx) tuples without loading data into RAM.
    Applies quality filters per patient: has_driver and cgm_std in valid range.
    
    Returns:
        List of (filepath, window_idx) tuples — one entry per valid window.
    """
    npz_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('.npz')])
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {processed_dir}")
    if max_patients is not None:
        npz_files = npz_files[:max_patients]
        print(f"  (--max_patients {max_patients}: loading subset)")

    index        = []
    n_before     = 0
    n_filtered   = 0

    for fname in npz_files:
        fpath = os.path.join(processed_dir, fname)
        data  = np.load(fpath, allow_pickle=True)
        wins  = data['windows'].astype(np.float32)  # (N_windows, 288, 11)

        bolus   = wins[:, :, BOLUS_IDX]
        carbs   = wins[:, :, CARBS_IDX]
        cgm     = wins[:, :, CGM_IDX]

        has_driver = ((bolus + carbs) > 0).any(axis=1)
        cgm_std    = cgm.std(axis=1)
        cgm_ok     = (cgm_std > 0.3) & (cgm_std < 4.0)
        keep       = has_driver & cgm_ok

        n_before   += len(wins)
        n_filtered += (~keep).sum()

        for i in np.where(keep)[0]:
            index.append((fpath, int(i)))

    pct = n_filtered / n_before * 100 if n_before > 0 else 0
    print(f"  Filtered: {n_filtered:,} pathological windows "
          f"({pct:.1f}%)  →  {len(index):,} remaining")
    print(f"  Patients: {len(npz_files)}   Windows: {len(index):,}")
    return index



def load_windows_from_index(index_sample: list) -> np.ndarray:
    """Loads specific windows by (filepath, window_idx) from disk into RAM."""
    windows = []
    for fpath, win_idx in index_sample:
        data = np.load(fpath, allow_pickle=True)
        windows.append(data['windows'][win_idx].astype(np.float32))
    return np.stack(windows, axis=0)

# ── Save run config ───────────────────────────────────────────────────────────

def save_run_config(results_dir: str, args):
    config_path = os.path.join(results_dir, 'run_config.txt')
    with open(config_path, 'w') as f:
        f.write("MTSM Run Configuration\n")
        f.write("=" * 40 + "\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write("\nFixed hyperparameters:\n")
        f.write(f"D_MODEL: {D_MODEL}\n")
        f.write(f"N_HEADS: {N_HEADS}\n")
        f.write(f"N_LAYERS: {N_LAYERS}\n")
        f.write(f"D_FF: {D_FF}\n")
        f.write(f"DROPOUT: {DROPOUT}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"LR: {LR}\n")
        f.write(f"DRIVER_LOSS_WEIGHT: {DRIVER_LOSS_WEIGHT}\n")
        f.write(f"DRIVER_EFFECT_STEPS: {DRIVER_EFFECT_STEPS}\n")
        f.write(f"N_FEATURES_MODEL: {10 if getattr(args, 'no_age', False) else N_FEATURES}\n")
    print(f"  Config guardada: {config_path}")


# ── Load scalers ───────────────────────────────────────────────────────────────

def load_scalers_from_index(index_sample: list) -> tuple:
    """Returns (mean_cgm, std_cgm) float32 arrays of shape (N,) for inverse-transform."""
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


# ── Plot: reconstruction quality (6-panel) ────────────────────────────────────

def plot_reconstruction_quality(model, windows_orig: np.ndarray,
                                 masks: np.ndarray,
                                 scaler_mean_cgm: np.ndarray,
                                 scaler_std_cgm: np.ndarray,
                                 save_path: str):
    """
    6-panel figure comparing real vs reconstructed CGM on masked spans:
      (0,0) KDE — probability density of values
      (0,1) Violin — distribution shape with IQR
      (0,2) Summary stats table — mean, std, median, IQR, skewness
      (1,0) TIR/TBR/TAR — glycaemic ranges in mg/dL
      (1,1) MAE by metabolic zone — postprandial / post-bolus / basal
      (1,2) Event response — mean CGM trajectory around bolus events
    """
    from scipy.stats import gaussian_kde, skew as scipy_skew

    print("  Computing reconstruction quality metrics...")

    # Predict
    x_masked = windows_orig.copy()
    for i in range(len(windows_orig)):
        m = masks[i].astype(bool)
        x_masked[i, m, CGM_IDX] = MASK_TOKEN
    y_pred = model.predict(x_masked, batch_size=BATCH_SIZE, verbose=0)  # (N, 288)

    cgm_real = windows_orig[:, :, CGM_IDX]   # (N, 288), z-score
    m_bool   = masks.astype(bool)             # (N, 288)

    # Masked values only (z-score)
    real_masked = cgm_real[m_bool]
    pred_masked = y_pred[m_bool]

    # Inverse transform to mg/dL for TIR
    sm = scaler_mean_cgm[:, np.newaxis]   # (N, 1)
    ss = scaler_std_cgm[:, np.newaxis]    # (N, 1)
    real_masked_mgdl = (cgm_real * ss + sm)[m_bool]
    pred_masked_mgdl = (y_pred   * ss + sm)[m_bool]

    def tir_stats(vals):
        tir = ((vals >= 70) & (vals <= 180)).mean() * 100
        tbr = (vals < 70).mean() * 100
        tar = (vals > 180).mean() * 100
        return tir, tbr, tar

    tir_r, tbr_r, tar_r = tir_stats(real_masked_mgdl)
    tir_p, tbr_p, tar_p = tir_stats(pred_masked_mgdl)

    # Driver zone MAE (z-score)
    bolus = windows_orig[:, :, BOLUS_IDX]
    carbs = windows_orig[:, :, CARBS_IDX]
    carbs_infl = np.zeros_like(carbs)
    bolus_infl = np.zeros_like(bolus)
    for offset in range(1, DRIVER_EFFECT_STEPS + 1):
        carbs_infl[:, offset:] += (carbs[:, :-offset] > 0).astype(np.float32)
        bolus_infl[:, offset:] += (bolus[:, :-offset] > 0).astype(np.float32)
    carbs_infl = np.clip(carbs_infl, 0, 1)
    bolus_infl = np.clip(bolus_infl, 0, 1)

    abs_err           = np.abs(cgm_real - y_pred) * masks
    pp_mask           = (carbs_infl > 0) & m_bool
    pb_mask           = (bolus_infl > 0) & m_bool & (carbs_infl == 0)
    bas_mask          = m_bool & (carbs_infl == 0) & (bolus_infl == 0)

    def safe_mae(err, zone):
        n = zone.sum()
        return (err * zone).sum() / n if n > 0 else 0.0

    mae_pp  = safe_mae(abs_err, pp_mask)
    mae_pb  = safe_mae(abs_err, pb_mask)
    mae_bas = safe_mae(abs_err, bas_mask)

    # Event response: CGM ±50 steps around bolus events
    WIN = 50
    real_traces, pred_traces = [], []
    bolus_events = (bolus > 0)
    for i in range(len(windows_orig)):
        for t in np.where(bolus_events[i])[0]:
            if t - WIN >= 0 and t + WIN + 1 <= WINDOW_LEN:
                real_traces.append(cgm_real[i, t - WIN:t + WIN + 1])
                pred_traces.append(y_pred[i,   t - WIN:t + WIN + 1])

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Reconstruction Quality — Masked CGM Spans', fontsize=14, fontweight='bold')

    # (0,0) KDE
    ax = axes[0, 0]
    lo = min(real_masked.min(), pred_masked.min())
    hi = max(real_masked.max(), pred_masked.max())
    xs = np.linspace(lo, hi, 300)
    kde_r = gaussian_kde(real_masked)
    kde_p = gaussian_kde(pred_masked)
    ax.fill_between(xs, kde_r(xs), alpha=0.3, color='#2563EB')
    ax.plot(xs, kde_r(xs), color='#2563EB', lw=2, label='Real')
    ax.fill_between(xs, kde_p(xs), alpha=0.3, color='#DC2626')
    ax.plot(xs, kde_p(xs), color='#DC2626', lw=2, label='Reconstructed')
    ax.set_xlabel('CGM (z-score)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Value Distribution (masked timesteps)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    # (0,1) Violin
    ax = axes[0, 1]
    parts = ax.violinplot([real_masked, pred_masked], positions=[1, 2],
                           showmedians=True, showextrema=True)
    for pc, col in zip(parts['bodies'], ['#2563EB', '#DC2626']):
        pc.set_facecolor(col); pc.set_alpha(0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Real', 'Reconstructed'], fontsize=10)
    ax.set_ylabel('CGM (z-score)', fontsize=10)
    ax.set_title('Distribution Shape (masked timesteps)', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines[['top', 'right']].set_visible(False)

    # (0,2) Stats table
    ax = axes[0, 2]
    ax.axis('off')
    def stats_row(vals):
        q25, q75 = np.percentile(vals, [25, 75])
        return [f'{vals.mean():.3f}', f'{vals.std():.3f}',
                f'{np.median(vals):.3f}', f'{q75 - q25:.3f}',
                f'{scipy_skew(vals):.3f}']
    tbl = ax.table(
        cellText=[stats_row(real_masked), stats_row(pred_masked)],
        colLabels=['Mean', 'Std', 'Median', 'IQR', 'Skewness'],
        rowLabels=['Real', 'Reconstructed'],
        cellLoc='center', loc='center', bbox=[0, 0.15, 1, 0.65]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    ax.set_title('Summary Statistics (z-score)', fontsize=11)

    # (1,0) TIR/TBR/TAR
    ax = axes[1, 0]
    categories = ['TIR\n(70–180 mg/dL)', 'TBR\n(<70 mg/dL)', 'TAR\n(>180 mg/dL)']
    real_vals  = [tir_r, tbr_r, tar_r]
    pred_vals  = [tir_p, tbr_p, tar_p]
    x = np.arange(len(categories))
    w = 0.35
    bars_r = ax.bar(x - w/2, real_vals, w, label='Real',          color='#2563EB', alpha=0.8)
    bars_p = ax.bar(x + w/2, pred_vals, w, label='Reconstructed', color='#DC2626', alpha=0.8)
    for bar in list(bars_r) + list(bars_p):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel('% of masked timesteps', fontsize=10)
    ax.set_title('Glycaemic Ranges (masked timesteps)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines[['top', 'right']].set_visible(False)

    # (1,1) MAE by metabolic zone
    ax = axes[1, 1]
    zones  = ['Postprandial\n(0–2h after carbs)',
              'Post-bolus\n(0–2h after bolus)',
              'Basal\n(no driver)']
    values = [mae_pp, mae_pb, mae_bas]
    colors_z = [COLORS['carbs'], COLORS['bolus'], '#6B7280']
    bars = ax.bar(zones, values, color=colors_z, alpha=0.8, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('MAE (z-score)', fontsize=10)
    ax.set_title('Reconstruction Error by Metabolic Zone', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines[['top', 'right']].set_visible(False)

    # (1,2) Event response
    ax = axes[1, 2]
    if real_traces:
        real_arr = np.array(real_traces)
        pred_arr = np.array(pred_traces)
        t_ax     = np.arange(-WIN, WIN + 1) * 5   # minutes
        r_m, r_s = real_arr.mean(0), real_arr.std(0)
        p_m, p_s = pred_arr.mean(0), pred_arr.std(0)
        ax.plot(t_ax, r_m, color='#2563EB', lw=2, label='Real')
        ax.fill_between(t_ax, r_m - r_s, r_m + r_s, alpha=0.2, color='#2563EB')
        ax.plot(t_ax, p_m, color='#DC2626', lw=2, label='Reconstructed')
        ax.fill_between(t_ax, p_m - p_s, p_m + p_s, alpha=0.2, color='#DC2626')
        ax.axvline(0, color='black', ls='--', lw=1, alpha=0.5)
        ax.set_title(f'Avg CGM Around Bolus Events  (n={len(real_traces)})', fontsize=11)
    ax.set_xlabel('Time from bolus (min)', fontsize=10)
    ax.set_ylabel('CGM (z-score)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  MAE — Postprandial: {mae_pp:.3f}  Post-bolus: {mae_pb:.3f}  Basal: {mae_bas:.3f}")
    print(f"  TIR  Real: {tir_r:.1f}%  Recon: {tir_p:.1f}%  |  "
          f"TBR  Real: {tbr_r:.1f}%  Recon: {tbr_p:.1f}%")
    print(f"  Saved: {save_path}")


# ── Plot: reconstruction timeseries (2-panel) ─────────────────────────────────

def plot_reconstruction_timeseries(model, windows_orig: np.ndarray,
                                    masks: np.ndarray,
                                    save_path: str):
    """
    2-panel figure — per-window shape metrics on masked spans:
      Left  — distribution of per-window Pearson r (real vs reconstructed)
      Right — distribution of per-window DTW distance (real vs reconstructed)
    """
    from scipy.stats import pearsonr, gaussian_kde
    from dtaidistance import dtw as dtw_lib

    print("  Computing per-window Pearson r and DTW...")

    x_masked = windows_orig.copy()
    for i in range(len(windows_orig)):
        m = masks[i].astype(bool)
        x_masked[i, m, CGM_IDX] = MASK_TOKEN
    y_pred   = model.predict(x_masked, batch_size=BATCH_SIZE, verbose=0)
    cgm_real = windows_orig[:, :, CGM_IDX]

    r_vals, dtw_vals = [], []
    for i in range(len(windows_orig)):
        m  = masks[i].astype(bool)
        yr = cgm_real[i, m]
        yp = y_pred[i, m]
        if len(yr) > 3:
            r, _ = pearsonr(yr, yp)
            r_vals.append(float(r))
            try:
                d = dtw_lib.distance_fast(yr.astype(np.double), yp.astype(np.double))
                dtw_vals.append(float(d))
            except Exception:
                dtw_vals.append(float(np.nan))

    r_vals   = np.array(r_vals)
    dtw_vals = np.array(dtw_vals)
    dtw_ok   = dtw_vals[np.isfinite(dtw_vals)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Reconstruction Quality — Per-Window Shape Metrics', fontsize=14, fontweight='bold')

    # Pearson r distribution
    ax = axes[0]
    xs = np.linspace(max(r_vals.min() - 0.05, -1.0), min(r_vals.max() + 0.05, 1.0), 200)
    kde = gaussian_kde(r_vals)
    ax.fill_between(xs, kde(xs), alpha=0.4, color='#2563EB')
    ax.plot(xs, kde(xs), color='#2563EB', lw=2)
    ax.axvline(np.median(r_vals), color='#DC2626', ls='--', lw=1.5,
               label=f'Median r = {np.median(r_vals):.3f}')
    ax.set_xlabel('Pearson r  (real vs reconstructed, masked region)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'Per-Window Correlation  (n = {len(r_vals)} windows)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    # DTW distribution
    ax = axes[1]
    if len(dtw_ok) > 1:
        hi = np.percentile(dtw_ok, 99)
        xs2 = np.linspace(dtw_ok.min(), hi, 200)
        kde2 = gaussian_kde(dtw_ok)
        ax.fill_between(xs2, kde2(xs2), alpha=0.4, color='#059669')
        ax.plot(xs2, kde2(xs2), color='#059669', lw=2)
        ax.axvline(np.median(dtw_ok), color='#DC2626', ls='--', lw=1.5,
                   label=f'Median DTW = {np.median(dtw_ok):.3f}')
        ax.set_title(f'Per-Window DTW Distance  (n = {len(dtw_ok)} windows)', fontsize=11)
    ax.set_xlabel('DTW distance  (z-score, masked region)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Pearson r  — median: {np.median(r_vals):.3f}  mean: {r_vals.mean():.3f}")
    if len(dtw_ok) > 0:
        print(f"  DTW        — median: {np.median(dtw_ok):.3f}  mean: {dtw_ok.mean():.3f}")
    print(f"  Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    from datetime import datetime

    # ── Override architecture globals from CLI (Run 18 — scale up) ───────────
    global D_MODEL, N_HEADS, D_FF
    D_MODEL = args.d_model
    N_HEADS = args.n_heads
    D_FF    = args.d_ff

    run_id      = args.run_id if args.run_id else datetime.now().strftime('%Y%m%d_%H%M')
    results_dir = os.path.join(RESULTS_BASE, run_id)
    os.makedirs(results_dir, exist_ok=True)

    vicreg      = args.vicreg_lambda > 0.0
    contrastive = args.contrastive_lambda > 0.0

    print(f"\n{'='*52}")
    print(f"  MTSM — Stage 1 Pre-training")
    print(f"  Run ID:          {run_id}")
    print(f"  mask_ratio:      {args.mask_ratio:.0%}")
    print(f"  mask_max_len:    {args.mask_max_len} steps = {args.mask_max_len*5//60}h")
    print(f"  shape_loss L:    {args.shape_loss}")
    print(f"  multimodal_prob: {args.multimodal_prob:.0%}")
    print(f"  vicreg_lambda:   {args.vicreg_lambda}")
    print(f"  contrastive_λ:   {args.contrastive_lambda}")
    print(f"  d_model/heads/ff:{D_MODEL}/{N_HEADS}/{D_FF}")
    print(f"{'='*52}")

    save_run_config(results_dir, args)

    # 1. Index dataset
    print(f"\n  Indexing dataset from: {args.data}")
    index = index_dataset(args.data, args.max_patients)

    # 2. Patient-level split
    all_fpaths = sorted(list(set(fp for fp, _ in index)))
    n          = len(all_fpaths)
    perm       = np.random.permutation(n)
    n_test     = int(n * TEST_SPLIT)
    n_val      = int(n * VAL_SPLIT)

    test_set  = set(all_fpaths[i] for i in perm[:n_test])
    val_set   = set(all_fpaths[i] for i in perm[n_test:n_test + n_val])
    train_set = set(all_fpaths[i] for i in perm[n_test + n_val:])

    train_index = [(fp, wi) for fp, wi in index if fp in train_set]
    val_index   = [(fp, wi) for fp, wi in index if fp in val_set]
    test_index  = [(fp, wi) for fp, wi in index if fp in test_set]

    print(f"  Patient split — Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")
    print(f"  Window split  — Train: {len(train_index):,}  Val: {len(val_index):,}  Test: {len(test_index):,}")

    # 3. Build tf.data pipelines
    ds_kwargs = dict(
        mask_ratio=args.mask_ratio, mask_min_len=MASK_MIN_LEN,
        mask_max_len=args.mask_max_len, multimodal_prob=args.multimodal_prob,
        no_logged_events=args.no_logged_events, no_age=args.no_age
    )

    if contrastive:
        # Paired dataset for InfoNCE — yields ((x_anchor, x_positive), Y_anchor)
        train_ds = make_paired_window_dataset(train_index, shuffle=True,  **ds_kwargs)
        val_ds   = make_paired_window_dataset(val_index,   shuffle=False, **ds_kwargs)
    else:
        train_ds = make_window_dataset(train_index, shuffle=True,  **ds_kwargs)
        val_ds   = make_window_dataset(val_index,   shuffle=False, **ds_kwargs)

    # 4. Build model
    n_features_model = 10 if args.no_age else N_FEATURES
    model, encoder = build_mtsm_model(
        WINDOW_LEN, n_features_model, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT,
        vicreg_lambda=args.vicreg_lambda
    )

    if vicreg:
        # Multi-output model: add dummy zero target for h_pool output
        def add_vicreg_dummy(x, y):
            batch_size = tf.shape(x)[0]
            return x, (y, tf.zeros((batch_size, D_MODEL)))
        train_ds = train_ds.map(add_vicreg_dummy)
        val_ds   = val_ds.map(add_vicreg_dummy)
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=LR, weight_decay=1e-4),
            loss=[MaskedMSELoss(shape_loss_lambda=args.shape_loss), VICRegLoss()],
            loss_weights=[1.0, args.vicreg_lambda],
            metrics={'output': MaskedMAE()}
        )
    elif contrastive:
        # Paired inputs: model sees x_anchor only for reconstruction.
        # InfoNCE computed manually in a custom train loop via GradientTape.
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=LR, weight_decay=1e-4),
            loss=MaskedMSELoss(shape_loss_lambda=args.shape_loss),
            metrics=[MaskedMAE()]
        )
    else:
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=LR, weight_decay=1e-4),
            loss=MaskedMSELoss(shape_loss_lambda=args.shape_loss),
            metrics=[MaskedMAE()]
        )

    model.summary()

    # 5. Train
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

    if contrastive:
        # Custom training loop: MTSM loss + InfoNCE on mean-pooled H pairs
        infonce     = InfoNCELoss(temperature=0.1)
        optimizer   = keras.optimizers.AdamW(learning_rate=LR, weight_decay=1e-4)
        mtsm_loss_fn = MaskedMSELoss(shape_loss_lambda=args.shape_loss)
        mae_metric  = MaskedMAE()
        history = {'loss': [], 'masked_mae': [], 'val_loss': [], 'val_masked_mae': []}

        best_val_loss  = float('inf')
        patience_count = 0
        best_weights   = None

        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            # ── Training ────────────────────────────────────────────────────
            train_loss_sum, train_steps = 0.0, 0
            mae_metric.reset_state()
            for (x_a, x_p), y_a in train_ds:
                with tf.GradientTape() as tape:
                    # Reconstruction loss on anchor
                    recon_a  = model(x_a, training=True)
                    loss_mtsm = mtsm_loss_fn(y_a, recon_a)
                    # InfoNCE on mean-pooled H pairs
                    H_a = encoder(x_a, training=True)
                    H_p = encoder(x_p, training=True)
                    h_a = tf.reduce_mean(H_a, axis=1)
                    h_p = tf.reduce_mean(H_p, axis=1)
                    loss_info = infonce(h_a, h_p)
                    loss = loss_mtsm + args.contrastive_lambda * loss_info
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                mae_metric.update_state(y_a, recon_a)
                train_loss_sum += float(loss_mtsm); train_steps += 1
            history['loss'].append(train_loss_sum / train_steps)
            history['masked_mae'].append(float(mae_metric.result()))

            # ── Validation ──────────────────────────────────────────────────
            val_loss_sum, val_steps = 0.0, 0
            mae_metric.reset_state()
            for (x_a, x_p), y_a in val_ds:
                recon_a = model(x_a, training=False)
                val_loss_sum += float(mtsm_loss_fn(y_a, recon_a)); val_steps += 1
                mae_metric.update_state(y_a, recon_a)
            val_loss = val_loss_sum / val_steps
            history['val_loss'].append(val_loss)
            history['val_masked_mae'].append(float(mae_metric.result()))

            print(f"  loss={history['loss'][-1]:.4f}  mae={history['masked_mae'][-1]:.4f}"
                  f"  val_loss={val_loss:.4f}  val_mae={history['val_masked_mae'][-1]:.4f}")

            # Manual early stopping + LR reduction
            if val_loss < best_val_loss - 1e-4:
                best_val_loss  = val_loss
                patience_count = 0
                best_weights   = model.get_weights()
            else:
                patience_count += 1
                if patience_count >= 5:
                    new_lr = max(float(optimizer.learning_rate) * 0.5, 1e-6)
                    optimizer.learning_rate.assign(new_lr)
                    print(f"  LR reduced to {new_lr:.2e}")
                    patience_count = 0
                if patience_count >= 10:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        if best_weights is not None:
            model.set_weights(best_weights)
    else:
        history_obj = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1
        )
        history = history_obj.history

    # 6. Save weights — encoder (for Stage 2) + full model (for replotting)
    encoder_path = os.path.join(results_dir, 'encoder_weights.weights.h5')
    encoder.save_weights(encoder_path)
    model_path = os.path.join(results_dir, 'model_weights.weights.h5')
    if vicreg:
        pred_model_to_save = keras.Model(model.input, model.output[0])
    else:
        pred_model_to_save = model
    pred_model_to_save.save_weights(model_path)
    print(f"\n  Encoder weights saved: {encoder_path}")
    print(f"  Full model weights saved: {model_path}")

    # 7. Load a bounded test sample for plots (500 windows max)
    N_PLOT = 500
    if len(test_index) > N_PLOT:
        rng_idx    = np.random.choice(len(test_index), N_PLOT, replace=False)
        plot_index = [test_index[i] for i in rng_idx]
    else:
        plot_index = test_index

    print(f"\n  Loading {len(plot_index)} test windows for plots...")
    test_windows = load_windows_from_index(plot_index)
    print(f"  Test sample shape: {test_windows.shape}")
    if args.no_age:
        test_windows = test_windows[:, :, :10]

    # For VICReg multi-output model, build a single-output pred_model for plots
    if vicreg:
        pred_model = keras.Model(model.input, model.output[0])
    else:
        pred_model = model

    # 8. Plots
    print(f"\n{'='*52}")
    print(f"  Generating plots")
    print(f"{'='*52}")

    masks_test = np.stack([
        create_mask(WINDOW_LEN, args.mask_ratio, MASK_MIN_LEN, args.mask_max_len)
        for _ in range(len(test_windows))
    ])

    # Load per-window scalers for inverse-transform (CGM z-score → mg/dL)
    print(f"\n  Loading scalers for {len(plot_index)} test windows...")
    scaler_mean_cgm, scaler_std_cgm = load_scalers_from_index(plot_index)

    plot_training_curves(
        history,
        os.path.join(results_dir, 'training_curves.png')
    )

    plot_reconstruction_examples(
        pred_model, test_windows, masks_test,
        mask_ratio=args.mask_ratio,
        mask_min_len=MASK_MIN_LEN,
        mask_max_len=args.mask_max_len,
        save_path=os.path.join(results_dir, 'reconstruction_examples.png')
    )

    plot_reconstruction_quality(
        pred_model, test_windows, masks_test,
        scaler_mean_cgm, scaler_std_cgm,
        save_path=os.path.join(results_dir, 'reconstruction_quality.png')
    )

    plot_reconstruction_timeseries(
        pred_model, test_windows, masks_test,
        save_path=os.path.join(results_dir, 'reconstruction_timeseries.png')
    )

    print(f"\n{'='*52}")
    print(f"  Done. Results in: {results_dir}/")
    print(f"{'='*52}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTSM Stage 1 pre-training')
    parser.add_argument('--data',            type=str,   default='data/processed')
    parser.add_argument('--epochs',          type=int,   default=EPOCHS)
    parser.add_argument('--max_patients',    type=int,   default=None)
    parser.add_argument('--run_id',          type=str,   default=None,
                        help='ID del run (default: timestamp YYYYMMDD_HHMM)')
    # Idea 1: enmascaramiento agresivo
    parser.add_argument('--mask_ratio',      type=float, default=MASK_RATIO,
                        help='Fraccion de timesteps a enmascarar '
                             '(default 0.25 | Idea 1: 0.45)')
    parser.add_argument('--mask_max_len',    type=int,   default=MASK_MAX_LEN,
                        help='Span maximo en steps '
                             '(default 72=6h | Idea 1: 120=10h)')
    # Idea 3: shape loss
    parser.add_argument('--shape_loss',      type=float, default=SHAPE_LOSS_LAMBDA,
                        help='Lambda shape loss derivada temporal '
                             '(default 0.0=off | Idea 3: 0.2)')
    # Idea 2: multimodal masking
    parser.add_argument('--multimodal_prob', type=float, default=MULTIMODAL_PROB,
                        help='Prob de enmascarar PI o RA en lugar de CGM '
                             '(default 0.0=off | Idea 2: 0.2)')
    # Ablation: remove discrete logged event features
    parser.add_argument('--no_logged_events', action='store_true', default=False,
                        help='Zero out bolus_logged and carbs_logged features — '
                             'forces model to rely on PI and RA only')
    # Late fusion: remove age_norm from encoder input
    parser.add_argument('--no_age', action='store_true', default=False,
                        help='Drop age_norm (feature 10) from encoder input. '
                             'Late fusion design: age passed to downstream heads only, '
                             'not to the encoder, to avoid demographic shortcuts.')
    # Run 15: VICReg regularisation on mean-pooled H
    parser.add_argument('--vicreg_lambda', type=float, default=0.0,
                        help='Weight of VICReg loss on mean-pooled H (default 0.0=off). '
                             'Penalises dimension collapse and inter-dim covariance. '
                             'Recommended: 0.05')
    # Run 17: InfoNCE contrastive loss on consecutive same-patient window pairs
    parser.add_argument('--contrastive_lambda', type=float, default=0.0,
                        help='Weight of InfoNCE loss on mean-pooled H pairs (default 0.0=off). '
                             'Positive pairs = consecutive windows from same patient. '
                             'Recommended: 0.1')
    # Run 18: architecture scale-up flags
    parser.add_argument('--d_model', type=int, default=D_MODEL,
                        help=f'Transformer embedding dimension (default {D_MODEL})')
    parser.add_argument('--n_heads', type=int, default=N_HEADS,
                        help=f'Number of attention heads (default {N_HEADS})')
    parser.add_argument('--d_ff', type=int, default=D_FF,
                        help=f'Feed-forward hidden dimension (default {D_FF})')
    args = parser.parse_args()
    main(args)