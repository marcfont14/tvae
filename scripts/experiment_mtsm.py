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
  training_curves.png          loss train/val por epoch
  transformer_H_analysis.png   attention weights + norma H_t + PCA
  reconstruction_examples.png  reconstrucción de spans con drivers superpuestos
  run_config.txt               configuración exacta del run para reproducibilidad

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
                     d_ff, dropout):
    inp     = keras.Input(shape=(window_len, n_features), name='input')
    encoder = build_transformer_encoder(
        window_len, n_features, d_model, n_heads, n_layers, d_ff, dropout
    )
    H   = encoder(inp)
    out = layers.Dense(64, activation='relu', name='recon_hidden')(H)  # (batch, 288, 64)
    out = layers.Dense(1, name='reconstruction_head')(out)             # (batch, 288, 1)
    out = layers.Reshape((window_len,), name='output')(out)            # (batch, 288)
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
            ch = int(np.random.choice([PI_IDX, RA_IDX]))
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




# ── Plot: training curves ─────────────────────────────────────────────────────

def plot_training_curves(history, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('MTSM Training Curves — Masked Reconstruction',
                 fontsize=14, fontweight='bold')

    for ax, (metric, label) in zip(axes, [('loss', 'Masked MSE Loss (+ shape)'),
                                           ('masked_mae', 'Masked MAE')]):
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


# ── Plot: H analysis ──────────────────────────────────────────────────────────

def plot_H_analysis(encoder, X_sample: np.ndarray, n_layers: int,
                    n_heads: int, d_model: int, save_path: str):
    from sklearn.decomposition import PCA

    print(f"  Analizando H...")

    x_in = tf.cast(X_sample[:1], tf.float32)
    H    = encoder(x_in, training=False).numpy()[0]   # (288, d_model)

    if n_layers > 1:
        pre_model = keras.Model(
            inputs=encoder.input,
            outputs=encoder.get_layer(f'norm2_{n_layers-2}').output
        )
    else:
        pre_model = keras.Model(
            inputs=encoder.input,
            outputs=encoder.get_layer('input_proj').output
        )
    x_pre = pre_model(x_in, training=False)

    last_mhsa = encoder.get_layer(f'mhsa_{n_layers-1}')
    _, attn_scores = last_mhsa(x_pre, x_pre,
                               return_attention_scores=True, training=False)
    attn_mean = attn_scores[0].numpy().mean(axis=0)

    cgm    = X_sample[0, :, CGM_IDX]
    H_norm = np.linalg.norm(H, axis=1)

    pca  = PCA(n_components=2)
    H_2d = pca.fit_transform(H)
    var  = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.35)

    # Panel 1: Attention heatmap
    ax1  = fig.add_subplot(gs[0, :])
    sstp = max(1, WINDOW_LEN // 64)
    asub = attn_mean[::sstp, ::sstp]
    n_sub = asub.shape[0]

    im = ax1.imshow(asub, aspect='auto', cmap='Blues', origin='upper',
                    interpolation='nearest')
    plt.colorbar(im, ax=ax1, shrink=0.5, label='Attention weight')

    tick_every = max(1, n_sub // 12)
    tick_pos   = list(range(0, n_sub, tick_every))
    tick_lbl   = [f'{(t*sstp)*5//60}h' for t in tick_pos]
    ax1.set_xticks(tick_pos); ax1.set_xticklabels(tick_lbl, fontsize=8)
    ax1.set_yticks(tick_pos); ax1.set_yticklabels(tick_lbl, fontsize=8)
    ax1.set_xlabel('Key timestep j  (a quien se atiende)', fontsize=9)
    ax1.set_ylabel('Query timestep i  (quien atiende)', fontsize=9)
    ax1.set_title(
        f'Panel 1 — Attention weights (ultima capa MHSA, media {n_heads} heads)\n'
        f'Diagonal = atencion local  |  Off-diagonal = dependencias largo alcance',
        fontsize=10
    )

    # Panel 2: Norma H_t vs CGM
    ax2      = fig.add_subplot(gs[1, 0])
    ax2_twin = ax2.twinx()
    t = np.arange(WINDOW_LEN)
    ax2.plot(t, H_norm, color='#7C3AED', lw=1.5, alpha=0.9, label='||H_t||_2')
    ax2_twin.plot(t, cgm, color='#6B7280', lw=1.2, alpha=0.6, label='CGM (z-score)')
    ax2.set_xlabel('Timestep (5 min)', fontsize=9)
    ax2.set_ylabel('||H_t||_2', fontsize=9, color='#7C3AED')
    ax2_twin.set_ylabel('CGM (z-score)', fontsize=9, color='#6B7280')
    ax2.set_title('Panel 2 — Norma L2 de H_t vs CGM', fontsize=10)
    lines = ax2.get_lines() + ax2_twin.get_lines()
    ax2.legend(lines, [l.get_label() for l in lines], fontsize=9)
    ax2.grid(True, alpha=0.2)
    ax2.spines[['top']].set_visible(False)
    step_t = 48
    ax2.set_xticks(range(0, WINDOW_LEN + 1, step_t))
    ax2.set_xticklabels([f'{t*5//60}h' for t in range(0, WINDOW_LEN + 1, step_t)],
                        fontsize=8)

    # Panel 3: PCA
    ax3 = fig.add_subplot(gs[1, 1])
    sc  = ax3.scatter(H_2d[:, 0], H_2d[:, 1], c=cgm, cmap='RdYlGn_r',
                      s=10, alpha=0.7)
    plt.colorbar(sc, ax=ax3, label='CGM (z-score)')
    ax3.set_xlabel(f'PC1 ({var[0]*100:.1f}% var)', fontsize=9)
    ax3.set_ylabel(f'PC2 ({var[1]*100:.1f}% var)', fontsize=9)
    ax3.set_title(
        'Panel 3 — PCA de H → R2\nCada punto = un timestep  |  color = CGM',
        fontsize=10
    )
    ax3.grid(True, alpha=0.2)
    ax3.spines[['top', 'right']].set_visible(False)

    fig.suptitle(
        f'Analisis de H — MTSM Encoder (Stage 1)\n'
        f'H in R^({WINDOW_LEN}x{d_model})  |  '
        f'{n_layers} capas  {n_heads} heads  d_model={d_model}',
        fontsize=13, fontweight='bold'
    )
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

def plot_driver_metrics(model, windows_orig: np.ndarray,
                        masks: np.ndarray, save_path: str):
    """
    Tres métricas cuantitativas de respuesta a drivers:
      - MAE postprandial (0-2h tras carbs)
      - MAE postbolus (0-2h tras bolus)
      - MAE basal (resto)
    Más histograma de distribución del error por zona.
    """
    cgm   = windows_orig[:, :, CGM_IDX]
    bolus = windows_orig[:, :, BOLUS_IDX]
    carbs = windows_orig[:, :, CARBS_IDX]

    # Generar predicciones sobre el test set completo
    x_masked_all = windows_orig.copy()
    for i in range(len(windows_orig)):
        m = masks[i].astype(bool)
        x_masked_all[i, m, CGM_IDX] = MASK_TOKEN

    y_pred = model.predict(x_masked_all, batch_size=BATCH_SIZE, verbose=0)
    # shape: (N, 288)

    # Propagar influencia de eventos hacia adelante 2h (24 steps)
    effect_steps = DRIVER_EFFECT_STEPS  # 24 steps = 2h

    carbs_influence = np.zeros_like(carbs)
    bolus_influence = np.zeros_like(bolus)
    for offset in range(1, effect_steps + 1):
        carbs_influence[:, offset:] += (carbs[:, :-offset] > 0).astype(np.float32)
        bolus_influence[:, offset:] += (bolus[:, :-offset] > 0).astype(np.float32)
    carbs_influence = np.clip(carbs_influence, 0, 1)
    bolus_influence = np.clip(bolus_influence, 0, 1)

    # Error absoluto solo en timesteps enmascarados
    abs_err = np.abs(cgm - y_pred) * masks  # (N, 288)

    # Clasificar timesteps enmascarados en tres zonas
    postprandial_mask = (carbs_influence > 0) & (masks > 0)
    postbolus_mask    = (bolus_influence  > 0) & (masks > 0) & (carbs_influence == 0)
    basal_mask        = (masks > 0) & (carbs_influence == 0) & (bolus_influence == 0)

    def safe_mae(err, zone):
        n = zone.sum()
        return (err * zone).sum() / n if n > 0 else 0.0

    mae_postprandial = safe_mae(abs_err, postprandial_mask)
    mae_postbolus    = safe_mae(abs_err, postbolus_mask)
    mae_basal        = safe_mae(abs_err, basal_mask)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Driver Response Metrics — MAE by Zone', 
                 fontsize=14, fontweight='bold')

    # Panel izquierdo: barras MAE por zona
    ax = axes[0]
    zones  = ['Postprandial\n(0-2h tras carbs)',
              'Post-bolus\n(0-2h tras bolus)',
              'Basal\n(sin driver)']
    values = [mae_postprandial, mae_postbolus, mae_basal]
    colors_bar = [COLORS['carbs'], COLORS['bolus'], '#6B7280']
    bars = ax.bar(zones, values, color=colors_bar, alpha=0.8, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('MAE (z-score)', fontsize=11)
    ax.set_title('MAE por zona de driver', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines[['top', 'right']].set_visible(False)

    # Panel derecho: histograma de errores por zona
    ax2 = axes[1]
    err_pp  = abs_err[postprandial_mask].flatten()
    err_pb  = abs_err[postbolus_mask].flatten()
    err_bas = abs_err[basal_mask].flatten()

    bins = np.linspace(0, np.percentile(abs_err[masks > 0], 95), 40)
    ax2.hist(err_bas, bins=bins, alpha=0.5, color='#6B7280',
             label=f'Basal (n={basal_mask.sum():,})', density=True)
    ax2.hist(err_pp,  bins=bins, alpha=0.6, color=COLORS['carbs'],
             label=f'Postprandial (n={postprandial_mask.sum():,})', density=True)
    ax2.hist(err_pb,  bins=bins, alpha=0.6, color=COLORS['bolus'],
             label=f'Post-bolus (n={postbolus_mask.sum():,})', density=True)
    ax2.set_xlabel('MAE (z-score)', fontsize=11)
    ax2.set_ylabel('Densidad', fontsize=11)
    ax2.set_title('Distribución del error por zona', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Imprimir resumen en consola
    print(f"  MAE postprandial: {mae_postprandial:.4f} z-score  "
          f"(n={postprandial_mask.sum():,} timesteps)")
    print(f"  MAE post-bolus:   {mae_postbolus:.4f} z-score  "
          f"(n={postbolus_mask.sum():,} timesteps)")
    print(f"  MAE basal:        {mae_basal:.4f} z-score  "
          f"(n={basal_mask.sum():,} timesteps)")
    print(f"  Guardado: {save_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    from datetime import datetime

    run_id      = args.run_id if args.run_id else datetime.now().strftime('%Y%m%d_%H%M')
    results_dir = os.path.join(RESULTS_BASE, run_id)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*52}")
    print(f"  MTSM — Stage 1 Pre-training")
    print(f"  Run ID:          {run_id}")
    print(f"  mask_ratio:      {args.mask_ratio:.0%}")
    print(f"  mask_max_len:    {args.mask_max_len} steps = {args.mask_max_len*5//60}h")
    print(f"  shape_loss L:    {args.shape_loss}")
    print(f"  multimodal_prob: {args.multimodal_prob:.0%}")
    print(f"{'='*52}")

    save_run_config(results_dir, args)

    # 1. Index dataset (reads metadata only — windows never accumulate in RAM)
    print(f"\n  Indexing dataset from: {args.data}")
    index = index_dataset(args.data, args.max_patients)

    # 2. Patient-level split (prevents data leakage across splits)
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

    # 3. Build tf.data pipelines — lazy loading, masking on-the-fly, no RAM accumulation
    ds_kwargs = dict(
        mask_ratio=args.mask_ratio, mask_min_len=MASK_MIN_LEN,
        mask_max_len=args.mask_max_len, multimodal_prob=args.multimodal_prob,
        no_logged_events=args.no_logged_events, no_age=args.no_age
    )
    train_ds = make_window_dataset(train_index, shuffle=True,  **ds_kwargs)
    val_ds   = make_window_dataset(val_index,   shuffle=False, **ds_kwargs)

    # 4. Build model
    n_features_model = 10 if args.no_age else N_FEATURES
    model, encoder = build_mtsm_model(
        WINDOW_LEN, n_features_model, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT
    )
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
    history_obj = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    history = history_obj.history

    # 6. Save encoder weights
    encoder_path = os.path.join(results_dir, 'encoder_weights.weights.h5')
    encoder.save_weights(encoder_path)
    print(f"\n  Encoder weights saved: {encoder_path}")

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
    # When --no_age, strip feature 10 so model input matches training shape.
    # Feature indices 0-9 (CGM, PI, RA, bolus, carbs, …) are unaffected.
    if args.no_age:
        test_windows = test_windows[:, :, :10]

    # 8. Plots
    print(f"\n{'='*52}")
    print(f"  Generating plots")
    print(f"{'='*52}")

    masks_test = np.stack([
        create_mask(WINDOW_LEN, args.mask_ratio, MASK_MIN_LEN, args.mask_max_len)
        for _ in range(len(test_windows))
    ])

    plot_training_curves(
        history,
        os.path.join(results_dir, 'training_curves.png')
    )

    variability = (test_windows[:, :, CGM_IDX].max(axis=1)
                 - test_windows[:, :, CGM_IDX].min(axis=1))
    h_idx = int(np.argmax(variability))
    plot_H_analysis(
        encoder,
        test_windows[h_idx:h_idx+1],
        n_layers=N_LAYERS, n_heads=N_HEADS, d_model=D_MODEL,
        save_path=os.path.join(results_dir, 'transformer_H_analysis.png')
    )

    plot_reconstruction_examples(
        model, test_windows, masks_test,
        mask_ratio=args.mask_ratio,
        mask_min_len=MASK_MIN_LEN,
        mask_max_len=args.mask_max_len,
        save_path=os.path.join(results_dir, 'reconstruction_examples.png')
    )

    plot_driver_metrics(
        model, test_windows, masks_test,
        save_path=os.path.join(results_dir, 'driver_metrics.png')
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
    args = parser.parse_args()
    main(args)