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
  Input (288, 10) → [masking de CGM] → Transformer Encoder → H (288, d_model)
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
  # Run original
  python scripts/experiment_mtsm.py --max_patients 100 --epochs 50

  # Run 4: enmascaramiento agresivo
  python scripts/experiment_mtsm.py --max_patients 100 --epochs 50 \\
      --mask_ratio 0.45 --mask_max_len 120 --run_id run4

  # Run 5: agresivo + shape loss
  python scripts/experiment_mtsm.py --max_patients 100 --epochs 50 \\
      --mask_ratio 0.45 --mask_max_len 120 --shape_loss 0.2 --run_id run5

  # Run 6: agresivo + shape loss + multimodal masking
  python scripts/experiment_mtsm.py --max_patients 100 --epochs 50 \\
      --mask_ratio 0.45 --mask_max_len 120 --shape_loss 0.2 \\
      --multimodal_prob 0.2 --run_id run6
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── Config (defaults — todos sobreescribibles por CLI) ────────────────────────

WINDOW_LEN  = 288      # steps por ventana (24h a 5min/step)
N_FEATURES  = 10
CGM_IDX     = 0        # feature 0 = CGM
PI_IDX      = 1        # feature 1 = Plasma Insulin
RA_IDX      = 2        # feature 2 = Rate of Appearance
BOLUS_IDX   = 5        # feature 5 = bolus_logged
CARBS_IDX   = 6        # feature 6 = carbs_logged

# Masking defaults
MASK_RATIO   = 0.25
MASK_MIN_LEN = 36       # 3h minimo
MASK_MAX_LEN = 72       # 6h maximo (Idea 1: hasta 120 = 10h)
MASK_TOKEN   = 0.0

# Transformer
D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 5
D_FF     = 256
DROPOUT  = 0.1

# Training
BATCH_SIZE = 64
EPOCHS     = 50
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
    out = layers.Dense(1, name='reconstruction_head')(H)       # (batch, 288, 1)
    out = layers.Reshape((window_len,), name='output')(out)    # (batch, 288)
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


def apply_mask_batch(windows: np.ndarray, mask_ratio: float,
                     min_len: int, max_len: int,
                     mask_token: float = 0.0,
                     multimodal_prob: float = 0.0):
    """
    Aplica masking a un batch de ventanas.

    Idea 2 — Multimodal masking (multimodal_prob > 0):
      Con probabilidad multimodal_prob, enmascara PI (feature 1) o RA (feature 2)
      en lugar de CGM (feature 0). El encoder debe entonces inferir el driver
      a partir de la curva de glucosa visible — fuerza el aprendizaje de la
      relacion causal bidireccional entre glucosa y drivers fisiologicos.

    Returns:
        x_masked:         (N, 288, 10)
        masks:            (N, 288)
        targets:          (N, 288)     — valores reales del canal enmascarado
        masked_channels:  (N,)         — 0=CGM, 1=PI, 2=RA
    """
    N = len(windows)
    x_masked        = windows.copy()
    masks           = np.zeros((N, WINDOW_LEN), dtype=np.float32)
    targets         = np.zeros((N, WINDOW_LEN), dtype=np.float32)
    masked_channels = np.zeros(N, dtype=np.int32)

    for i in range(N):
        m = create_mask(windows.shape[1], mask_ratio, min_len, max_len)

        # Idea 2: elegir canal a enmascarar
        if multimodal_prob > 0 and np.random.random() < multimodal_prob:
            ch = np.random.choice([PI_IDX, RA_IDX])
        else:
            ch = CGM_IDX

        x_masked[i, m.astype(bool), ch] = mask_token
        masks[i]           = m
        targets[i]         = windows[i, :, ch]
        masked_channels[i] = ch

    return x_masked, masks, targets, masked_channels


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

def compute_driver_weights(windows: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Pesos de driver: DRIVER_LOSS_WEIGHT en timesteps enmascarados
    que caen dentro de DRIVER_EFFECT_STEPS despues de un evento bolus/carbs."""
    bolus = windows[:, :, BOLUS_IDX]
    carbs = windows[:, :, CARBS_IDX]

    driver_event     = ((bolus + carbs) > 0).astype(np.float32)
    driver_influence = np.zeros_like(driver_event)
    for offset in range(1, DRIVER_EFFECT_STEPS + 1):
        driver_influence[:, offset:] += driver_event[:, :-offset]
    driver_influence = np.clip(driver_influence, 0, 1)

    driver_weight = np.where(
        (driver_influence > 0) & (masks > 0),
        DRIVER_LOSS_WEIGHT, 1.0
    ).astype(np.float32)
    return driver_weight


def prepare_data(windows: np.ndarray, mask_ratio: float,
                 mask_min_len: int, mask_max_len: int,
                 multimodal_prob: float):
    """
    Genera pares (X_masked, Y) para el MTSM.
    Y: (N, 288, 3) = [target_real, mask, driver_weight]
    """
    print(f"  Masking  ratio={mask_ratio:.0%}  "
          f"spans={mask_min_len*5//60}h-{mask_max_len*5//60}h  "
          f"multimodal_prob={multimodal_prob:.0%}")

    x_masked, masks, targets, masked_channels = apply_mask_batch(
        windows, mask_ratio, mask_min_len, mask_max_len, MASK_TOKEN, multimodal_prob
    )

    driver_weight = compute_driver_weights(windows, masks)
    Y = np.stack([targets, masks, driver_weight], axis=-1).astype(np.float32)

    n   = len(x_masked)
    idx = np.random.permutation(n)
    n_test = int(n * TEST_SPLIT)
    n_val  = int(n * VAL_SPLIT)
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    masked_per_window = masks.sum(axis=1).mean()
    n_cgm = (masked_channels == CGM_IDX).sum()
    n_pi  = (masked_channels == PI_IDX).sum()
    n_ra  = (masked_channels == RA_IDX).sum()
    print(f"  Timesteps enmascarados/ventana: "
          f"{masked_per_window:.1f} ({masked_per_window/WINDOW_LEN*100:.1f}%)")
    print(f"  Canales — CGM: {n_cgm} ({n_cgm/n*100:.1f}%)  "
          f"PI: {n_pi} ({n_pi/n*100:.1f}%)  RA: {n_ra} ({n_ra/n*100:.1f}%)")
    print(f"  Train: {len(train_idx):>7,}  Val: {len(val_idx):>7,}  "
          f"Test: {len(test_idx):>7,}")

    return (
        (x_masked[train_idx], Y[train_idx]),
        (x_masked[val_idx],   Y[val_idx]),
        (x_masked[test_idx],  Y[test_idx]),
        windows[test_idx],
        masks[test_idx],
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(model, X_train, Y_train, X_val, Y_val, epochs, shape_loss_lambda):
    print(f"\n{'─'*52}")
    print(f"  Entrenando MTSM")
    print(f"  Parametros encoder:  "
          f"{model.get_layer('TransformerEncoder').count_params():,}")
    print(f"  Parametros totales:  {model.count_params():,}")
    print(f"  Shape loss lambda:   {shape_loss_lambda}")
    print(f"{'─'*52}")

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LR, weight_decay=1e-4),
        loss=MaskedMSELoss(shape_loss_lambda=shape_loss_lambda),
        metrics=[MaskedMAE()]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
    ]

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    return history


# ── Plot: training curves ─────────────────────────────────────────────────────

def plot_training_curves(history, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('MTSM Training Curves — Masked Reconstruction',
                 fontsize=14, fontweight='bold')

    for ax, (metric, label) in zip(axes, [('loss', 'Masked MSE Loss (+ shape)'),
                                           ('masked_mae', 'Masked MAE')]):
        ax.plot(history.history[metric],
                color='#2563EB', lw=2, ls='-', label='train')
        ax.plot(history.history[f'val_{metric}'],
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

    selected = list(flat_candidates[:2]) + list(dynamic_candidates[:2])
    labels   = ['Basal (flat)', 'Basal (flat)', 'Dynamic (peak)', 'Dynamic (peak)']

    x_masked_sel = windows_orig[selected].copy()
    for j, idx in enumerate(selected):
        m = masks[idx].astype(bool)
        x_masked_sel[j, m, CGM_IDX] = MASK_TOKEN

    y_pred = model.predict(x_masked_sel, verbose=0)

    fig = plt.figure(figsize=(18, 5 * 4))
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.6)
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

def load_windows(processed_dir: str, max_patients: int = None) -> np.ndarray:
    npz_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('.npz')])
    if not npz_files:
        raise FileNotFoundError(f"No .npz encontrados en {processed_dir}")
    if max_patients is not None:
        npz_files = npz_files[:max_patients]
        print(f"  (--max_patients {max_patients}: cargando subset)")

    all_windows = []
    for fname in npz_files:
        data = np.load(os.path.join(processed_dir, fname), allow_pickle=True)
        all_windows.append(data['windows'])

    windows = np.concatenate(all_windows, axis=0).astype(np.float32)
    print(f"  Pacientes: {len(npz_files)}   "
          f"Ventanas: {windows.shape[0]:,}   Shape: {windows.shape}")
    return windows


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
    print(f"  Config guardada: {config_path}")


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

    # 1. Cargar datos
    print(f"\n  Cargando datos desde: {args.data}")
    windows = load_windows(args.data, args.max_patients)

    # 2. Preparar datos
    print(f"\n{'='*52}")
    print(f"  Preparando datos")
    print(f"{'='*52}")
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), \
        windows_test_orig, masks_test = prepare_data(
            windows,
            mask_ratio=args.mask_ratio,
            mask_min_len=MASK_MIN_LEN,
            mask_max_len=args.mask_max_len,
            multimodal_prob=args.multimodal_prob
        )

    # 3. Construir modelo
    model, encoder = build_mtsm_model(
        WINDOW_LEN, N_FEATURES, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT
    )
    model.summary()

    # 4. Entrenar
    history = train_model(
        model, X_train, Y_train, X_val, Y_val,
        epochs=args.epochs,
        shape_loss_lambda=args.shape_loss
    )

    # 5. Plots
    print(f"\n{'='*52}")
    print(f"  Generando plots")
    print(f"{'='*52}")

    plot_training_curves(
        history,
        os.path.join(results_dir, 'training_curves.png')
    )

    variability = (windows_test_orig[:, :, CGM_IDX].max(axis=1)
                 - windows_test_orig[:, :, CGM_IDX].min(axis=1))
    h_idx = int(np.argmax(variability))
    plot_H_analysis(
        encoder,
        windows_test_orig[h_idx:h_idx+1],
        n_layers=N_LAYERS, n_heads=N_HEADS, d_model=D_MODEL,
        save_path=os.path.join(results_dir, 'transformer_H_analysis.png')
    )

    plot_reconstruction_examples(
        model, windows_test_orig, masks_test,
        mask_ratio=args.mask_ratio,
        mask_min_len=MASK_MIN_LEN,
        mask_max_len=args.mask_max_len,
        save_path=os.path.join(results_dir, 'reconstruction_examples.png')
    )

    print(f"\n{'='*52}")
    print(f"  Completado. Resultados en: {results_dir}/")
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
    args = parser.parse_args()
    main(args)