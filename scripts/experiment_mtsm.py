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

Arquitectura:
  Input (288, 10) → [masking de CGM] → Transformer Encoder → H (288, d_model)
  → Reconstruction Head: Dense(1) por timestep → ŷ_CGM sobre spans enmascarados
  → MSE loss vs CGM real en esos timesteps

El Reconstruction Head se desecha después del pre-training.
En Stage 2 se conecta Attention Pooling + VAE en su lugar.

Outputs (en results/mtsm):
  training_curves.png          loss train/val por epoch
  transformer_H_analysis.png   attention weights + norma H_t + PCA
  reconstruction_examples.png  reconstrucción de spans con drivers superpuestos

Usage:
  python scripts/experiment_mtsm.py
  python scripts/experiment_mtsm.py --max_patients 100 --epochs 30
  python scripts/experiment_mtsm.py --mask_ratio 0.25 --debug
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from polars import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── Config ────────────────────────────────────────────────────────────────────

WINDOW_LEN  = 288      # steps por ventana (24h a 5min/step)
N_FEATURES  = 10
CGM_IDX     = 0        # feature 0 = CGM
PI_IDX      = 1        # feature 1 = Plasma Insulin
RA_IDX      = 2        # feature 2 = Rate of Appearance

# Masking
MASK_RATIO   = 0.25    # fracción de la ventana a enmascarar (hiperparámetro)
MASK_MIN_LEN = 36       # span mínimo: 3h
MASK_MAX_LEN = 72      # span máximo: 6h
MASK_TOKEN   = 0.0     # valor que reemplaza CGM en los timesteps enmascarados

# Transformer
D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 5
D_FF     = 256
DROPOUT  = 0.1

# Training
BATCH_SIZE = 64    # conservador — con seq=288 la attention matrix es grande
EPOCHS     = 50
LR         = 1e-3
VAL_SPLIT  = 0.1
TEST_SPLIT  = 0.1

RESULTS_DIR = 'results/mtsm'
SEED        = 42

COLORS = {
    'cgm_real':    '#111827',   # negro
    'cgm_masked':  '#E5E7EB',   # gris claro — zona enmascarada visible
    'recon':       '#2563EB',   # azul — reconstrucción
    'pi':          '#7C3AED',   # morado — PI
    'ra':          '#059669',   # verde — RA
    'bolus':       '#DC2626',   # rojo — eventos bolus
    'carbs':       '#D97706',   # naranja — eventos carbs
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
    """
    Transformer encoder puro — mismo diseño que el modelo final (Stage 1).

    Input:  (batch, window_len, n_features)   — CGM enmascarado + drivers intactos
    Output: H ∈ (batch, window_len, d_model)  — representaciones contextuales

    H_t ∈ R^d_model por cada timestep — valores reales sin restricción.
    En los timesteps enmascarados, H_t codifica lo que el encoder infiere
    del contexto (pasado, futuro, drivers) sobre el CGM que falta.
    """
    inp = keras.Input(shape=(window_len, n_features), name='input')

    x  = layers.Dense(d_model, name='input_proj')(inp)
    pe = get_positional_encoding(window_len, d_model)
    x  = x + pe

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

    # H: output del encoder — NO se aplica GAP aquí.
    # En Stage 1 necesitamos H completo para el reconstruction head.
    # En Stage 2 se añade attention pooling sobre H para obtener h.
    return keras.Model(inp, x, name='TransformerEncoder')


# ── MTSM Model (Encoder + Reconstruction Head) ────────────────────────────────

def build_mtsm_model(window_len, n_features, d_model, n_heads, n_layers,
                     d_ff, dropout):
    """
    Encoder + Reconstruction Head para Stage 1.

    Flujo:
      x_masked (batch, 288, 10) → Encoder → H (batch, 288, d_model)
                                           → Dense(1) por timestep
                                           → ŷ_CGM (batch, 288, 1)

    El reconstruction head es un Dense(1) compartido (same weights)
    aplicado a cada timestep independientemente — position-wise.
    Es intencionalmente simple: su único rol es proyectar H_t → scalar CGM
    para poder calcular el loss. No se quiere que el head haga trabajo
    que debería hacer el encoder.

    Loss: MSE solo sobre timesteps enmascarados (ver MaskedMSELoss).
    El head se descarta después del pre-training.
    """
    inp     = keras.Input(shape=(window_len, n_features), name='input')
    encoder = build_transformer_encoder(
        window_len, n_features, d_model, n_heads, n_layers, d_ff, dropout
    )
    H    = encoder(inp)                                          # (batch, 288, d_model)
    out  = layers.Dense(1, name='reconstruction_head')(H)       # (batch, 288, 1)
    out  = layers.Reshape((window_len,), name='output')(out)    # (batch, 288)

    model = keras.Model(inp, out, name='MTSM')
    return model, encoder


# ── Masking ───────────────────────────────────────────────────────────────────

def create_mask(window_len: int, mask_ratio: float,
                min_len: int, max_len: int) -> np.ndarray:
    """
    Genera una máscara binaria para una ventana.
    mask[t] = 1 → timestep t enmascarado (CGM oculto)
    mask[t] = 0 → timestep t visible

    Estrategia: spans contiguos aleatorios hasta cubrir ~mask_ratio.
    Spans contiguos fuerzan al modelo a reconstruir la FORMA de la curva,
    no solo interpolar valores individuales.
    """
    mask       = np.zeros(window_len, dtype=np.float32)
    target_len = int(window_len * mask_ratio)
    masked_so_far = 0

    max_attempts = 50
    attempts = 0
    while masked_so_far < target_len and attempts < max_attempts:
        span_len = np.random.randint(min_len, max_len + 1)
        start    = np.random.randint(0, window_len - span_len)
        mask[start:start + span_len] = 1
        masked_so_far = mask.sum()
        attempts += 1

    return mask


def apply_mask_batch(windows: np.ndarray, mask_ratio: float,
                     min_len: int, max_len: int, mask_token: float = 0.0):
    """
    Aplica masking a un batch de ventanas.

    Solo se enmascara CGM (feature 0). PI, RA y el resto de features
    permanecen visibles — el modelo puede usarlos para la reconstrucción.

    Returns:
        x_masked:  (N, 288, 10) — input con CGM enmascarado
        masks:     (N, 288)     — 1 donde CGM fue enmascarado
        cgm_real:  (N, 288)     — CGM original para calcular el loss
    """
    N = len(windows)
    x_masked = windows.copy()
    masks    = np.zeros((N, window_len), dtype=np.float32)

    for i in range(N):
        m = create_mask(windows.shape[1], mask_ratio, min_len, max_len)
        x_masked[i, m.astype(bool), CGM_IDX] = mask_token
        masks[i] = m

    cgm_real = windows[:, :, CGM_IDX]   # (N, 288)
    return x_masked, masks, cgm_real


# ── Custom Loss ───────────────────────────────────────────────────────────────

DRIVER_LOSS_WEIGHT  = 3.0   # peso extra en timesteps con influencia de driver
DRIVER_EFFECT_STEPS = 24    # 2h de efecto fisiológico después de un evento (24×5min)

class MaskedMSELoss(keras.losses.Loss):
    """
    MSE ponderado calculado solo sobre los timesteps enmascarados.

    Los timesteps dentro del span enmascarado que siguen a un evento de
    bolus o carbs reciben peso DRIVER_LOSS_WEIGHT (default 3.0).
    El resto de timesteps enmascarados reciben peso 1.0.

    Esto fuerza al modelo a aprender la respuesta fisiológica a los drivers
    en lugar de simplemente interpolar entre los bordes del span enmascarado.
    """
    def call(self, y_true, y_pred, sample_weight=None):
        # y_true: (batch, 288, 3) — [cgm_real, mask, driver_weight]
        # y_pred: (batch, 288)
        cgm_real      = y_true[:, :, 0]
        mask          = y_true[:, :, 1]
        driver_weight = y_true[:, :, 2]

        sq_err   = tf.square(cgm_real - y_pred)          # (batch, 288)
        weighted = sq_err * mask * driver_weight          # peso extra en zonas de driver
        n_masked = tf.reduce_sum(mask, axis=1, keepdims=True) + 1e-8
        loss     = tf.reduce_sum(weighted, axis=1) / tf.squeeze(n_masked, axis=1)
        return tf.reduce_mean(loss)


class MaskedMAE(keras.metrics.Metric):
    """MAE sobre timesteps enmascarados — para monitorizar durante el training."""
    def __init__(self, name='masked_mae', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        cgm_real = y_true[:, :, 0]
        mask     = y_true[:, :, 1]
        # MAE sin ponderar — métrica interpretable independiente del driver_weight
        abs_err  = tf.abs(cgm_real - y_pred) * mask
        self.total.assign_add(tf.reduce_sum(abs_err))
        self.count.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.total / (self.count + 1e-8)

    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_data(windows: np.ndarray, mask_ratio: float):
    """
    Genera los pares (X_masked, Y) para el MTSM.

    Y tiene shape (N, 288, 3): concatenación de [cgm_real, mask, driver_weight]
    porque Keras necesita un solo tensor de targets — empaquetamos
    el mask, el target y los pesos de driver juntos para el loss.
    """
    print(f"  Aplicando masking  (ratio={mask_ratio}, "
          f"spans={MASK_MIN_LEN}-{MASK_MAX_LEN} steps = "
          f"{MASK_MIN_LEN*5}-{MASK_MAX_LEN*5} min)...")

    x_masked, masks, cgm_real = apply_mask_batch(
        windows, mask_ratio, MASK_MIN_LEN, MASK_MAX_LEN, MASK_TOKEN
    )

    # Driver weight: timesteps dentro de la máscara que siguen a un evento
    # de bolus o carbs reciben peso DRIVER_LOSS_WEIGHT, el resto peso 1.0.
    # Fuerza al modelo a aprender la respuesta fisiológica a los drivers
    # en lugar de simplemente interpolar entre los bordes del span.
    bolus = windows[:, :, 5]   # bolus_logged (binary)
    carbs = windows[:, :, 6]   # carbs_logged (binary)

    # Propagar el efecto del evento hacia adelante DRIVER_EFFECT_STEPS steps
    # (un bolus/carbs tiene efecto fisiológico durante ~1-2h después del evento)
    driver_event = ((bolus + carbs) > 0).astype(np.float32)
    driver_influence = np.zeros_like(driver_event)
    for offset in range(1, DRIVER_EFFECT_STEPS + 1):
        driver_influence[:, offset:] += driver_event[:, :-offset]
    driver_influence = np.clip(driver_influence, 0, 1)

    # weight = DRIVER_LOSS_WEIGHT donde hay influencia de driver Y máscara activa
    # weight = 1.0 en el resto de timesteps enmascarados
    driver_weight = np.where(
        (driver_influence > 0) & (masks > 0),
        DRIVER_LOSS_WEIGHT,
        1.0
    ).astype(np.float32)

    # Empaquetar target: (N, 288, 3) = [cgm_real, mask, driver_weight]
    Y = np.stack([cgm_real, masks, driver_weight], axis=-1).astype(np.float32)

    n   = len(x_masked)
    idx = np.random.permutation(n)
    n_test = int(n * TEST_SPLIT)
    n_val  = int(n * VAL_SPLIT)

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    masked_per_window = masks.sum(axis=1).mean()
    print(f"  Timesteps enmascarados por ventana (media): {masked_per_window:.1f} "
          f"({masked_per_window/WINDOW_LEN*100:.1f}%)")
    print(f"  Train: {len(train_idx):>7,} ventanas")
    print(f"  Val:   {len(val_idx):>7,} ventanas")
    print(f"  Test:  {len(test_idx):>7,} ventanas")

    return (
        (x_masked[train_idx], Y[train_idx]),
        (x_masked[val_idx],   Y[val_idx]),
        (x_masked[test_idx],  Y[test_idx]),
        windows[test_idx],   # ventanas originales (sin mask) para los plots
        masks[test_idx],
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(model, X_train, Y_train, X_val, Y_val):
    print(f"\n{'─'*52}")
    print(f"  Entrenando MTSM")
    print(f"  Parametros encoder:  {model.get_layer('TransformerEncoder').count_params():,}")
    print(f"  Parametros totales:  {model.count_params():,}")
    print(f"{'─'*52}")

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LR, weight_decay=1e-4),
        loss=MaskedMSELoss(),
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
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    return history


# ── Plot: training curves ─────────────────────────────────────────────────────

def plot_training_curves(history, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('MTSM Training Curves — Masked CGM Reconstruction',
                 fontsize=14, fontweight='bold')

    for ax, (metric, label) in zip(axes, [('loss', 'Masked MSE Loss'),
                                           ('masked_mae', 'Masked MAE')]):
        ax.plot(history.history[metric],           color='#2563EB', lw=2,
                ls='-',  label='train')
        ax.plot(history.history[f'val_{metric}'],  color='#2563EB', lw=1.5,
                ls='--', label='val')
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

def plot_H_analysis(encoder, X_sample: np.ndarray, save_path: str):
    """
    Análisis de las representaciones internas H del encoder.
    Igual que en el experimento de forecasting — tres paneles:
      1. Attention weights (última capa MHSA)
      2. Norma L2 de H_t vs CGM
      3. PCA de H → R²
    """
    from sklearn.decomposition import PCA

    print(f"  Analizando H...")

    x_in = tf.cast(X_sample[:1], tf.float32)
    H    = encoder(x_in, training=False).numpy()[0]   # (288, d_model)

    # Representación pre-última MHSA
    if N_LAYERS > 1:
        pre_model = keras.Model(
            inputs=encoder.input,
            outputs=encoder.get_layer(f'norm2_{N_LAYERS-2}').output
        )
    else:
        pre_model = keras.Model(
            inputs=encoder.input,
            outputs=encoder.get_layer('input_proj').output
        )
    x_pre = pre_model(x_in, training=False)

    last_mhsa = encoder.get_layer(f'mhsa_{N_LAYERS-1}')
    _, attn_scores = last_mhsa(x_pre, x_pre,
                               return_attention_scores=True, training=False)
    attn_mean = attn_scores[0].numpy().mean(axis=0)   # (288, 288)

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
    ax1.set_xlabel('Key timestep j  (a quién se atiende)', fontsize=9)
    ax1.set_ylabel('Query timestep i  (quién atiende)', fontsize=9)
    ax1.set_title(
        f'Panel 1 — Attention weights  (última capa MHSA, media {N_HEADS} heads)\n'
        f'Celda (i,j) = cuánto atiende el timestep i al timestep j  '
        f'[subsampled ×{sstp}]',
        fontsize=10
    )

    # Panel 2: Norma H_t + CGM
    ax2      = fig.add_subplot(gs[1, 0])
    ax2_twin = ax2.twinx()
    t = np.arange(WINDOW_LEN)
    ax2.plot(t, H_norm, color='#7C3AED', lw=1.5, alpha=0.9, label='‖H_t‖₂')
    ax2_twin.plot(t, cgm, color='#6B7280', lw=1.2, alpha=0.6, label='CGM (z-score)')
    ax2.set_xlabel('Timestep (5 min)', fontsize=9)
    ax2.set_ylabel('‖H_t‖₂', fontsize=9, color='#7C3AED')
    ax2_twin.set_ylabel('CGM (z-score)', fontsize=9, color='#6B7280')
    ax2.set_title('Panel 2 — Norma L2 de H_t vs CGM', fontsize=10)
    lines  = ax2.get_lines() + ax2_twin.get_lines()
    ax2.legend(lines, [l.get_label() for l in lines], fontsize=9)
    ax2.grid(True, alpha=0.2)
    ax2.spines[['top']].set_visible(False)
    step_t = 48
    ax2.set_xticks(range(0, WINDOW_LEN + 1, step_t))
    ax2.set_xticklabels([f'{t*5//60}h' for t in range(0, WINDOW_LEN + 1, step_t)],
                        fontsize=8)

    # Panel 3: PCA
    ax3 = fig.add_subplot(gs[1, 1])
    sc  = ax3.scatter(H_2d[:, 0], H_2d[:, 1], c=cgm, cmap='RdYlGn_r', s=10, alpha=0.7)
    plt.colorbar(sc, ax=ax3, label='CGM (z-score)')
    ax3.set_xlabel(f'PC1 ({var[0]*100:.1f}% var)', fontsize=9)
    ax3.set_ylabel(f'PC2 ({var[1]*100:.1f}% var)', fontsize=9)
    ax3.set_title(
        'Panel 3 — PCA de H ∈ ℝ^(288×d_model) → ℝ²\n'
        'Cada punto = un timestep · color = valor CGM',
        fontsize=10
    )
    ax3.grid(True, alpha=0.2)
    ax3.spines[['top', 'right']].set_visible(False)

    fig.suptitle(
        f'Análisis de H — MTSM Encoder (post Stage 1 pre-training)\n'
        f'H ∈ ℝ^({WINDOW_LEN}×{D_MODEL})  |  {N_LAYERS} capas · {N_HEADS} heads · d_model={D_MODEL}',
        fontsize=13, fontweight='bold'
    )
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {save_path}")


# ── Plot: reconstruction examples ─────────────────────────────────────────────

def plot_reconstruction_examples(model, windows_orig: np.ndarray,
                                  masks: np.ndarray, save_path: str):
    """
    Visualiza la reconstrucción de spans enmascarados con drivers superpuestos.

    Selecciona 4 ejemplos:
      - 2 con span enmascarado en zona PLANA/BASAL  → reconstrucción fácil
      - 2 con span enmascarado sobre PICO POSTPRANDIAL → reconstrucción difícil

    Cada panel muestra:
      - CGM real completo (negro, línea continua)
      - Zona enmascarada (fondo gris) — lo que el modelo NO ve
      - Reconstrucción del modelo (azul) — solo sobre la zona enmascarada
      - PI (morado, eje secundario) — insulina activa
      - RA (verde, eje secundario) — absorción de carbohidratos
      - Eventos bolus (triángulos rojos) — si los hay
      - Eventos carbs (triángulos naranjas) — si los hay

    Ver si la reconstrucción sigue la forma correcta Y si responde
    a los drivers (PI alto → bajada CGM reconstruida, RA alto → subida).
    """
    N = len(windows_orig)
    cgm   = windows_orig[:, :, CGM_IDX]
    pi    = windows_orig[:, :, PI_IDX]
    ra    = windows_orig[:, :, RA_IDX]
    bolus = windows_orig[:, :, 5]   # bolus_logged
    carbs = windows_orig[:, :, 6]   # carbs_logged

    # Calcular variabilidad de CGM en la zona enmascarada por ventana
    cgm_range_in_mask = []
    for i in range(N):
        m = masks[i].astype(bool)
        if m.sum() > 0:
            cgm_range_in_mask.append(cgm[i][m].max() - cgm[i][m].min())
        else:
            cgm_range_in_mask.append(0.0)
    cgm_range_in_mask = np.array(cgm_range_in_mask)

    # Seleccionar 2 ejemplos "planos" (baja variabilidad en zona enmascarada)
    # y 2 ejemplos "dinámicos" (alta variabilidad — pico postprandial probable)
    pct_low  = np.percentile(cgm_range_in_mask, 20)
    pct_high = np.percentile(cgm_range_in_mask, 80)

    flat_candidates    = np.where(cgm_range_in_mask <= pct_low)[0]
    dynamic_candidates = np.where(cgm_range_in_mask >= pct_high)[0]

    np.random.shuffle(flat_candidates)
    np.random.shuffle(dynamic_candidates)

    selected = list(flat_candidates[:2]) + list(dynamic_candidates[:2])
    labels   = ['Basal (flat)', 'Basal (flat)', 'Dynamic (peak)', 'Dynamic (peak)']

    # Generar predicciones: aplicar mismo masking y predecir
    x_masked_sel = windows_orig[selected].copy()
    for j, idx in enumerate(selected):
        m = masks[idx].astype(bool)
        x_masked_sel[j, m, CGM_IDX] = MASK_TOKEN

    y_pred = model.predict(x_masked_sel, verbose=0)   # (4, 288)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 5 * 4))
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.6)
    t   = np.arange(WINDOW_LEN)

    for row, (idx, label) in enumerate(zip(selected, labels)):
        ax      = fig.add_subplot(gs[row])
        ax_drv  = ax.twinx()   # eje secundario para PI y RA

        m = masks[idx].astype(bool)

        # Fondo gris en la zona enmascarada
        mask_starts = np.where(np.diff(m.astype(int)) == 1)[0] + 1
        mask_ends   = np.where(np.diff(m.astype(int)) == -1)[0] + 1
        if m[0]:   mask_starts = np.insert(mask_starts, 0, 0)
        if m[-1]:  mask_ends   = np.append(mask_ends, WINDOW_LEN)
        for s, e in zip(mask_starts, mask_ends):
            ax.axvspan(s, e, alpha=0.12, color='#9CA3AF', label='_nolegend_')

        # CGM real
        ax.plot(t, cgm[idx], color=COLORS['cgm_real'], lw=2, label='CGM real', zorder=5)

        # Reconstrucción (solo en zona enmascarada)
        recon = np.full(WINDOW_LEN, np.nan)
        recon[m] = y_pred[row][m]
        ax.plot(t, recon, color=COLORS['recon'], lw=2.5, ls='--',
                label='Reconstrucción', zorder=6)

        # PI y RA en eje secundario
        ax_drv.plot(t, pi[idx], color=COLORS['pi'],  lw=1.2, alpha=0.7, label='PI')
        ax_drv.plot(t, ra[idx], color=COLORS['ra'],  lw=1.2, alpha=0.7, label='RA')

        # Eventos bolus y carbs (triángulos en la base del plot)
        bolus_t = t[bolus[idx] > 0]
        carbs_t = t[carbs[idx] > 0]
        y_min   = ax.get_ylim()[0] if ax.get_ylim()[0] != ax.get_ylim()[1] else -2.0
        if len(bolus_t) > 0:
            ax.scatter(bolus_t, [cgm[idx].min() - 0.15] * len(bolus_t),
                       marker='^', color=COLORS['bolus'], s=40, zorder=7,
                       label='Bolus event')
        if len(carbs_t) > 0:
            ax.scatter(carbs_t, [cgm[idx].min() - 0.3] * len(carbs_t),
                       marker='^', color=COLORS['carbs'], s=40, zorder=7,
                       label='Carbs event')

        # Leyendas y formato
        ax.set_ylabel('CGM (z-score)', fontsize=9)
        ax_drv.set_ylabel('PI / RA (z-score)', fontsize=9, color='#6B7280')
        ax.set_xlabel('Timestep (5 min)', fontsize=9)

        n_masked = m.sum()
        ax.set_title(
            f'{label}  —  window idx={idx}  '
            f'({n_masked} steps = {n_masked*5} min enmascarados  |  '
            f'ΔCGM en zona: {cgm_range_in_mask[idx]:.2f} z-score)',
            fontsize=10
        )

        # Leyenda combinada
        lines_ax  = ax.get_lines()
        lines_drv = ax_drv.get_lines()
        all_lines  = [l for l in lines_ax  if not l.get_label().startswith('_')]
        all_lines += [l for l in lines_drv if not l.get_label().startswith('_')]
        ax.legend(all_lines, [l.get_label() for l in all_lines],
                  fontsize=8, loc='upper right', ncol=3)

        ax.grid(True, alpha=0.2)
        ax.spines[['top']].set_visible(False)
        ax_drv.spines[['top']].set_visible(False)

        # Ticks cada 4h
        step_t = 48
        ticks  = list(range(0, WINDOW_LEN + 1, step_t))
        labels_t = [f'{t*5//60}h' for t in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels_t, fontsize=8)

    fig.suptitle(
        f'MTSM Reconstruction Examples\n'
        f'Gris = zona enmascarada (CGM oculto al modelo)  |  '
        f'Azul = reconstrucción  |  PI y RA siempre visibles',
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
    print(f"  Pacientes: {len(npz_files)}   Ventanas: {windows.shape[0]:,}   Shape: {windows.shape}")
    return windows


# ── Debug ─────────────────────────────────────────────────────────────────────

def debug_forward_pass(model, encoder, X_sample: np.ndarray):
    print(f"\n{'='*62}")
    print(f"  DEBUG — MTSM forward pass  (batch_size=1)")
    print(f"{'='*62}")

    def fmt(t, name):
        a = t.numpy() if hasattr(t, 'numpy') else np.array(t)
        print(f"  {name:<26} shape={str(a.shape):<22} "
              f"μ={a.mean():+.4f}  σ={a.std():.4f}  "
              f"[{a.min():+.5f}, {a.max():+.5f}]")

    x = tf.cast(X_sample[:1], tf.float32)
    fmt(x, 'x_masked (input)')
    print(f"  CGM en timesteps enmascarados = {MASK_TOKEN} (token fijo)")
    print(f"  PI, RA, drivers — intactos, siempre visibles")

    x_proj = encoder.get_layer('input_proj')(x, training=False)
    fmt(x_proj, 'x_proj (Dense→d_model)')

    pe   = get_positional_encoding(WINDOW_LEN, D_MODEL)
    x_pe = x_proj + pe
    fmt(x_pe, 'x_proj + PE')

    x_enc = x_pe
    for i in range(N_LAYERS):
        attn_out = encoder.get_layer(f'mhsa_{i}')(x_enc, x_enc, training=False)
        x_enc    = encoder.get_layer(f'norm1_{i}')(x_enc + attn_out, training=False)
        ffn_out  = encoder.get_layer(f'ffn2_{i}')(
                       encoder.get_layer(f'ffn1_{i}')(x_enc, training=False),
                       training=False)
        x_enc    = encoder.get_layer(f'norm2_{i}')(x_enc + ffn_out, training=False)
        fmt(x_enc, f'H after layer {i}')

    fmt(x_enc, 'H_final')
    print(f"\n  H_final: {WINDOW_LEN} timesteps × {D_MODEL} dims")
    print(f"  En los timesteps enmascarados, H_t codifica lo que el encoder")
    print(f"  infiere del contexto (pasado + futuro + drivers) sobre el CGM faltante")

    head = model.get_layer('reconstruction_head')
    y    = head(x_enc)
    fmt(y, 'ŷ_CGM (reconstruction head)')
    print(f"  Dense(1) aplicado a cada H_t → scalar CGM reconstruido")
    print(f"  Loss se calcula SOLO sobre los timesteps enmascarados")
    print(f"{'='*62}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

# Variable global necesaria para apply_mask_batch
window_len = WINDOW_LEN


def main(processed_dir: str, epochs: int, max_patients: int = None,
         mask_ratio: float = MASK_RATIO, debug: bool = False):

    from datetime import datetime
    run_id = datetime.now().strftime('%Y%m%d_%H%M')
    global RESULTS_DIR
    RESULTS_DIR = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"  Run ID: {run_id}")

    print(f"\n{'='*52}")
    print(f"  MTSM — Masked Time Series Modelling")
    print(f"  Stage 1 Pre-training Experiment")
    print(f"{'='*52}")

    # ── 1. Cargar datos ──
    print(f"\n  Cargando datos desde: {processed_dir}")
    windows = load_windows(processed_dir, max_patients)

    # ── 2. Preparar datos con masking ──
    print(f"\n{'='*52}")
    print(f"  Preparando datos")
    print(f"{'='*52}")
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), \
        windows_test_orig, masks_test = prepare_data(windows, mask_ratio)

    # ── 3. Construir modelo ──
    model, encoder = build_mtsm_model(
        WINDOW_LEN, N_FEATURES, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT
    )
    model.summary()

    if debug:
        # Aplicar masking a un ejemplo para el debug
        x_dbg, _, _ = apply_mask_batch(
            windows[:1], mask_ratio, MASK_MIN_LEN, MASK_MAX_LEN, MASK_TOKEN
        )
        debug_forward_pass(model, encoder, x_dbg)

    # ── 4. Entrenar ──
    global EPOCHS
    EPOCHS = epochs
    history = train_model(model, X_train, Y_train, X_val, Y_val)

    # ── 5. Plots ──
    print(f"\n{'='*52}")
    print(f"  Generando plots")
    print(f"{'='*52}")

    plot_training_curves(
        history,
        os.path.join(RESULTS_DIR, 'training_curves.png')
    )

    # Para el análisis de H usamos una ventana del test set sin masking
    # (queremos ver H del encoder en condiciones normales, no enmascaradas)
    variability = windows_test_orig[:, :, CGM_IDX].max(axis=1) - \
                  windows_test_orig[:, :, CGM_IDX].min(axis=1)
    h_idx = int(np.argmax(variability))

    # Pasar la ventana SIN masking al encoder para el análisis de H
    plot_H_analysis(
        encoder,
        windows_test_orig[h_idx:h_idx+1],
        os.path.join(RESULTS_DIR, 'transformer_H_analysis.png')
    )

    plot_reconstruction_examples(
        model,
        windows_test_orig,
        masks_test,
        os.path.join(RESULTS_DIR, 'reconstruction_examples.png')
    )

    print(f"\n{'='*52}")
    print(f"  Completado. Resultados en: {RESULTS_DIR}/")
    print(f"{'='*52}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTSM Stage 1 pre-training')
    parser.add_argument('--data',         type=str,   default='data/processed')
    parser.add_argument('--epochs',       type=int,   default=EPOCHS)
    parser.add_argument('--max_patients', type=int,   default=None)
    parser.add_argument('--mask_ratio',   type=float, default=MASK_RATIO,
                        help=f'Fraccion de timesteps a enmascarar (default: {MASK_RATIO})')
    parser.add_argument('--debug',        action='store_true')
    args = parser.parse_args()

    main(args.data, args.epochs,
         max_patients=args.max_patients,
         mask_ratio=args.mask_ratio,
         debug=args.debug)