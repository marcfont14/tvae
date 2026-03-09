"""
experiment_forecasting.py
=========================
Baseline forecasting experiment: Transformer vs MLP  (v2)

Cambios respecto a v1:
  1. Target corregido: predecir los últimos FORECAST_HORIZON steps de la
     ventana desde dentro — interpolación, no extrapolación fuera de la ventana.
  2. Tabla arreglada: números formateados con anchura fija (round 4/2).
  3. Plot H analysis: attention weights, norma L2 de H_t, PCA de H.

Outputs (en results/):
    metrics_table.csv            RMSE / MAE por modelo
    metrics_table.png            tabla visual formateada
    training_curves.png          loss curves train/val
    forecast_examples.png        ejemplos visuales de prediccion
    transformer_H_analysis.png   attention weights + norma H_t + PCA

Usage:
    python scripts/experiment_forecasting.py
    python scripts/experiment_forecasting.py --max_patients 50 --epochs 20 --debug
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── Config ────────────────────────────────────────────────────────────────────

WINDOW_LEN       = 288
FORECAST_HORIZON = 12
CONTEXT_LEN      = 144   # 276→144 (12h): attention matrix 4x más pequeña (seq² escala cuadrático)
                          # 12h de contexto es suficiente para capturar un ciclo postprandial
                          # completo + patrones circadianos. El modelo final usará 288 steps
                          # pero para este experimento de validación 144 es adecuado.
N_FEATURES       = 10
CGM_IDX          = 0

# Features (sin modificación — igual que en el modelo final):
#   0=CGM  1=PI  2=RA  3=hour_sin  4=hour_cos
#   5=bolus_logged  6=carbs_logged  7=AID  8=SAP  9=MDI

# Transformer
D_MODEL  = 128   # suficiente para validar el encoder en este experimento
N_HEADS  = 4
N_LAYERS = 3
D_FF     = 256
DROPOUT  = 0.1

# MLP baseline
MLP_UNITS = 256

# Training
BATCH_SIZE = 128  # cabe bien con seq=144: 128×4×144×144 ~13MB/capa
EPOCHS     = 50
LR         = 1e-3
VAL_SPLIT  = 0.1
TEST_SPLIT  = 0.1

RESULTS_DIR = 'results'
SEED        = 42

COLORS = {
    'Transformer': '#2563EB',
    'MLP':         '#DC2626',
    'real':        '#111827',
    'context':     '#6B7280',
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


# ── Transformer ───────────────────────────────────────────────────────────────

def build_transformer(context_len, n_features, d_model, n_heads, n_layers,
                      d_ff, dropout, forecast_horizon):
    """
    Transformer encoder + linear forecast head.

    Flujo:
    ┌───────────────────────────────────────────────────────┐
    │ x_raw     (batch, 276, 10)  input crudo               │
    │ input_proj Dense(128)                                  │
    │ x_proj    (batch, 276, 128) cada timestep → R^d_model │
    │ + PE      (batch, 276, 128) añade info posicional      │
    │                                                        │
    │ × N_LAYERS encoder blocks:                             │
    │   MHSA(4 heads, key_dim=32) → residual + LayerNorm    │
    │   FFN(256→128)              → residual + LayerNorm    │
    │                                                        │
    │ H        (batch, 276, 128) representaciones finales    │
    │   H_t ∈ R^128 por timestep — valores reales,          │
    │   no probabilidades, sin restricción de rango          │
    │                                                        │
    │ GAP → h  (batch, 128)      resumen temporal            │
    │ head → ŷ (batch, 12)       CGM predicho               │
    └───────────────────────────────────────────────────────┘
    """
    inp = keras.Input(shape=(context_len, n_features), name='input')

    x  = layers.Dense(d_model, name='input_proj')(inp)
    pe = get_positional_encoding(context_len, d_model)
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

    x   = layers.GlobalAveragePooling1D(name='gap')(x)
    out = layers.Dense(forecast_horizon, name='forecast_head')(x)
    return keras.Model(inp, out, name='Transformer')


# ── MLP baseline ──────────────────────────────────────────────────────────────

def build_mlp_baseline(context_len, n_features, hidden_units, forecast_horizon):
    """
    MLP sin mecanismo temporal: aplana y predice con capas densas.
    Baseline para comparar: si el Transformer gana, la atención aporta algo real.
    """
    inp = keras.Input(shape=(context_len, n_features), name='input')
    x   = layers.Flatten()(inp)
    x   = layers.Dense(hidden_units, activation='relu')(x)
    x   = layers.Dropout(0.1)(x)
    x   = layers.Dense(hidden_units // 2, activation='relu')(x)
    x   = layers.Dropout(0.1)(x)
    out = layers.Dense(forecast_horizon, name='forecast_head')(x)
    return keras.Model(inp, out, name='MLP')


# ── Data ──────────────────────────────────────────────────────────────────────

def prepare_data(windows: np.ndarray):
    """
    X = windows[:, :CONTEXT_LEN, :]       → (N, 276, 10)  input 23h
    y = windows[:, CONTEXT_LEN:, CGM_IDX] → (N, 12)       target última 1h

    Target DENTRO de la ventana: el modelo aprende a completar la ventana,
    no a extrapolar fuera. Más fácil y más adecuado para validar el encoder.
    Sin ninguna modificación de las features — igual que en el modelo final.
    """

    # Con CONTEXT_LEN=144 y FORECAST_HORIZON=12:
    # Tomamos steps [132:276] como input y [276:288] como target
    # — el target sigue siendo la última hora de la ventana,
    # y el input son las 12h inmediatamente anteriores.
    start = WINDOW_LEN - CONTEXT_LEN - FORECAST_HORIZON  # 288-144-12 = 132
    X = windows[:, start:start + CONTEXT_LEN, :]          # (N, 144, 10)
    y = windows[:, start + CONTEXT_LEN:, CGM_IDX]         # (N, 12)

    assert y.shape[1] == FORECAST_HORIZON

    n   = len(X)
    idx = np.random.permutation(n)
    n_test = int(n * TEST_SPLIT)
    n_val  = int(n * VAL_SPLIT)

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    print(f"  Input:  steps [0:{CONTEXT_LEN}]  ({CONTEXT_LEN*5//60}h {CONTEXT_LEN*5%60}min)")
    print(f"  Target: steps [{CONTEXT_LEN}:{CONTEXT_LEN+FORECAST_HORIZON}]  ({FORECAST_HORIZON*5} min) — dentro de la ventana")
    print(f"  Train: {len(train_idx):>7,} ventanas")
    print(f"  Val:   {len(val_idx):>7,} ventanas")
    print(f"  Test:  {len(test_idx):>7,} ventanas")

    return (
        (X[train_idx], y[train_idx]),
        (X[val_idx],   y[val_idx]),
        (X[test_idx],  y[test_idx]),
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(model, X_train, y_train, X_val, y_val, model_name):
    print(f"\n{'─'*50}")
    print(f"  Entrenando: {model_name}  ({model.count_params():,} params)")
    print(f"{'─'*50}")

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LR, weight_decay=1e-4),
        loss='mse',
        metrics=['mae']
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
    ]
    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name, cgm_std=50.0):
    y_pred = model.predict(X_test, verbose=0)
    rmse   = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    mae    = float(np.mean(np.abs(y_test - y_pred)))

    results = {
        'Model':          model_name,
        'RMSE (z-score)': round(rmse, 4),
        'MAE (z-score)':  round(mae,  4),
        'RMSE (mg/dL)':   round(rmse * cgm_std, 2),
        'MAE (mg/dL)':    round(mae  * cgm_std, 2),
    }
    print(f"  {model_name:<14} RMSE={rmse:.4f}  MAE={mae:.4f}  "
          f"({rmse*cgm_std:.1f} / {mae*cgm_std:.1f} mg/dL)")
    return results, y_pred


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_training_curves(histories: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Curves — Transformer vs MLP', fontsize=14, fontweight='bold')

    for ax, (metric, label) in zip(axes, [('loss', 'MSE Loss'), ('mae', 'MAE')]):
        for name, hist in histories.items():
            c = COLORS[name]
            ax.plot(hist.history[metric],          color=c, lw=2,   ls='-',  label=f'{name} train')
            ax.plot(hist.history[f'val_{metric}'], color=c, lw=1.5, ls='--', label=f'{name} val')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {save_path}")


def plot_forecast_examples(X_test, y_test, predictions: dict,
                           n_examples: int, save_path: str):
    cgm_ctx = X_test[:, :, CGM_IDX]
    ranges  = cgm_ctx.max(axis=1) - cgm_ctx.min(axis=1)
    pcts    = np.percentile(ranges, np.linspace(10, 90, n_examples))
    idxs    = [np.argmin(np.abs(ranges - p)) for p in pcts]

    fig = plt.figure(figsize=(16, 4 * n_examples))
    gs  = gridspec.GridSpec(n_examples, 1, hspace=0.55)

    t_ctx  = np.arange(CONTEXT_LEN)
    t_fcst = np.arange(CONTEXT_LEN, CONTEXT_LEN + FORECAST_HORIZON)

    for row, idx in enumerate(idxs):
        ax = fig.add_subplot(gs[row])
        ax.plot(t_ctx,  cgm_ctx[idx],  color=COLORS['context'], lw=1.5, label='Context (input)')
        ax.plot(t_fcst, y_test[idx],   color=COLORS['real'],    lw=2.5, label='Real CGM', zorder=5)
        for name, preds in predictions.items():
            ax.plot(t_fcst, preds[idx], color=COLORS[name], lw=2, ls='--',
                    label=f'{name} forecast', zorder=4)

        ax.axvline(x=CONTEXT_LEN - 0.5, color='#9CA3AF', ls=':', lw=1.5)
        ax.set_ylabel('CGM (z-score)', fontsize=10)
        ax.set_xlabel('Timestep (5 min)', fontsize=10)
        ax.set_title(f'Example {row+1}  (window idx={idx})', fontsize=11)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)

        step   = 24
        ticks  = list(range(0, CONTEXT_LEN + FORECAST_HORIZON + 1, step))
        labels = [f'{t*5//60}h{t*5%60:02d}' for t in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)

    fig.suptitle(
        f'Forecast Examples — Transformer vs MLP\n'
        f'Context: {CONTEXT_LEN} steps ({CONTEXT_LEN*5//60}h)  →  '
        f'Target: last {FORECAST_HORIZON} steps ({FORECAST_HORIZON*5} min, inside window)',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {save_path}")


def plot_metrics_table(metrics_list: list, save_path: str):
    """Tabla visual con números bien formateados."""
    df = pd.DataFrame(metrics_list).set_index('Model')

    fmt_map = {
        'RMSE (z-score)': '{:.4f}',
        'MAE (z-score)':  '{:.4f}',
        'RMSE (mg/dL)':   '{:.2f}',
        'MAE (mg/dL)':    '{:.2f}',
    }
    df_str = df.copy().astype(str)
    for col, f in fmt_map.items():
        if col in df.columns:
            df_str[col] = df[col].apply(lambda v: f.format(float(v)))

    fig, ax = plt.subplots(figsize=(10, 2.5 + 0.5 * len(metrics_list)))
    ax.axis('off')

    col_labels = list(df_str.columns)
    row_labels  = list(df_str.index)
    cell_text   = df_str.values.tolist()

    table = ax.table(
        cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
        cellLoc='center', rowLoc='center', loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.4, 2.2)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#1E3A5F')
        table[0, j].set_text_props(color='white', fontweight='bold')

    row_colors = {'Transformer': '#EFF6FF', 'MLP': '#FEF2F2'}
    for i, rn in enumerate(row_labels):
        c = row_colors.get(rn, '#F9FAFB')
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(c)

    plt.title(
        f'Test Set Metrics\n'
        f'Input: {CONTEXT_LEN} steps ({CONTEXT_LEN*5//60}h)  →  '
        f'Target: {FORECAST_HORIZON} steps ({FORECAST_HORIZON*5} min, inside window)',
        fontsize=12, fontweight='bold', pad=20
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {save_path}")


def plot_H_analysis(transformer, X_sample: np.ndarray, save_path: str):
    """
    Visualiza las representaciones internas H del Transformer encoder.

    Panel 1 — Attention weights (última capa MHSA, media de heads):
        Matriz (seq × seq): fila i = timestep query, columna j = timestep key.
        Celda (i,j) = cuánto atiende el timestep i al timestep j.
        Patrón diagonal → atención local. Patrón disperso → contexto global.

    Panel 2 — Norma L2 de H_t vs CGM:
        ||H_t||₂ para t=0..275 superpuesto con la señal CGM.
        Norma alta = representación densa = timestep "informativo" para el encoder.
        Si covaría con picos postprandiales → el encoder los detecta.

    Panel 3 — PCA de H ∈ R^(276×128) → R²:
        Cada punto = un timestep, color = valor CGM en ese timestep.
        Si hay gradiente de color ordenado en el espacio PCA, el encoder
        organiza las representaciones por estado glucémico.
    """
    from sklearn.decomposition import PCA

    print(f"  Analizando H (ventana con máxima variabilidad de CGM)...")

    x_in = tf.cast(X_sample[:1], tf.float32)   # (1, 276, 10)

    # Modelo intermedio: devuelve H_final (input al GAP)
    H_model = keras.Model(
        inputs=transformer.input,
        outputs=transformer.get_layer('gap').input
    )
    H = H_model(x_in, training=False).numpy()[0]   # (276, 128)

    # Modelo intermedio: devuelve representación antes de la última MHSA
    if N_LAYERS > 1:
        pre_layer = transformer.get_layer(f'norm2_{N_LAYERS-2}')
    else:
        pre_layer = transformer.get_layer('input_proj')
    pre_model = keras.Model(inputs=transformer.input, outputs=pre_layer.output)
    x_pre = pre_model(x_in, training=False)

    # Attention scores de la última capa
    last_mhsa = transformer.get_layer(f'mhsa_{N_LAYERS-1}')
    _, attn_scores = last_mhsa(x_pre, x_pre, return_attention_scores=True, training=False)
    attn_mean = attn_scores[0].numpy().mean(axis=0)   # (276, 276) media de heads

    cgm   = X_sample[0, :, CGM_IDX]                  # (276,)
    H_norm = np.linalg.norm(H, axis=1)               # (276,)

    pca  = PCA(n_components=2)
    H_2d = pca.fit_transform(H)                      # (276, 2)
    var  = pca.explained_variance_ratio_

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.35)

    # Panel 1: Attention heatmap (subsampled)
    ax1  = fig.add_subplot(gs[0, :])
    sstp = max(1, CONTEXT_LEN // 64)                 # subsample para legibilidad
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
    t = np.arange(CONTEXT_LEN)
    ax2.plot(t, H_norm, color='#7C3AED', lw=1.5, alpha=0.9, label='‖H_t‖₂')
    ax2_twin.plot(t, cgm, color=COLORS['context'], lw=1.2, alpha=0.6, label='CGM (z-score)')
    ax2.set_xlabel('Timestep (5 min)', fontsize=9)
    ax2.set_ylabel('‖H_t‖₂', fontsize=9, color='#7C3AED')
    ax2_twin.set_ylabel('CGM (z-score)', fontsize=9, color=COLORS['context'])
    ax2.set_title(
        'Panel 2 — Norma L2 de H_t vs CGM\n'
        '‖H_t‖₂ alta → timestep rico en información para el encoder\n'
        '¿Covaría con picos postprandiales?',
        fontsize=10
    )
    lines  = ax2.get_lines() + ax2_twin.get_lines()
    ax2.legend(lines, [l.get_label() for l in lines], fontsize=9)
    ax2.grid(True, alpha=0.2)
    ax2.spines[['top']].set_visible(False)
    step_t = 48
    ax2.set_xticks(range(0, CONTEXT_LEN + 1, step_t))
    ax2.set_xticklabels([f'{t*5//60}h' for t in range(0, CONTEXT_LEN + 1, step_t)], fontsize=8)

    # Panel 3: PCA
    ax3 = fig.add_subplot(gs[1, 1])
    sc  = ax3.scatter(H_2d[:, 0], H_2d[:, 1], c=cgm, cmap='RdYlGn_r', s=10, alpha=0.7)
    plt.colorbar(sc, ax=ax3, label='CGM (z-score)')
    ax3.set_xlabel(f'PC1 ({var[0]*100:.1f}% var)', fontsize=9)
    ax3.set_ylabel(f'PC2 ({var[1]*100:.1f}% var)', fontsize=9)
    ax3.set_title(
        'Panel 3 — PCA de H ∈ ℝ^(276×128) → ℝ²\n'
        'Cada punto = un timestep · color = valor CGM\n'
        'Gradiente ordenado → encoder organiza por estado glucémico',
        fontsize=10
    )
    ax3.grid(True, alpha=0.2)
    ax3.spines[['top', 'right']].set_visible(False)

    fig.suptitle(
        f'Análisis de H — Representaciones del Transformer Encoder\n'
        f'H ∈ ℝ^({CONTEXT_LEN}×{D_MODEL})   |   {N_LAYERS} capas · {N_HEADS} heads · d_model={D_MODEL}',
        fontsize=13, fontweight='bold'
    )
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {save_path}")


# ── Debug: forward pass paso a paso ──────────────────────────────────────────

def debug_forward_pass(transformer, X_sample: np.ndarray):
    print(f"\n{'='*62}")
    print(f"  DEBUG — Forward pass paso a paso  (batch_size=1)")
    print(f"{'='*62}")

    def fmt(t, name):
        a = t.numpy() if hasattr(t, 'numpy') else np.array(t)
        print(f"  {name:<24} shape={str(a.shape):<22} "
              f"μ={a.mean():+.4f}  σ={a.std():.4f}  "
              f"[{a.min():+.5f}, {a.max():+.5f}]")

    x = tf.cast(X_sample[:1], tf.float32)
    fmt(x, 'x_raw')

    x_proj = transformer.get_layer('input_proj')(x, training=False)
    fmt(x_proj, 'x_proj  (Dense→128)')

    pe   = get_positional_encoding(CONTEXT_LEN, D_MODEL)
    x_pe = x_proj + pe
    fmt(x_pe, 'x_proj + PE')

    x_enc = x_pe
    for i in range(N_LAYERS):
        attn_out = transformer.get_layer(f'mhsa_{i}')(x_enc, x_enc, training=False)
        x_enc    = transformer.get_layer(f'norm1_{i}')(x_enc + attn_out, training=False)
        ffn_out  = transformer.get_layer(f'ffn2_{i}')(
                       transformer.get_layer(f'ffn1_{i}')(x_enc, training=False),
                       training=False)
        x_enc    = transformer.get_layer(f'norm2_{i}')(x_enc + ffn_out, training=False)
        fmt(x_enc, f'H  after layer {i}')

    print(f"\n  H_final: shape {x_enc.shape}  →  {CONTEXT_LEN} timesteps × {D_MODEL} dims")
    print(f"  H_t[0]:  {x_enc[0, 0, :8].numpy().round(4)} ...")
    print(f"  Valores reales sin restricción de rango — NO son probabilidades")

    h_gap = transformer.get_layer('gap')(x_enc)
    fmt(h_gap, 'h_pooled  (GAP)')
    print(f"  GAP colapsa ({CONTEXT_LEN}, {D_MODEL}) → ({D_MODEL},) vía media temporal")

    y_hat = transformer.get_layer('forecast_head')(h_gap)
    fmt(y_hat, 'ŷ  (forecast head)')
    print(f"  ŷ = {y_hat[0].numpy().round(4)}")
    print(f"{'='*62}\n")


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main(processed_dir: str, epochs: int, max_patients: int = None, debug: bool = False):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'='*52}")
    print(f"  Cargando datos desde: {processed_dir}")
    print(f"{'='*52}")
    windows = load_windows(processed_dir, max_patients)

    print(f"\n{'='*52}")
    print(f"  Preparando datos")
    print(f"{'='*52}")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data(windows)

    transformer = build_transformer(
        CONTEXT_LEN, N_FEATURES, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT, FORECAST_HORIZON
    )
    mlp = build_mlp_baseline(CONTEXT_LEN, N_FEATURES, MLP_UNITS, FORECAST_HORIZON)
    transformer.summary()
    mlp.summary()

    if debug:
        debug_forward_pass(transformer, X_train[:1])

    global EPOCHS
    EPOCHS = epochs

    histories    = {}
    all_metrics  = []
    all_preds    = {}

    for model_name, model in [('Transformer', transformer), ('MLP', mlp)]:
        ckpt_path = os.path.join(RESULTS_DIR, f'{model_name}_best.keras')

        try:
            if os.path.exists(ckpt_path):
                # Resume: cargar pesos del mejor checkpoint
                print(f"\n  Cargando checkpoint: {ckpt_path}")
                model.compile(
                    optimizer=keras.optimizers.AdamW(learning_rate=LR, weight_decay=1e-4),
                    loss='mse', metrics=['mae']
                )
                model.load_weights(ckpt_path)
                histories[model_name] = None   # no hay history para plot
            else:
                histories[model_name] = train_model(
                    model, X_train, y_train, X_val, y_val, model_name
                )

            # Evaluar y guardar métricas inmediatamente
            m, p = evaluate_model(model, X_test, y_test, model_name)
            all_metrics.append(m)
            all_preds[model_name] = p

            # CSV parcial — si el siguiente modelo peta, este ya está guardado
            pd.DataFrame(all_metrics).set_index('Model').to_csv(
                os.path.join(RESULTS_DIR, 'metrics_table.csv')
            )
            print(f"  ✓ {model_name} completado y guardado")

        except Exception as e:
            print(f"\n  AVISO: {model_name} falló — {e}")
            print(f"  Continuando con los modelos completados hasta ahora...")
            continue

    if not all_metrics:
        print("  No hay resultados que guardar.")
        return

    print(f"\n{'='*52}")
    print(f"  Generando plots")
    print(f"{'='*52}")

    valid_histories = {k: v for k, v in histories.items() if v is not None}
    if valid_histories:
        plot_training_curves(valid_histories,
                             os.path.join(RESULTS_DIR, 'training_curves.png'))

    plot_forecast_examples(X_test, y_test, all_preds, n_examples=4,
                           save_path=os.path.join(RESULTS_DIR, 'forecast_examples.png'))
    plot_metrics_table(all_metrics,
                       os.path.join(RESULTS_DIR, 'metrics_table.png'))

    variability = X_test[:, :, CGM_IDX].max(axis=1) - X_test[:, :, CGM_IDX].min(axis=1)
    h_idx       = int(np.argmax(variability))
    plot_H_analysis(transformer, X_test[h_idx:h_idx+1],
                    save_path=os.path.join(RESULTS_DIR, 'transformer_H_analysis.png'))

    print(f"\n{'='*52}")
    print(f"  Completado. Resultados en: {RESULTS_DIR}/")
    print(f"{'='*52}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Forecasting: Transformer vs MLP (v2)')
    parser.add_argument('--data',         type=str, default='data/processed')
    parser.add_argument('--epochs',       type=int, default=EPOCHS)
    parser.add_argument('--max_patients', type=int, default=None)
    parser.add_argument('--debug',        action='store_true')
    args = parser.parse_args()
    main(args.data, args.epochs, max_patients=args.max_patients, debug=args.debug)