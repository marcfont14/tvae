import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

D_MODEL    = 128
N_HORIZONS = 24


# ── Shared utility ─────────────────────────────────────────────────────────────

class AttentionPool(layers.Layer):
    """Learnable attention pooling: H (B, T, d) → h (B, d)."""
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.w = layers.Dense(1)

    def call(self, H):
        scores = tf.nn.softmax(self.w(H), axis=1)   # (B, T, 1)
        return tf.reduce_sum(scores * H, axis=1)     # (B, d)


class QueryCrossAttention(layers.Layer):
    """Single learned query token attending over H: (B, T, d) → (B, d)."""
    def __init__(self, d_model=D_MODEL, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.q_token = self.add_weight(
            shape=(1, 1, d_model), initializer='glorot_uniform', trainable=True
        )
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )

    def call(self, H, training=False):
        Q   = tf.tile(self.q_token, [tf.shape(H)[0], 1, 1])         # (B, 1, d)
        out = self.attn(query=Q, key=H, value=H, training=training)  # (B, 1, d)
        return tf.squeeze(out, axis=1)                                # (B, d)


# ── LSTM forecaster — attention-pooled H as initial hidden state ──────────────

def build_forecasting_lstm(encoder: keras.Model,
                            d_model: int = D_MODEL,
                            n_horizons: int = N_HORIZONS) -> keras.Model:
    """
    FM / Raw LSTM forecaster.
      H, h_cls = encoder(window)
      AttentionPool(H) → LSTM initial hidden state
      LSTM decoder predicts delta from last observed CGM
      output = delta + last_cgm  (skip connection guarantees continuity)
    """
    x        = keras.Input(shape=(288, 10), name='window')
    last_cgm = keras.Input(shape=(1,),      name='last_cgm')

    H, h_cls = encoder(x)                                             # unpack encoder outputs
    h = AttentionPool(d_model, name='attn_pool')(H)                   # (B, d_model)

    horizon_emb = layers.Embedding(n_horizons, 32, name='horizon_emb')
    idx = tf.range(n_horizons)
    Q   = horizon_emb(idx)
    Q   = tf.tile(tf.expand_dims(Q, 0), [tf.shape(H)[0], 1, 1])      # (B, n_horizons, 32)

    lstm_out = layers.LSTM(d_model, return_sequences=True, unroll=True, name='lstm')(
        Q, initial_state=[h, tf.zeros_like(h)]
    )                                                                  # (B, n_horizons, d_model)

    delta = layers.Dense(64, activation='relu', name='head_dense')(lstm_out)
    delta = layers.Dense(1,  name='head_out')(delta)
    delta = layers.Reshape((n_horizons,), name='delta')(delta)

    anchor = layers.Lambda(
        lambda lc, nh=n_horizons: tf.tile(lc, [1, nh]), name='anchor'
    )(last_cgm)
    out = layers.Add(name='output')([delta, anchor])

    return keras.Model([x, last_cgm], out, name='ForecastLSTM')


def build_raw_forecasting_lstm(d_model: int = D_MODEL,
                                n_horizons: int = N_HORIZONS) -> keras.Model:
    """
    Raw LSTM forecaster — no encoder.
    Input (288, 10) → Conv1D(stride=4) → (72, 128) → LSTM(128) → h → same 6-step decoder.
    Conv1D downsamples 288→72 steps; unroll=True mandatory (cuDNN 9.0 requires sequence lengths TF 2.17 doesn't provide).
    """
    x        = keras.Input(shape=(288, 10), name='window')
    last_cgm = keras.Input(shape=(1,),      name='last_cgm')

    z = layers.Conv1D(d_model, kernel_size=4, strides=4, padding='same',
                      activation='relu', name='downsample')(x)          # (B, 72, 128)
    h = layers.LSTM(d_model, return_sequences=False, unroll=True, name='enc_lstm')(z)

    horizon_emb = layers.Embedding(n_horizons, 32, name='horizon_emb')
    idx = tf.range(n_horizons)
    Q_h = horizon_emb(idx)
    Q_h = tf.tile(tf.expand_dims(Q_h, 0), [tf.shape(x)[0], 1, 1])   # (B, n_horizons, 32)

    lstm_out = layers.LSTM(d_model, return_sequences=True, unroll=True, name='dec_lstm')(
        Q_h, initial_state=[h, tf.zeros_like(h)]
    )

    delta = layers.Dense(64, activation='relu', name='head_dense')(lstm_out)
    delta = layers.Dense(1,  name='head_out')(delta)
    delta = layers.Reshape((n_horizons,), name='delta')(delta)

    anchor = layers.Lambda(
        lambda lc, nh=n_horizons: tf.tile(lc, [1, nh]), name='anchor'
    )(last_cgm)
    out = layers.Add(name='output')([delta, anchor])

    return keras.Model([x, last_cgm], out, name='RawForecastLSTM')


def build_forecasting_lstm_hcls(encoder: keras.Model,
                                 d_model: int = D_MODEL,
                                 n_horizons: int = N_HORIZONS) -> keras.Model:
    """
    FM LSTM forecaster using h_cls directly as decoder initial state.
    No pooling over H — h_cls is the CLS token, pre-trained via causal aux loss
    to predict the last 6 CGM steps of the window (closest proxy to our task).
    """
    x        = keras.Input(shape=(288, 10), name='window')
    last_cgm = keras.Input(shape=(1,),      name='last_cgm')

    H, h_cls = encoder(x)
    h = h_cls                                                            # (B, d_model)

    horizon_emb = layers.Embedding(n_horizons, 32, name='horizon_emb')
    idx = tf.range(n_horizons)
    Q   = horizon_emb(idx)
    Q   = tf.tile(tf.expand_dims(Q, 0), [tf.shape(H)[0], 1, 1])

    lstm_out = layers.LSTM(d_model, return_sequences=True, unroll=True, name='lstm')(
        Q, initial_state=[h, tf.zeros_like(h)]
    )

    delta = layers.Dense(64, activation='relu', name='head_dense')(lstm_out)
    delta = layers.Dense(1,  name='head_out')(delta)
    delta = layers.Reshape((n_horizons,), name='delta')(delta)

    anchor = layers.Lambda(
        lambda lc, nh=n_horizons: tf.tile(lc, [1, nh]), name='anchor'
    )(last_cgm)
    out = layers.Add(name='output')([delta, anchor])

    return keras.Model([x, last_cgm], out, name='ForecastLSTM_hcls')


def build_forecasting_lstm_query(encoder: keras.Model,
                                  d_model: int = D_MODEL,
                                  n_horizons: int = N_HORIZONS) -> keras.Model:
    """
    FM LSTM forecaster with QueryCrossAttention pooling.
      H, h_cls = encoder(window)
      QueryCrossAttention(H) → LSTM initial hidden state
      Identical decoder to build_forecasting_lstm — single variable changed.
    """
    x        = keras.Input(shape=(288, 10), name='window')
    last_cgm = keras.Input(shape=(1,),      name='last_cgm')

    H, h_cls = encoder(x)
    h = QueryCrossAttention(d_model, name='query_pool')(H)            # (B, d_model)

    horizon_emb = layers.Embedding(n_horizons, 32, name='horizon_emb')
    idx  = tf.range(n_horizons)
    Q_h  = horizon_emb(idx)
    Q_h  = tf.tile(tf.expand_dims(Q_h, 0), [tf.shape(H)[0], 1, 1])  # (B, n_horizons, 32)

    lstm_out = layers.LSTM(d_model, return_sequences=True, unroll=True, name='lstm')(
        Q_h, initial_state=[h, tf.zeros_like(h)]
    )                                                                  # (B, n_horizons, d_model)

    delta = layers.Dense(64, activation='relu', name='head_dense')(lstm_out)
    delta = layers.Dense(1,  name='head_out')(delta)
    delta = layers.Reshape((n_horizons,), name='delta')(delta)

    anchor = layers.Lambda(
        lambda lc, nh=n_horizons: tf.tile(lc, [1, nh]), name='anchor'
    )(last_cgm)
    out = layers.Add(name='output')([delta, anchor])

    return keras.Model([x, last_cgm], out, name='ForecastLSTM_Query')


# ── Decoder FM forecasters ───────────────────────────────────────────────────

def build_forecasting_lstm_decoder(encoder: keras.Model,
                                    d_model: int = D_MODEL,
                                    n_horizons: int = N_HORIZONS) -> keras.Model:
    """
    FM LSTM forecaster using h_last from causal decoder as initial hidden state.
    Identical structure to build_forecasting_lstm_hcls — only the encoder source differs.
    h_last = H[:, -1, :]: causally conditioned on the full 288-step window.
    """
    x        = keras.Input(shape=(288, 10), name='window')
    last_cgm = keras.Input(shape=(1,),      name='last_cgm')

    H, h_last = encoder(x)
    h = h_last

    horizon_emb = layers.Embedding(n_horizons, 32, name='horizon_emb')
    idx = tf.range(n_horizons)
    Q   = horizon_emb(idx)
    Q   = tf.tile(tf.expand_dims(Q, 0), [tf.shape(H)[0], 1, 1])

    lstm_out = layers.LSTM(d_model, return_sequences=True, unroll=True, name='lstm')(
        Q, initial_state=[h, tf.zeros_like(h)]
    )

    delta = layers.Dense(64, activation='relu', name='head_dense')(lstm_out)
    delta = layers.Dense(1,  name='head_out')(delta)
    delta = layers.Reshape((n_horizons,), name='delta')(delta)

    anchor = layers.Lambda(
        lambda lc, nh=n_horizons: tf.tile(lc, [1, nh]), name='anchor'
    )(last_cgm)
    out = layers.Add(name='output')([delta, anchor])

    return keras.Model([x, last_cgm], out, name='ForecastLSTM_decoder')


def predict_ar_decoder(encoder: keras.Model, ntp_head: keras.Model,
                        windows: np.ndarray, scaler_std: np.ndarray,
                        n_horizons: int = N_HORIZONS,
                        last_cgm_mg: np.ndarray = None,
                        chunk_size: int = 512,
                        delta_mode: bool = False) -> np.ndarray:
    """
    Autoregressive decoder rollout — zero additional trainable parameters.
    Processes windows in chunks to avoid materialising H (N,288,128) for the
    full test set at once (would be ~18 GB for 125k windows).

    windows:     (N, 288, 10) z-scored input
    scaler_std:  (N,) per-window CGM std in mg/dL
    last_cgm_mg: (N,) last observed CGM in mg/dL
    chunk_size:  windows per chunk — trades speed for memory
    delta_mode:  if True, ntp_head predicts Δ = CGM[t+1]−CGM[t] (decoder2)
    Returns: (N, n_horizons) predictions in mg/dL
    """
    import math
    if last_cgm_mg is None:
        raise ValueError('last_cgm_mg required for mg/dL conversion')
    N = len(windows)
    delta_angle = 2 * math.pi * (5.0 / 60.0) / 24.0
    cos_d = np.cos(delta_angle).astype(np.float32)
    sin_d = np.sin(delta_angle).astype(np.float32)

    preds_mg = np.zeros((N, n_horizons), dtype=np.float32)

    for start in range(0, N, chunk_size):
        end     = min(start + chunk_size, N)
        ctx     = windows[start:end].copy()          # (C, 288, 10)
        C       = end - start
        last_z  = ctx[:, -1, 0].copy()
        preds_z = np.zeros((C, n_horizons), dtype=np.float32)

        for k in range(n_horizons):
            H      = encoder.predict(ctx, batch_size=64, verbose=0)[0]  # (C, 288, 128)
            H_last = H[:, -1:, :]                                        # (C, 1, 128)
            out    = ntp_head.predict(H_last, verbose=0).reshape(C)      # (C,)
            pred_z = ctx[:, -1, 0] + out if delta_mode else out          # accumulate delta or use absolute
            preds_z[:, k] = pred_z

            pi_slope = (ctx[:, -1, 1] - ctx[:, -13, 1]) / 12.0
            ra_slope = (ctx[:, -1, 2] - ctx[:, -13, 2]) / 12.0
            pi_next  = ctx[:, -1, 1] + pi_slope
            ra_next  = ctx[:, -1, 2] + ra_slope

            hs = ctx[:, -1, 3] * cos_d + ctx[:, -1, 4] * sin_d
            hc = ctx[:, -1, 4] * cos_d - ctx[:, -1, 3] * sin_d

            new_step = np.stack([
                pred_z, pi_next, ra_next, hs, hc,
                np.zeros(C), np.zeros(C),
                ctx[:, -1, 7], ctx[:, -1, 8], ctx[:, -1, 9]
            ], axis=1)[:, np.newaxis, :]             # (C, 1, 10)

            ctx = np.concatenate([ctx[:, 1:, :], new_step], axis=1)

        preds_mg[start:end] = (
            last_cgm_mg[start:end, np.newaxis]
            + (preds_z - last_z[:, np.newaxis]) * scaler_std[start:end, np.newaxis]
        )

    return preds_mg


# ── Hypo risk — Weibull survival head ────────────────────────────────────────

def build_hypo_risk_decoder(encoder: keras.Model,
                            d_model: int = D_MODEL) -> keras.Model:
    """
    Nocturnal hypo-risk model using h_last from causal decoder.
    Identical structure to build_hypo_risk_model — only the encoder source differs.
    """
    x = keras.Input(shape=(288, 10), name='window')
    H, h_last = encoder(x)
    h   = layers.Dense(64, activation='relu', name='head_dense')(h_last)
    out = layers.Dense(2, name='weibull_params')(h)
    return keras.Model(x, out, name='HypoRiskDecoder')


def build_raw_hypo_risk_model(d_model: int = D_MODEL) -> keras.Model:
    """
    Raw hypo-risk baseline — no encoder.
    Window (288, 10) → Conv1D(stride=4) → (72, 128) → LSTM(128) → Dense(64) → [log_λ, log_k].
    Mirrors build_raw_forecasting_lstm structure for consistency.
    """
    x = keras.Input(shape=(288, 10), name='window')
    z = layers.Conv1D(d_model, kernel_size=4, strides=4, padding='same',
                      activation='relu', name='downsample')(x)        # (B, 72, 128)
    h = layers.LSTM(d_model, return_sequences=False, unroll=True, name='enc_lstm')(z)
    h = layers.Dense(64, activation='relu', name='head_dense')(h)
    out = layers.Dense(2, name='weibull_params')(h)
    return keras.Model(x, out, name='RawHypoRisk')


def build_hypo_risk_model(encoder: keras.Model,
                          d_model: int = D_MODEL) -> keras.Model:
    """
    FM nocturnal hypo-risk model (frozen or fine-tuned encoder).
      H, _ = encoder(window)
      AttentionPool(H) → FC(64) → [log_lambda, log_k]  (Weibull survival params)
    Encoder trainability set by caller (trainable=False → frozen, True → fine-tuned).
    """
    x = keras.Input(shape=(288, 10), name='window')
    H, _ = encoder(x)
    h = AttentionPool(d_model, name='attn_pool')(H)
    h = layers.Dense(64, activation='relu', name='head_dense')(h)
    out = layers.Dense(2, name='weibull_params')(h)                   # [log_lambda, log_k]
    return keras.Model(x, out, name='HypoRiskModel')


# ── Imputation — zero-shot MTSM reconstruction ───────────────────────────────

MTSM_MODEL_WEIGHTS = 'results/mtsm/encoder2/model_weights.weights.h5'


def build_raw_imputation_model() -> keras.Model:
    """
    Same reconstruction head as MTSM (Dense 64 → Dense 1 per timestep),
    applied directly to the raw (288, 10) window — no encoder.
    Trained from scratch for imputation.
    Isolates the encoder contribution: MTSM zero-shot vs head-only trained baseline.
    """
    from src.encoder import WINDOW_LEN, N_FEATURES
    x   = keras.Input(shape=(WINDOW_LEN, N_FEATURES), name='input')
    out = layers.Dense(64, activation='relu', name='recon_hidden')(x)
    out = layers.Dense(1,  name='reconstruction_head')(out)
    out = layers.Reshape((WINDOW_LEN,), name='output')(out)
    return keras.Model(x, out, name='RawImputation')


def build_mtsm_imputation_model(
        weights_path: str = MTSM_MODEL_WEIGHTS) -> keras.Model:
    """
    Rebuild full MTSM model (encoder + reconstruction head) and load pre-trained
    weights. Used for zero-shot imputation — no task-specific training.
    Input: window with CGM masked to 0 in gap → output: full (288,) CGM reconstruction.
    """
    from src.encoder import build_encoder, WINDOW_LEN, N_FEATURES
    enc = build_encoder()
    x   = keras.Input(shape=(WINDOW_LEN, N_FEATURES), name='input')
    H, h_cls = enc(x)                                                  # unpack
    out = layers.Dense(64, activation='relu', name='recon_hidden')(H)
    out = layers.Dense(1,  name='reconstruction_head')(out)
    out = layers.Reshape((WINDOW_LEN,), name='output')(out)
    model = keras.Model(x, out, name='MTSM')
    model(tf.zeros((1, WINDOW_LEN, N_FEATURES)))
    model.load_weights(weights_path)
    model.trainable = False
    return model


# ── Stubs ────────────────────────────────────────────────────────────────────

def build_isf_cr_model(encoder, d_model=D_MODEL):
    raise NotImplementedError


def build_digital_twin_model(encoder, d_model=D_MODEL, d_z=32):
    raise NotImplementedError
