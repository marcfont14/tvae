import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

D_MODEL    = 128
N_HORIZONS = 6


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


# ── Hypo risk — Weibull survival head ────────────────────────────────────────

def build_hypo_risk_model(encoder: keras.Model,
                          d_model: int = D_MODEL) -> keras.Model:
    """
    FM / Raw nocturnal hypo-risk model.
      H, h_cls = encoder(window)
      h_cls → FC(64) → [log_lambda, log_k]  (Weibull survival params)
    Encoder trainability set by caller.
    """
    x = keras.Input(shape=(288, 10), name='window')
    H, h_cls = encoder(x)                                             # H unused here
    h = layers.Dense(64, activation='relu', name='head_dense')(h_cls)
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
