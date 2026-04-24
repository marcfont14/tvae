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


# ── Transformer decoder — learned query cross-attention ───────────────────────

def build_forecasting_transformer(encoder: keras.Model,
                                   d_model: int = D_MODEL,
                                   n_horizons: int = N_HORIZONS) -> keras.Model:
    """
    FM/TS Transformer forecaster.
      H (288, d_model) → K, V
      n_horizons learned embeddings → Q
      CrossAttention(Q, K, V) → (n_horizons, d_model) → Dense → ŷ (n_horizons,)
    """
    x = keras.Input(shape=(288, 10), name='window')
    H = encoder(x)                                                   # (B, 288, d_model)

    # Learned horizon queries
    horizon_emb = layers.Embedding(n_horizons, d_model, name='horizon_emb')
    idx = tf.range(n_horizons)
    Q   = horizon_emb(idx)                                           # (n_horizons, d_model)
    Q   = tf.tile(tf.expand_dims(Q, 0), [tf.shape(H)[0], 1, 1])     # (B, n_horizons, d_model)

    context = layers.MultiHeadAttention(
        num_heads=4, key_dim=d_model // 4, name='cross_attn'
    )(query=Q, value=H, key=H)                                       # (B, n_horizons, d_model)

    out = layers.Dense(64, activation='relu', name='head_dense')(context)
    out = layers.Dense(1,  name='head_out')(out)                     # (B, n_horizons, 1)
    out = layers.Reshape((n_horizons,), name='output')(out)          # (B, n_horizons)

    return keras.Model(x, out, name='ForecastTransformer')


# ── LSTM decoder — attention-pooled H as initial hidden state ─────────────────

def build_forecasting_lstm(encoder: keras.Model,
                            d_model: int = D_MODEL,
                            n_horizons: int = N_HORIZONS) -> keras.Model:
    """
    FM/TS LSTM forecaster.
      AttentionPool(H) → h (d_model,) — LSTM initial hidden state
      6 learned horizon embeddings as input sequence
      LSTM(d_model, return_sequences=True) → (n_horizons, d_model) → Dense → ŷ (n_horizons,)
    """
    x = keras.Input(shape=(288, 10), name='window')
    H = encoder(x)                                                   # (B, 288, d_model)
    h = AttentionPool(d_model, name='attn_pool')(H)                  # (B, d_model)

    # Learned horizon embeddings as input to LSTM
    horizon_emb = layers.Embedding(n_horizons, 32, name='horizon_emb')
    idx = tf.range(n_horizons)
    Q   = horizon_emb(idx)                                           # (n_horizons, 32)
    Q   = tf.tile(tf.expand_dims(Q, 0), [tf.shape(H)[0], 1, 1])     # (B, n_horizons, 32)

    # h is initial hidden state; cell state initialised to zeros
    lstm_out = layers.LSTM(d_model, return_sequences=True, name='lstm')(
        Q, initial_state=[h, tf.zeros_like(h)]
    )                                                                # (B, n_horizons, d_model)

    out = layers.Dense(64, activation='relu', name='head_dense')(lstm_out)
    out = layers.Dense(1,  name='head_out')(out)                     # (B, n_horizons, 1)
    out = layers.Reshape((n_horizons,), name='output')(out)          # (B, n_horizons)

    return keras.Model(x, out, name='ForecastLSTM')


# ── Stubs — implemented as each app is built ──────────────────────────────────

def build_hypo_risk_model(encoder, d_model=D_MODEL):
    raise NotImplementedError


def build_isf_cr_model(encoder, d_model=D_MODEL):
    raise NotImplementedError


def build_digital_twin_model(encoder, d_model=D_MODEL, d_z=32):
    raise NotImplementedError


def build_tir_model(encoder, d_model=D_MODEL):
    raise NotImplementedError
