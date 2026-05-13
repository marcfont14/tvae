import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

WINDOW_LEN = 288
N_FEATURES = 10   # --no_age: features 0-9
D_MODEL    = 128
N_HEADS    = 4
N_LAYERS   = 5
D_FF       = 256
DROPOUT    = 0.2

WEIGHTS_PATH        = 'results/mtsm/encoder_clean/encoder_weights.weights.h5'
WEIGHTS_PATH_E3     = 'results/mtsm/encoder3/encoder_weights.weights.h5'
WEIGHTS_PATH_DECODER  = 'results/mtsm/decoder_clean/encoder_weights.weights.h5'
NTP_MODEL_WEIGHTS     = 'results/mtsm/decoder/model_weights.weights.h5'
WEIGHTS_PATH_DECODER2 = 'results/mtsm/decoder2/encoder_weights.weights.h5'
NTP_HEAD2_WEIGHTS     = 'results/mtsm/decoder2/ntp_head_weights.weights.h5'
PREFIX_LEN           = 144   # steps 0-143 bidirectional, 144-287 causal+full-prefix


class CLSToken(layers.Layer):
    """Prepend a learned CLS token to the sequence: (B, T, d) → (B, T+1, d)."""
    def build(self, input_shape):
        self.cls = self.add_weight(shape=(1, 1, input_shape[-1]), name='cls_token')

    def call(self, x):
        cls = tf.tile(self.cls, [tf.shape(x)[0], 1, 1])
        return tf.concat([cls, x], axis=1)


def _positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    positions = np.arange(seq_len)[:, np.newaxis]
    dims      = np.arange(d_model)[np.newaxis, :]
    angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)


def build_encoder(n_features=N_FEATURES, d_model=D_MODEL, n_heads=N_HEADS,
                  n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT,
                  window_len=WINDOW_LEN) -> keras.Model:
    inp = keras.Input(shape=(window_len, n_features), name='input')
    x   = layers.Dense(d_model, name='input_proj')(inp)
    x   = x + _positional_encoding(window_len, d_model)  # PE on feature tokens only
    x   = CLSToken(name='cls_token')(x)                   # (B, window_len+1, d_model)
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
    # x is (B, window_len+1, d_model): position 0 = CLS, 1: = feature tokens
    H     = layers.Lambda(lambda z: z[:, 1:, :],  name='H')(x)      # (B, 288, 128)
    h_cls = layers.Lambda(lambda z: z[:, 0, :],   name='h_cls')(x)  # (B, 128)
    return keras.Model(inp, [H, h_cls], name='TransformerEncoder')


def load_encoder(weights_path: str = WEIGHTS_PATH, trainable: bool = False,
                 **kwargs) -> keras.Model:
    encoder = build_encoder(**kwargs)
    encoder(tf.zeros((1, WINDOW_LEN, kwargs.get('n_features', N_FEATURES))))
    encoder.load_weights(weights_path)
    encoder.trainable = trainable
    return encoder


# ── Encoder3 — Prefix-LM, no CLS ─────────────────────────────────────────────

class PrefixLMBlock(layers.Layer):
    """
    Transformer block with prefix-LM attention mask.
    Positions [0, prefix_len) are fully bidirectional.
    Positions [prefix_len, seq_len) are causal but attend to the full prefix.
    Mask is precomputed once and stored as a constant — no per-call overhead.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout,
                 prefix_len, seq_len, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.mhsa  = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout)
        self.drop1 = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn1  = layers.Dense(d_ff, activation='relu')
        self.drop2 = layers.Dropout(dropout)
        self.ffn2  = layers.Dense(d_model)
        self.drop3 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        rows    = np.arange(seq_len)[:, np.newaxis]
        cols    = np.arange(seq_len)[np.newaxis, :]
        blocked = (rows >= prefix_len) & (cols >= prefix_len) & (cols > rows)
        self._mask = tf.constant((~blocked)[np.newaxis], dtype=tf.bool)  # (1, T, T)

    def call(self, x, training=False):
        attn = self.mhsa(x, x, attention_mask=self._mask, training=training)
        attn = self.drop1(attn, training=training)
        x    = self.norm1(x + attn)
        ffn  = self.ffn1(x)
        ffn  = self.drop2(ffn, training=training)
        ffn  = self.ffn2(ffn)
        ffn  = self.drop3(ffn, training=training)
        return self.norm2(x + ffn)


def build_encoder3(n_features: int = N_FEATURES, d_model: int = D_MODEL,
                   n_heads: int = N_HEADS, n_layers: int = N_LAYERS,
                   d_ff: int = D_FF, dropout: float = DROPOUT,
                   window_len: int = WINDOW_LEN,
                   prefix_len: int = PREFIX_LEN) -> keras.Model:
    """
    Encoder3: Prefix-LM masking, no CLS token, sequence length = window_len.
    Steps [0, prefix_len) fully bidirectional; [prefix_len, window_len) causal + full prefix.
    Returns [H (B, T, d), h_last (B, d)] — same two-output API as encoder2.
    h_last = H[:, -1, :]: final causal step, used as per-window summary in Stage 2.
    """
    inp = keras.Input(shape=(window_len, n_features), name='input')
    x   = layers.Dense(d_model, name='input_proj')(inp)
    x   = x + _positional_encoding(window_len, d_model)
    for i in range(n_layers):
        x = PrefixLMBlock(d_model, n_heads, d_ff, dropout,
                          prefix_len, window_len, name=f'prefix_lm_{i}')(x)
    h_last = layers.Lambda(lambda z: z[:, -1, :], name='h_last')(x)
    return keras.Model(inp, [x, h_last], name='TransformerEncoder3')


def load_encoder3(weights_path: str = WEIGHTS_PATH_E3, trainable: bool = False,
                  **kwargs) -> keras.Model:
    encoder = build_encoder3(**kwargs)
    encoder(tf.zeros((1, WINDOW_LEN, kwargs.get('n_features', N_FEATURES))))
    encoder.load_weights(weights_path)
    encoder.trainable = trainable
    return encoder


# ── Decoder — fully causal (GPT-style, next-token prediction) ────────────────

def build_decoder(n_features: int = N_FEATURES, d_model: int = D_MODEL,
                  n_heads: int = N_HEADS, n_layers: int = N_LAYERS,
                  d_ff: int = D_FF, dropout: float = DROPOUT,
                  window_len: int = WINDOW_LEN) -> keras.Model:
    """
    Causal decoder: PrefixLMBlock(prefix_len=0) → fully lower-triangular attention mask.
    Identical hyperparams to encoder2. No CLS token.
    Returns [H (B, T, d), h_last (B, d)] — same two-output API as encoder3.
    h_last = H[:, -1, :]: causally conditioned summary of the full 24h window.
    """
    model = build_encoder3(n_features=n_features, d_model=d_model, n_heads=n_heads,
                           n_layers=n_layers, d_ff=d_ff, dropout=dropout,
                           window_len=window_len, prefix_len=0)
    model._name = 'CausalDecoder'
    return model


def load_decoder(weights_path: str = WEIGHTS_PATH_DECODER, trainable: bool = False,
                 **kwargs) -> keras.Model:
    decoder = build_decoder(**kwargs)
    decoder(tf.zeros((1, WINDOW_LEN, kwargs.get('n_features', N_FEATURES))))
    decoder.load_weights(weights_path)
    decoder.trainable = trainable
    return decoder


def load_ntp_head(model_weights_path: str = NTP_MODEL_WEIGHTS) -> keras.Model:
    """
    Load NTP Dense head from a saved full NTP model for AR rollout inference.
    Returns a head model: (B, 1, D_MODEL) → (B,) predicting next CGM z-score.

    The full NTP model structure: decoder → H[:,:-1,:] → Dense(64,relu,'ntp_hidden')
    → Dense(1,'ntp_head') → Reshape(287,). We rebuild it, load weights, then
    wrap the two Dense layers in a small head model for per-step AR inference.
    """
    dec   = build_decoder()
    inp_f = keras.Input(shape=(WINDOW_LEN, N_FEATURES), name='input')
    H_f, _ = dec(inp_f)
    H_p   = layers.Lambda(lambda z: z[:, :-1, :], name='H_pred')(H_f)
    h_out = layers.Dense(64, activation='relu', name='ntp_hidden')(H_p)
    h_out = layers.Dense(1,  name='ntp_head')(h_out)
    h_out = layers.Reshape((WINDOW_LEN - 1,), name='ntp_output')(h_out)
    full  = keras.Model(inp_f, h_out, name='NTP_Decoder')
    full(tf.zeros((1, WINDOW_LEN, N_FEATURES)))
    full.load_weights(model_weights_path)

    # Build small head reusing the Dense layers (shared weights)
    h_inp  = keras.Input(shape=(1, D_MODEL), name='h_input')
    x      = full.get_layer('ntp_hidden')(h_inp)   # (B, 1, 64)
    x      = full.get_layer('ntp_head')(x)          # (B, 1, 1)
    out    = layers.Lambda(lambda z: z[:, 0, 0], name='pred_z')(x)  # (B,)
    return keras.Model(h_inp, out, name='NTPHead')


def load_decoder2(weights_path: str = WEIGHTS_PATH_DECODER2, trainable: bool = False,
                  **kwargs) -> keras.Model:
    decoder = build_decoder(**kwargs)
    decoder(tf.zeros((1, WINDOW_LEN, kwargs.get('n_features', N_FEATURES))))
    decoder.load_weights(weights_path)
    decoder.trainable = trainable
    return decoder


def load_ntp_head2(head_weights_path: str = NTP_HEAD2_WEIGHTS) -> keras.Model:
    """
    Load decoder2 NTP delta head for AR rollout.
    Outputs Δ = CGM[t+1] − CGM[t] (z-score delta, not absolute).
    Returns head model: (B, 1, D_MODEL) → (B,).
    """
    h_inp = keras.Input(shape=(1, D_MODEL), name='h_input')
    x     = layers.Dense(64, activation='relu', name='ntp_hidden')(h_inp)
    x     = layers.Dense(1,  name='ntp_head')(x)
    out   = layers.Lambda(lambda z: z[:, 0, 0], name='pred_delta')(x)
    head  = keras.Model(h_inp, out, name='NTPHead2')
    head(tf.zeros((1, 1, D_MODEL)))
    head.load_weights(head_weights_path)
    return head
