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

WEIGHTS_PATH        = 'results/mtsm/encoder_global_norm/encoder_weights.weights.h5'
WEIGHTS_PATH_DECODER  = 'results/mtsm/decoder_global_norm/encoder_weights.weights.h5'
NTP_MODEL_WEIGHTS     = 'results/mtsm/decoder_global_norm/model_weights.weights.h5'


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


# ── Decoder — fully causal (GPT-style, next-token prediction) ────────────────

class _CausalBlock(layers.Layer):
    """Transformer block with fully lower-triangular (causal) attention mask."""
    def __init__(self, d_model, n_heads, d_ff, dropout, seq_len, name=None, **kwargs):
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
        rows = np.arange(seq_len)[:, np.newaxis]
        cols = np.arange(seq_len)[np.newaxis, :]
        self._mask = tf.constant((cols <= rows)[np.newaxis], dtype=tf.bool)  # (1, T, T)

    def call(self, x, training=False):
        attn = self.mhsa(x, x, attention_mask=self._mask, training=training)
        attn = self.drop1(attn, training=training)
        x    = self.norm1(x + attn)
        ffn  = self.ffn1(x)
        ffn  = self.drop2(ffn, training=training)
        ffn  = self.ffn2(ffn)
        ffn  = self.drop3(ffn, training=training)
        return self.norm2(x + ffn)


def build_decoder(n_features: int = N_FEATURES, d_model: int = D_MODEL,
                  n_heads: int = N_HEADS, n_layers: int = N_LAYERS,
                  d_ff: int = D_FF, dropout: float = DROPOUT,
                  window_len: int = WINDOW_LEN) -> keras.Model:
    """
    Causal decoder: fully lower-triangular attention mask, no CLS token.
    Identical hyperparams to the encoder.
    Returns [H (B, T, d), h_last (B, d)] — h_last = H[:, -1, :].
    """
    inp = keras.Input(shape=(window_len, n_features), name='input')
    x   = layers.Dense(d_model, name='input_proj')(inp)
    x   = x + _positional_encoding(window_len, d_model)
    for i in range(n_layers):
        x = _CausalBlock(d_model, n_heads, d_ff, dropout,
                         window_len, name=f'causal_{i}')(x)
    h_last = layers.Lambda(lambda z: z[:, -1, :], name='h_last')(x)
    return keras.Model(inp, [x, h_last], name='CausalDecoder')


def load_decoder(weights_path: str = WEIGHTS_PATH_DECODER, trainable: bool = False,
                 **kwargs) -> keras.Model:
    decoder = build_decoder(**kwargs)
    decoder(tf.zeros((1, WINDOW_LEN, kwargs.get('n_features', N_FEATURES))))
    decoder.load_weights(weights_path)
    decoder.trainable = trainable
    return decoder


_NTP_K_BINS    = 200
_NTP_BIN_Z_MIN = (40.0  - 144.40) / 57.11
_NTP_BIN_Z_MAX = (400.0 - 144.40) / 57.11
_NTP_BIN_CTRS  = tf.constant(
    [_NTP_BIN_Z_MIN + (_NTP_BIN_Z_MAX - _NTP_BIN_Z_MIN) * (k + 0.5) / _NTP_K_BINS
     for k in range(_NTP_K_BINS)],
    dtype=tf.float32,
)   # (K,)


def load_ntp_head(model_weights_path: str = NTP_MODEL_WEIGHTS) -> keras.Model:
    """
    Load NTP Dense head from decoder_global_norm for AR rollout inference.
    Returns a head model: (B, 1, D_MODEL) → (B,) predicting next CGM z-score.

    decoder_global_norm uses K=200 bin CE head (build_ntp_model_clean):
    H[:,:-1,:] → Dense(64,relu,'ntp_hidden') → Dense(200,'ntp_output').
    Soft-argmax decodes logits → continuous z-score.
    """
    dec   = build_decoder()
    inp_f = keras.Input(shape=(WINDOW_LEN, N_FEATURES), name='input')
    H_f, _ = dec(inp_f)
    H_p   = layers.Lambda(lambda z: z[:, :-1, :], name='H_pred')(H_f)
    h_out = layers.Dense(64,          activation='relu', name='ntp_hidden')(H_p)
    h_out = layers.Dense(_NTP_K_BINS,                   name='ntp_output')(h_out)
    full  = keras.Model(inp_f, h_out, name='NTP_Decoder')
    full(tf.zeros((1, WINDOW_LEN, N_FEATURES)))
    full.load_weights(model_weights_path)

    # Small head: (B, 1, D_MODEL) → (B,) via soft-argmax
    h_inp = keras.Input(shape=(1, D_MODEL), name='h_input')
    x     = full.get_layer('ntp_hidden')(h_inp)    # (B, 1, 64)
    x     = full.get_layer('ntp_output')(x)         # (B, 1, K)
    out   = layers.Lambda(
        lambda logits: tf.reduce_sum(
            tf.nn.softmax(logits[:, 0, :], axis=-1) * _NTP_BIN_CTRS, axis=-1
        ),
        name='pred_z',
    )(x)   # (B,)
    return keras.Model(h_inp, out, name='NTPHead')
