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

WEIGHTS_PATH = 'results/mtsm/encoder/encoder_weights.weights.h5'


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
    x   = x + _positional_encoding(window_len, d_model)
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
    return keras.Model(inp, x, name='TransformerEncoder')


def load_encoder(weights_path: str = WEIGHTS_PATH, trainable: bool = False,
                 **kwargs) -> keras.Model:
    encoder = build_encoder(**kwargs)
    encoder(tf.zeros((1, WINDOW_LEN, kwargs.get('n_features', N_FEATURES))))
    encoder.load_weights(weights_path)
    encoder.trainable = trainable
    return encoder
