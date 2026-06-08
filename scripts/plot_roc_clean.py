"""
Clean ROC curve figure for nocturnal hypo risk.
Loads saved model weights, re-runs inference on bedtime test set,
saves predictions as NPZ for future use, plots with approved aesthetic.
"""
import os, sys, gc, json
sys.path.insert(0, '/mnt/workspace/tvae')
os.chdir('/mnt/workspace/tvae')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import roc_curve, roc_auc_score

from src.stage2.data import load_all_patients, make_hypo_dataset
from src.stage2.models import (build_hypo_risk_model, build_hypo_risk_decoder,
                                build_raw_hypo_risk_model)
from src.encoder import load_encoder, _positional_encoding
from tensorflow.keras import layers as klayers


# ── Compatibility decoder builder ─────────────────────────────────────────────
# Saved hypo-risk weights use PrefixLMBlock auto-naming (prefix_lm_block, ...)
# while current build_decoder() uses _CausalBlock (causal_0, ...).

class PrefixLMBlock(klayers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout, seq_len, **kwargs):
        super().__init__(**kwargs)
        self.mhsa  = klayers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout)
        self.drop1 = klayers.Dropout(dropout)
        self.norm1 = klayers.LayerNormalization(epsilon=1e-6)
        self.ffn1  = klayers.Dense(d_ff, activation='relu')
        self.drop2 = klayers.Dropout(dropout)
        self.ffn2  = klayers.Dense(d_model)
        self.drop3 = klayers.Dropout(dropout)
        self.norm2 = klayers.LayerNormalization(epsilon=1e-6)
        rows = np.arange(seq_len)[:, np.newaxis]
        cols = np.arange(seq_len)[np.newaxis, :]
        self._mask = tf.constant((cols <= rows)[np.newaxis], dtype=tf.bool)

    def call(self, x, training=False):
        attn = self.mhsa(x, x, attention_mask=self._mask, training=training)
        attn = self.drop1(attn, training=training)
        x    = self.norm1(x + attn)
        ffn  = self.ffn1(x)
        ffn  = self.drop2(ffn, training=training)
        ffn  = self.ffn2(ffn)
        ffn  = self.drop3(ffn, training=training)
        return self.norm2(x + ffn)


def _build_decoder_compat(n_features=10, d_model=128, n_heads=4,
                           n_layers=5, d_ff=256, dropout=0.2, window_len=288):
    """Decoder architecture matching the original PrefixLMBlock layer names."""
    inp    = keras.Input(shape=(window_len, n_features), name='input')
    x      = klayers.Dense(d_model)(inp)
    x      = x + _positional_encoding(window_len, d_model)
    for _  in range(n_layers):
        x  = PrefixLMBlock(d_model, n_heads, d_ff, dropout, window_len)(x)
    h_last = klayers.Lambda(lambda z: z[:, -1, :], name='h_last')(x)
    return keras.Model(inp, [x, h_last])

DATA_DIR  = 'data/processed/adults_global_norm'
OUT_DIR   = 'results/stage2/hypo_risk/gn_run01'
PREDS_NPZ = f'{OUT_DIR}/predictions.npz'

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading patients...')
patients = load_all_patients(DATA_DIR)
splits   = make_hypo_dataset(patients, batch_size=256)

print('Collecting test set...')
x_list, y_list = [], []
for xb, yb in splits['test']:
    x_list.append(xb.numpy())
    y_list.append(yb.numpy())
x_all  = np.concatenate(x_list)
y_true = np.concatenate(y_list)
last_cgm_z = x_all[:, -1, 0]
print(f'Test windows: {len(x_all):,}  |  hypo rate: {y_true.mean():.3f}')

# ── Run inference for each variant ───────────────────────────────────────────
VARIANTS = [
    ('raw',           'Raw LSTM',            '#D95319', '-'),
    ('fm_ft',         'Enc. fine-tuned',     '#0072BD', '-'),
    ('fm_decoder_ft', 'Dec. fine-tuned',     '#EDB120', '-'),
]

def build_model(mode):
    if mode == 'fm':           return build_hypo_risk_model(load_encoder(trainable=False))
    elif mode == 'fm_ft':      return build_hypo_risk_model(load_encoder(trainable=True))
    elif mode == 'fm_decoder': return build_hypo_risk_decoder(_build_decoder_compat())
    elif mode == 'fm_decoder_ft': return build_hypo_risk_decoder(_build_decoder_compat())
    else:                      return build_raw_hypo_risk_model()

all_scores = {}
for mode, label, color, ls in VARIANTS:
    tag   = f'{mode}_lstm'
    wpath = f'{OUT_DIR}/weights_{tag}.weights.h5'
    if not os.path.exists(wpath):
        print(f'  {tag}: no weights, skipping')
        continue
    keras.backend.clear_session(); gc.collect()
    print(f'  {tag}...')
    m = build_model(mode)
    m(tf.zeros((1, 288, 10)))
    m.load_weights(wpath)
    scores = m.predict(x_all, batch_size=256, verbose=0).ravel()
    all_scores[tag] = scores
    del m; gc.collect()

# Naive: lower last CGM → higher risk
all_scores['naive'] = -last_cgm_z

# Save predictions
np.savez(PREDS_NPZ, y_true=y_true, **all_scores)
print(f'Saved predictions to {PREDS_NPZ}')

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.size':        10,
    'axes.labelsize':   10,
    'axes.titlesize':   9.5,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'legend.fontsize':  8.5,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.edgecolor':   'black',
    'axes.linewidth':   0.8,
    'axes.grid':        True,
    'grid.color':       '#cccccc',
    'grid.linewidth':   0.5,
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    'xtick.top':        True,
    'ytick.right':      True,
    'lines.linewidth':  1.6,
})

plot_specs = [
    ('fm_decoder_ft_lstm', 'Dec. fine-tuned',  '#EDB120', '-'),
    ('fm_ft_lstm',         'Enc. fine-tuned',  '#0072BD', '-'),
    ('raw_lstm',           'Raw LSTM',         '#D95319', '-'),
    ('naive',              'Naive',            '#000000', '--'),
]

fig, ax = plt.subplots(figsize=(6, 5.5))
for tag, label, color, ls in plot_specs:
    if tag not in all_scores:
        continue
    fpr, tpr, _ = roc_curve(y_true, all_scores[tag])
    auc = roc_auc_score(y_true, all_scores[tag])
    ax.plot(fpr, tpr, color=color, linestyle=ls,
            lw=1.6, label=f'{label}  (AUC={auc:.3f})')

ax.plot([0, 1], [0, 1], color='#aaaaaa', lw=0.8, ls=':')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.legend(loc='lower right', framealpha=0.9, edgecolor='#cccccc')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.tight_layout(pad=1.5)
out = f'{OUT_DIR}/roc_curves_clean.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved {out}')
