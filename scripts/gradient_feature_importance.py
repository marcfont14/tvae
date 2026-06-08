"""
Gradient-based feature importance for the imputation model.

For each test window, compute the gradient of the imputed CGM (at masked
positions) with respect to every input channel. Mean |gradient| per channel,
averaged across all masked timesteps and test windows, is the importance score.

This is vanilla saliency: |∂output/∂input|. No patient-level aggregation,
no proxy variables — works directly on the 7 model inputs at timestep level.

Run from /mnt/workspace/tvae:
  python -u scripts/gradient_feature_importance.py 2>&1 | tee results/gradient_importance.txt
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stage2.data import (
    load_all_patients, make_imputation_dataset, make_eval_imputation_numpy,
    IMPUTATION_GAP_LENGTHS, IMPUTATION_GAP_LABELS,
)
from src.stage2.models import build_mtsm_imputation_model

OUT_DIR   = 'results/feature_importance'
os.makedirs(OUT_DIR, exist_ok=True)

FEAT_NAMES = ['CGM\n(col 0)', 'PI\n(col 1)', 'RA\n(col 2)',
              'hour_sin\n(col 3)', 'hour_cos\n(col 4)',
              'bolus\n(col 5)', 'carbs\n(col 6)']
FEAT_COLORS = ['#374151', '#1d4ed8', '#2563eb', '#ca8a04', '#d97706',
               '#16a34a', '#15803d']
N_ACTIVE = 7   # cols 0–6; cols 7–9 (therapy) are zero-filled


def compute_gradient_importance(mtsm, W_masked, masks, batch_size=64):
    """
    Returns mean |∂output/∂input| per feature channel (shape: N_ACTIVE,).

    W_masked : (N, 288, 10)  — context with CGM zeroed in gap
    masks    : (N, 288)      — 1 = masked position
    """
    N = len(W_masked)
    importance_sum = np.zeros(N_ACTIVE, dtype=np.float64)
    n_masked_total = 0

    for start in range(0, N, batch_size):
        W_b = tf.constant(W_masked[start:start + batch_size, :, :N_ACTIVE],
                          dtype=tf.float32)          # (B, 288, 7)
        # Pad to 10 channels with zeros for cols 7–9 (therapy excluded)
        zeros = tf.zeros((*W_b.shape[:2], 3), dtype=tf.float32)
        W_b10 = tf.concat([W_b, zeros], axis=-1)    # (B, 288, 10)

        m_b = masks[start:start + batch_size]        # (B, 288)

        with tf.GradientTape() as tape:
            tape.watch(W_b10)
            pred = mtsm(W_b10, training=False)       # (B, 288)
            # Differentiate w.r.t. sum of predictions at masked positions
            masked_pred = pred * tf.constant(m_b, dtype=tf.float32)
            scalar = tf.reduce_sum(masked_pred)

        grads = tape.gradient(scalar, W_b10)         # (B, 288, 10)
        grads_np = np.abs(grads.numpy()[:, :, :N_ACTIVE])  # (B, 288, 7)

        n_masked = int(m_b.sum())
        # Weight by mask: only accumulate at masked output positions
        # (gradient at ALL input timesteps, but normalised by n_masked)
        importance_sum += grads_np.sum(axis=(0, 1))
        n_masked_total += N * grads_np.shape[1]     # normalise by all positions

    # Return mean |grad| per channel across all (sample, timestep) pairs
    return importance_sum / (N * W_masked.shape[1])


def main():
    print('\n=== Gradient Feature Importance — Imputation ===\n')

    patients = load_all_patients('data/processed/adults_global_norm')
    splits   = make_imputation_dataset(patients, batch_size=128)

    print('Loading FM imputation model...')
    mtsm = build_mtsm_imputation_model()
    print(f'  Params: {mtsm.count_params():,}\n')

    gap_importance = {}

    for gap_len, gap_label in zip(IMPUTATION_GAP_LENGTHS, IMPUTATION_GAP_LABELS):
        print(f'--- Gap: {gap_label} ---', flush=True)
        ev       = make_eval_imputation_numpy(splits['test_patients'], gap_len,
                                              max_windows=1000)
        W_masked = ev['windows_masked']
        masks    = ev['masks']

        imp = compute_gradient_importance(mtsm, W_masked, masks)
        gap_importance[gap_label] = imp

        print(f'  Mean |grad| per channel:')
        for i, (name, v) in enumerate(zip(FEAT_NAMES, imp)):
            bar = '█' * int(v / imp.max() * 30)
            print(f'    col {i}: {v:.4e}  {bar}')
        print()

    # ── Plot ────────────────────────────────────────────────────────────────
    gaps  = list(gap_importance.keys())
    n_gap = len(gaps)
    bar_w = 0.18
    x     = np.arange(N_ACTIVE)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: grouped bar per gap length
    ax = axes[0]
    offsets = np.linspace(-(n_gap - 1) / 2, (n_gap - 1) / 2, n_gap) * bar_w
    gap_colors = ['#1e3a5f', '#2563eb', '#60a5fa', '#bfdbfe']
    for j, (gl, col) in enumerate(zip(gaps, gap_colors)):
        vals = gap_importance[gl]
        ax.bar(x + offsets[j], vals / vals.max(), bar_w,
               label=f'{gl} gap', color=col, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(FEAT_NAMES, fontsize=8.5)
    ax.set_ylabel('Relative mean |∂output/∂input|', fontsize=9)
    ax.set_title('Gradient feature importance\n(normalised within each gap)', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.12)

    # Right: 4h gap, absolute values, coloured by feature
    ax2 = axes[1]
    vals_4h = gap_importance['4h']
    bars = ax2.bar(x, vals_4h, color=FEAT_COLORS, alpha=0.88)
    ax2.set_xticks(x)
    ax2.set_xticklabels(FEAT_NAMES, fontsize=8.5)
    ax2.set_ylabel('Mean |∂output/∂input|', fontsize=9)
    ax2.set_title('Gradient importance — 4h gap\n(absolute values)', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vals_4h):
        ax2.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                 f'{v:.2e}', ha='center', va='bottom', fontsize=7)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'gradient_importance.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved {path}')
    plt.close(fig)

    # Print summary table
    print('\nSummary — normalised importance (4h gap = reference):')
    ref = gap_importance['4h']
    header = f'  {"Feature":<18}' + ''.join(f'{g:>10}' for g in gaps)
    print(header)
    print('  ' + '-' * (18 + 10 * n_gap))
    for i, name in enumerate(FEAT_NAMES):
        tag = name.replace('\n', ' ')
        row = f'  {tag:<18}' + ''.join(
            f'{gap_importance[g][i] / gap_importance[g].max():>10.3f}' for g in gaps
        )
        print(row)


if __name__ == '__main__':
    main()
