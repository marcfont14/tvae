"""
Time-of-day gradient saliency for hour_sin / hour_cos features.

Computes the gradient of imputed CGM w.r.t. hour_sin (col 3) and hour_cos (col 4)
per test window, then groups windows by the hour at the gap centre and plots
mean |gradient| as a function of time of day.

Also overlays the R² drop from the ablation (full vs no_time) per period,
which was already computed in variable_justification.py.

Run from /mnt/workspace/tvae:
  python -u scripts/time_feature_importance.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stage2.data import (
    load_all_patients, make_imputation_dataset, make_eval_imputation_numpy,
    IDX_HSIN, IDX_HCOS,
)
from src.stage2.models import build_mtsm_imputation_model

OUT_DIR = 'results/feature_importance'
os.makedirs(OUT_DIR, exist_ok=True)

# Ablation R² drop (full vs no_time) from variable_justification.py
# Columns: [4h, 6h]
ABLATION = {
    'Dawn\n04–08h':       [-0.0025, -0.0160],
    'Morning\n08–14h':    [-0.0039, -0.0047],
    'Afternoon\n14–20h':  [-0.0013, -0.0021],
    'Night\n20–04h':      [-0.0054, -0.0148],
}


def hour_of_window(W, gap_centre):
    """Return hour of day (0–24) at the gap centre for each window."""
    s = W[:, gap_centre, IDX_HSIN]
    c = W[:, gap_centre, IDX_HCOS]
    return (np.arctan2(s, c) / (2 * np.pi) * 24) % 24


def period_label(h):
    if 4 <= h < 8:
        return 'Dawn\n04–08h'
    elif 8 <= h < 14:
        return 'Morning\n08–14h'
    elif 14 <= h < 20:
        return 'Afternoon\n14–20h'
    else:
        return 'Night\n20–04h'


def compute_time_gradients(mtsm, W_masked, masks, gap_centre, batch_size=64):
    """
    Per-window mean |∂output/∂hour_sin| and |∂output/∂hour_cos|,
    averaged over all timesteps (not just the gap).
    Returns arrays of shape (N,) for each.
    """
    N = len(W_masked)
    grad_sin = np.zeros(N)
    grad_cos = np.zeros(N)

    for start in range(0, N, batch_size):
        sl = slice(start, start + batch_size)
        # Use only cols 0–6; zero-fill cols 7–9 (therapy excluded)
        W7  = W_masked[sl, :, :7].astype(np.float32)
        pad = np.zeros((W7.shape[0], W7.shape[1], 3), dtype=np.float32)
        W10 = tf.constant(np.concatenate([W7, pad], axis=-1))

        m_b = masks[sl].astype(np.float32)

        with tf.GradientTape() as tape:
            tape.watch(W10)
            pred   = mtsm(W10, training=False)        # (B, 288)
            scalar = tf.reduce_sum(pred * tf.constant(m_b))

        grads = tape.gradient(scalar, W10).numpy()    # (B, 288, 10)
        # Mean absolute gradient for hour_sin and hour_cos over all timesteps
        grad_sin[sl] = np.abs(grads[:, :, IDX_HSIN]).mean(axis=1)
        grad_cos[sl] = np.abs(grads[:, :, IDX_HCOS]).mean(axis=1)

    return grad_sin, grad_cos


def main():
    print('=== Time feature gradient saliency ===\n')

    patients = load_all_patients('data/processed/adults_global_norm')
    splits   = make_imputation_dataset(patients, batch_size=128)

    print('Loading model...')
    mtsm = build_mtsm_imputation_model()

    # Use 6h gap (most informative for time effects from the ablation)
    gap_len, gap_label = 72, '6h'
    print(f'Computing gradients for {gap_label} gap...\n')

    ev         = make_eval_imputation_numpy(splits['test_patients'], gap_len,
                                            max_windows=2000)
    W_masked   = ev['windows_masked']
    masks      = ev['masks']
    gap_centre = (ev['gap_start'] + ev['gap_end']) // 2

    hours    = hour_of_window(ev['windows_orig'], gap_centre)
    periods  = [period_label(h) for h in hours]

    grad_sin, grad_cos = compute_time_gradients(mtsm, W_masked, masks, gap_centre)
    grad_time = (grad_sin + grad_cos) / 2   # combined time gradient

    # Group by period
    period_order = ['Dawn\n04–08h', 'Morning\n08–14h',
                    'Afternoon\n14–20h', 'Night\n20–04h']
    mean_grad, n_windows = {}, {}
    for p in period_order:
        idx = np.array([i for i, lbl in enumerate(periods) if lbl == p])
        mean_grad[p] = grad_time[idx].mean() if len(idx) else 0.0
        n_windows[p] = len(idx)
        print(f'  {p.replace(chr(10), " "):25s}  n={len(idx):5d}  '
              f'mean|grad|={mean_grad[p]:.4e}')

    print()

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4.2))

    x = np.arange(4)
    w = 0.32
    for j, (gap, gap_col) in enumerate([('4h', '#60a5fa'), ('6h', '#1d4ed8')]):
        drops = [-ABLATION[p][j] for p in period_order]   # flip sign → positive bar = cost
        ax.bar(x + (j - 0.5) * w, drops, w, label=f'{gap} gap',
               color=gap_col, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(period_order, fontsize=9)
    ax.set_ylabel('R² drop when time features zeroed', fontsize=9)
    ax.set_title('Ablation: cost of removing time features\n'
                 'by time-of-day period', fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'time_feature_importance.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Saved {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
