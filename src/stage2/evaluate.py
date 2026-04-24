import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# ── Metrics ───────────────────────────────────────────────────────────────────

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-horizon RMSE, MAE, R² for shape (N, H) arrays."""
    metrics = {}
    n_horizons = y_true.shape[1]
    for i in range(n_horizons):
        yt, yp = y_true[:, i], y_pred[:, i]
        metrics[f'RMSE_h{i}'] = float(np.sqrt(np.mean((yt - yp) ** 2)))
        metrics[f'MAE_h{i}']  = float(np.mean(np.abs(yt - yp)))
        metrics[f'R2_h{i}']   = float(r2_score(yt, yp))
    return metrics


def clarke_zones(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Clarke Error Grid zone percentages for 1-D arrays (mg/dL)."""
    n = len(y_true)
    counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    for ref, pred in zip(y_true, y_pred):
        counts[_clarke_point(float(ref), float(pred))] += 1
    return {z: v / n for z, v in counts.items()}


def _clarke_point(y: float, yp: float) -> str:
    """Zone classification matching Guevara & Gonzalez MATLAB implementation."""
    # Zone A
    if (yp <= 70 and y <= 70) or (yp <= 1.2 * y and yp >= 0.8 * y):
        return 'A'
    # Zone E
    if (y >= 180 and yp <= 70) or (y <= 70 and yp >= 180):
        return 'E'
    # Zone C
    if ((y >= 70 and y <= 290) and yp >= y + 110) or \
       ((y >= 130 and y <= 180) and yp <= (7 / 5) * y - 182):
        return 'C'
    # Zone D
    if ((y >= 240) and (yp >= 70 and yp <= 180)) or \
       (y <= 175 / 3 and (yp <= 180 and yp >= 70)) or \
       ((y >= 175 / 3 and y <= 70) and yp >= (6 / 5) * y):
        return 'D'
    return 'B'


def save_metrics(metrics: dict, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)


def print_metrics(metrics: dict) -> None:
    for k, v in metrics.items():
        print(f'  {k:<20} {v:.4f}')


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_training_curves(histories: dict, path: str) -> None:
    n = len(histories)
    ncols = min(n, 2)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for ax, (tag, history) in zip(axes, histories.items()):
        ax.plot(history.history['loss'],     label='train')
        ax.plot(history.history['val_loss'], label='val')
        ax.set_title(tag.upper())
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Huber loss')
        ax.legend()
    for ax in axes[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_prediction_scatter(y_true: np.ndarray, y_pred: np.ndarray,
                             path: str) -> None:
    n_horizons = y_true.shape[1]
    ncols = min(n_horizons, 3)
    nrows = (n_horizons + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for i in range(n_horizons):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.3, s=5)
        lim = [min(y_true[:, i].min(), y_pred[:, i].min()),
               max(y_true[:, i].max(), y_pred[:, i].max())]
        ax.plot(lim, lim, 'r--', linewidth=1)
        ax.set_xlabel('True CGM (mg/dL)')
        ax.set_ylabel('Predicted CGM (mg/dL)')
        ax.set_title(f't+{(i+1)*5}min')
    for ax in axes[n_horizons:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_clarke_grid(y_true: np.ndarray, y_pred: np.ndarray,
                     path: str, horizon_min: int = None) -> None:
    """
    Clarke Error Grid faithful to Guevara & Gonzalez MATLAB implementation.
    Uses the last horizon of y_true/y_pred (both shape (N, H) or (N,)).
    """
    if y_true.ndim == 2:
        h = y_true.shape[1]
        ref  = y_true[:, -1]
        pred = y_pred[:, -1]
        label_min = horizon_min if horizon_min is not None else h * 5
    else:
        ref  = y_true
        pred = y_pred
        label_min = horizon_min if horizon_min is not None else ''

    fig, ax = plt.subplots(figsize=(6, 6))

    # Zone-coloured scatter: colour each point by its zone
    zone_colors = {'A': '#2ca02c', 'B': '#1f77b4', 'C': '#ff7f0e',
                   'D': '#9467bd', 'E': '#d62728'}
    point_zones = [_clarke_point(float(r), float(p)) for r, p in zip(ref, pred)]
    colors = [zone_colors[z] for z in point_zones]
    ax.scatter(ref, pred, c=colors, s=6, alpha=0.35, linewidths=0)

    # Diagonal (perfect agreement)
    ax.plot([0, 400], [0, 400], 'k:', linewidth=1)

    # Zone boundaries (exact MATLAB lines)
    kw = dict(color='k', linewidth=1)
    ax.plot([0,       175/3],  [70,  70],  **kw)
    ax.plot([175/3,   400/1.2],[70,  400], **kw)   # upper A boundary
    ax.plot([70,      70],     [84,  400], **kw)
    ax.plot([0,       70],     [180, 180], **kw)
    ax.plot([70,      290],    [180, 400], **kw)   # upper B-C boundary
    ax.plot([70,      70],     [0,   56],  **kw)
    ax.plot([70,      400],    [56,  320], **kw)   # lower A boundary
    ax.plot([180,     180],    [0,   70],  **kw)
    ax.plot([180,     400],    [70,  70],  **kw)
    ax.plot([240,     240],    [70,  180], **kw)
    ax.plot([240,     400],    [180, 180], **kw)
    ax.plot([130,     180],    [0,   70],  **kw)   # lower B-C boundary

    # Zone labels
    font = dict(fontsize=12)
    ax.text(30,  20,  'A', **font)
    ax.text(30,  150, 'D', **font)
    ax.text(30,  380, 'E', **font)
    ax.text(150, 380, 'C', **font)
    ax.text(160, 20,  'C', **font)
    ax.text(380, 20,  'E', **font)
    ax.text(380, 120, 'D', **font)
    ax.text(380, 260, 'B', **font)
    ax.text(280, 380, 'B', **font)

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_aspect('equal')
    ax.set_xlabel('Reference Concentration [mg/dL]')
    ax.set_ylabel('Predicted Concentration [mg/dL]')
    title = f"Clarke's Error Grid"
    if label_min:
        title += f' (t+{label_min}min)'
    ax.set_title(title)

    # Zone percentages as annotation
    zones = clarke_zones(ref, pred)
    summary = '  '.join(f'{z}:{zones[z]:.1%}' for z in 'ABCDE')
    ax.text(0.02, 0.97, summary, transform=ax.transAxes,
            fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
