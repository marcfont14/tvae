import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve, brier_score_loss
from sklearn.calibration import calibration_curve


# ── Metrics ───────────────────────────────────────────────────────────────────

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-horizon RMSE, MAE, R², MARD, %within15 for shape (N, H) arrays."""
    metrics = {}
    n_horizons = y_true.shape[1]
    for i in range(n_horizons):
        yt, yp = y_true[:, i], y_pred[:, i]
        err = yp - yt
        metrics[f'RMSE_h{i}']       = float(np.sqrt(np.mean(err ** 2)))
        metrics[f'MAE_h{i}']        = float(np.mean(np.abs(err)))
        metrics[f'R2_h{i}']         = float(r2_score(yt, yp))
        metrics[f'MARD_h{i}']       = float(np.mean(np.abs(err) / np.maximum(yt, 1.0)) * 100)
        metrics[f'within15_h{i}']   = float(np.mean(np.abs(err) <= 15.0) * 100)
        metrics[f'mean_error_h{i}'] = float(np.mean(err))
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


def plot_horizon_comparison(all_metrics: dict, horizon_labels: list, path: str) -> None:
    """
    One figure with three subplots (RMSE, MAE, R²) showing all variants
    across forecast horizons. Naive baseline drawn as dashed black line.
    """
    horizons = [int(h.replace('min', '')) for h in horizon_labels]
    n_h = len(horizons)

    style = {
        'naive':    dict(color='black',   linestyle='--', linewidth=1.5, label='Naive (last value)'),
        'fm_lstm':  dict(color='#1f77b4', linestyle='-',  linewidth=1.5, marker='s', label='FM LSTM'),
        'raw_lstm': dict(color='#2ca02c', linestyle='-',  linewidth=1.5, marker='s', label='Raw LSTM'),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics_to_plot = [
        ('RMSE', 'RMSE (mg/dL)'),
        ('MAE',  'MAE (mg/dL)'),
        ('R2',   'R²'),
    ]

    for ax, (key, ylabel) in zip(axes, metrics_to_plot):
        for tag, metrics in all_metrics.items():
            kw = style.get(tag, dict(linewidth=1.5, label=tag))
            vals = [metrics.get(f'{key}_h{i}', np.nan) for i in range(n_h)]
            ax.plot(horizons, vals, **kw)
        ax.set_xlabel('Forecast horizon (min)')
        ax.set_ylabel(ylabel)
        ax.set_xticks(horizons)
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=7, loc='upper left')
    plt.suptitle('Forecasting performance across horizons', fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_forecast_traces(samples: list, models: dict, path: str,
                          context_steps: int = 12) -> None:
    """
    Plot side-by-side forecast traces for sampled test windows.

    samples : list of (window (288,10), context_cgm_mg (288,), future_cgm_mg (6,))
    models  : {tag: keras.Model}  — models in insertion order set the legend
    """
    if not samples or not models:
        return

    n     = len(samples)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    all_windows = np.stack([s[0] for s in samples])          # (N, 288, 10)
    last_cgm    = np.array([[s[1][-1]] for s in samples],
                           dtype=np.float32)                  # (N, 1) last observed mg/dL
    preds       = {tag: model.predict(
                       {'window': all_windows, 'last_cgm': last_cgm}, verbose=0)
                   for tag, model in models.items()}

    _colors = {
        'raw_lstm': '#2ca02c',
        'fm_lstm':  '#1f77b4',
    }
    _labels = {
        'raw_lstm': 'Raw LSTM',
        'fm_lstm':  'FM (frozen)',
    }

    t_ctx = np.arange(-context_steps + 1, 1) * 5   # -55 … 0 min
    t_fut = np.arange(1, 25) * 5                    # 5 … 120 min

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes_flat  = np.array(axes).flatten()

    for i, ax in enumerate(axes_flat[:n]):
        ctx_mg  = samples[i][1][-context_steps:]
        true_mg = samples[i][2]

        ax.plot(t_ctx, ctx_mg, color='#555555', linewidth=1.5, zorder=2)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axhspan(70, 180, alpha=0.06, color='green', zorder=0)

        ax.scatter(t_fut, true_mg, color='black', s=35, zorder=5,
                   marker='o', label='Ground truth')

        for tag, pred_all in preds.items():
            ax.plot(t_fut, pred_all[i],
                    color=_colors.get(tag, 'purple'),
                    linewidth=1.5, marker='s', markersize=4,
                    label=_labels.get(tag, tag), zorder=4)

        ax.set_xlabel('Time (min)', fontsize=7)
        ax.set_ylabel('CGM (mg/dL)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25)

    axes_flat[0].legend(fontsize=6, loc='best')

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.suptitle('Sample forecast traces — test windows', fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Classification metrics for binary risk prediction.
    Includes ranking (AUROC, AUPRC), calibration (Brier), and operating-point
    metrics at both 90% and 95% specificity plus the optimal-F1 threshold.
    """
    auroc = float(roc_auc_score(y_true, y_pred))
    auprc = float(average_precision_score(y_true, y_pred))
    if y_pred.min() >= 0.0 and y_pred.max() <= 1.0:
        brier = float(brier_score_loss(y_true, y_pred))
    else:
        brier = float('nan')  # not a probability (e.g. naive raw-score baseline)

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    spec = 1.0 - fpr

    def _sens_at_spec(target_spec):
        idx = np.searchsorted(spec[::-1], target_spec)
        return float(tpr[::-1][min(idx, len(tpr) - 1)])

    sens_at_90spec = _sens_at_spec(0.90)
    sens_at_95spec = _sens_at_spec(0.95)

    prec, rec, thresh_pr = precision_recall_curve(y_true, y_pred)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    best = int(np.argmax(f1))
    ppv_optimal = float(prec[best])
    nna_optimal = float(1.0 / max(prec[best], 1e-6))   # number needed to alarm

    return {
        'auroc':             auroc,
        'auprc':             auprc,
        'brier':             brier,
        'prevalence':        float(y_true.mean()),
        'sens_at_90spec':    sens_at_90spec,
        'sens_at_95spec':    sens_at_95spec,
        'ppv_optimal':       ppv_optimal,
        'nna_optimal':       nna_optimal,
        'f1_optimal':        float(f1[best]),
        'threshold_optimal': float(thresh_pr[best]) if best < len(thresh_pr) else 0.5,
    }


def stratified_auroc(y_true: np.ndarray, y_pred: np.ndarray,
                     last_cgm_z: np.ndarray, threshold: float = 0.0) -> dict:
    """
    AUROC split by last observed CGM z-score.

    "Hard" cases (last_cgm_z > threshold): current glucose is normal/high — the
    naive last-value baseline predicts low risk here, so only models that have
    learned trajectory and driver dynamics can do well.

    "Easy" cases (last_cgm_z <= threshold): glucose is already low — all models
    (including naive) should detect these.
    """
    results = {}
    for name, mask in [('hard', last_cgm_z > threshold),
                       ('easy', last_cgm_z <= threshold)]:
        n_pos = int(y_true[mask].sum())
        n_neg = int((~y_true[mask].astype(bool)).sum())
        if n_pos < 5 or n_neg < 5:
            continue
        try:
            results[f'auroc_{name}'] = float(roc_auc_score(y_true[mask], y_pred[mask]))
            results[f'auprc_{name}'] = float(average_precision_score(y_true[mask], y_pred[mask]))
            results[f'n_{name}']     = int(mask.sum())
            results[f'prev_{name}']  = float(y_true[mask].mean())
        except Exception:
            pass
    return results


def plot_calibration(all_preds: dict, path: str) -> None:
    """
    Reliability diagram (calibration curve) for all variants except naive.
    Naive scores are not probabilities so cannot be calibrated.
    A perfectly calibrated model lies on the diagonal.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    _colors = {'raw_lstm': '#2ca02c', 'fm_lstm': '#1f77b4',
               'fm_ft_lstm': '#ff7f0e', 'fm_decoder_lstm': '#9467bd',
               'fm_decoder_ft_lstm': '#d62728'}
    for tag, (yt, yp) in all_preds.items():
        if tag == 'naive':
            continue
        try:
            frac_pos, mean_pred = calibration_curve(yt, yp, n_bins=10, strategy='quantile')
            ax.plot(mean_pred, frac_pos, marker='o', linewidth=1.5,
                    color=_colors.get(tag, 'grey'), label=tag)
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, label='Perfect')
    ax.set_xlabel('Mean predicted risk')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration — Hypo Risk')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_roc_curves(all_preds: dict, path: str) -> None:
    """ROC curves for all variants. all_preds = {tag: (y_true, y_pred_proba)}."""
    _colors = {'raw_lstm': '#2ca02c', 'fm_lstm': '#1f77b4', 'naive': 'black'}
    fig, ax = plt.subplots(figsize=(5, 5))
    for tag, (yt, yp) in all_preds.items():
        fpr, tpr, _ = roc_curve(yt, yp)
        auc = roc_auc_score(yt, yp)
        ls  = '--' if tag == 'naive' else '-'
        ax.plot(fpr, tpr, color=_colors.get(tag, 'grey'),
                linewidth=1.8, linestyle=ls, label=f'{tag}  AUC={auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k:', linewidth=0.8, alpha=0.4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC — Hypo Risk (next 2 h)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_pr_curves(all_preds: dict, prevalence: float, path: str) -> None:
    """Precision–Recall curves. all_preds = {tag: (y_true, y_pred_proba)}."""
    _colors = {'raw_lstm': '#2ca02c', 'fm_lstm': '#1f77b4', 'naive': 'black'}
    fig, ax = plt.subplots(figsize=(5, 5))
    for tag, (yt, yp) in all_preds.items():
        prec, rec, _ = precision_recall_curve(yt, yp)
        ap  = average_precision_score(yt, yp)
        ls  = '--' if tag == 'naive' else '-'
        ax.plot(rec, prec, color=_colors.get(tag, 'grey'),
                linewidth=1.8, linestyle=ls, label=f'{tag}  AP={ap:.3f}')
    ax.axhline(prevalence, color='grey', linewidth=0.8, linestyle=':',
               alpha=0.7, label=f'No skill ({prevalence:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision–Recall — Hypo Risk (next 2 h)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_classification_table(all_metrics: dict, path: str) -> None:
    cols = ['model', 'auroc', 'auprc', 'sens_at_90spec', 'f1_optimal',
            'prevalence', 'epochs', 'time_min']
    rows = []
    for tag, m in all_metrics.items():
        rows.append({
            'model':          tag,
            'auroc':          f"{m.get('auroc', float('nan')):.4f}",
            'auprc':          f"{m.get('auprc', float('nan')):.4f}",
            'sens_at_90spec': f"{m.get('sens_at_90spec', float('nan')):.4f}",
            'f1_optimal':     f"{m.get('f1_optimal', float('nan')):.4f}",
            'prevalence':     f"{m.get('prevalence', float('nan')):.4f}",
            'epochs':         str(int(m['epochs_trained'])) if 'epochs_trained' in m else '-',
            'time_min':       f"{m['train_time_min']:.1f}" if 'train_time_min' in m else '-',
        })
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


# ── Imputation ────────────────────────────────────────────────────────────────

def linear_interpolate(windows_masked: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """
    Fill each masked gap with piecewise linear interpolation between the last observed
    value before the gap and the first observed value after it.
    Returns (N, 288) array of full-window CGM z-scores (only gap positions changed).
    """
    result = windows_masked[:, :, 0].copy()   # CGM z-score, feature index 0
    L      = result.shape[1]
    for i in range(len(windows_masked)):
        m = masks[i].astype(bool)
        if not m.any():
            continue
        gap_s = int(np.where(m)[0][0])
        gap_e = int(np.where(m)[0][-1]) + 1
        v0 = result[i, gap_s - 1] if gap_s > 0  else result[i, gap_e % L]
        v1 = result[i, gap_e]     if gap_e < L  else result[i, gap_s - 1]
        for t in range(gap_s, gap_e):
            alpha        = (t - gap_s + 1) / (gap_e - gap_s + 1)
            result[i, t] = v0 * (1 - alpha) + v1 * alpha
    return result


def imputation_metrics(y_true_z: np.ndarray, y_pred_z: np.ndarray,
                        masks: np.ndarray,
                        scaler_means: np.ndarray, scaler_stds: np.ndarray) -> dict:
    """
    Compute imputation quality metrics on masked timesteps only.
    y_true_z / y_pred_z : (N, 288) CGM z-scores
    masks               : (N, 288) binary  (1 = was masked, 0 = observed)
    scaler_means/stds   : (N,) per-window inverse-transform parameters
    """
    m      = masks.astype(bool)
    true_z = y_true_z[m]
    pred_z = y_pred_z[m]

    sm = scaler_means[:, np.newaxis]   # (N, 1)
    ss = scaler_stds[:, np.newaxis]
    true_mg = (y_true_z * ss + sm)[m]
    pred_mg = (y_pred_z * ss + sm)[m]

    ss_res = float(np.sum((true_z - pred_z) ** 2))
    ss_tot = float(np.sum((true_z - true_z.mean()) ** 2))

    within15 = float(np.mean(
        (np.abs(pred_mg - true_mg) <= 15.0) |
        (np.abs(pred_mg - true_mg) / np.maximum(true_mg, 1.0) <= 0.15)
    ) * 100)

    return {
        'MAE_z':      float(np.mean(np.abs(true_z - pred_z))),
        'RMSE_z':     float(np.sqrt(np.mean((true_z - pred_z) ** 2))),
        'R2_z':       float(1 - ss_res / (ss_tot + 1e-8)),
        'MAE_mg':     float(np.mean(np.abs(true_mg - pred_mg))),
        'RMSE_mg':    float(np.sqrt(np.mean((true_mg - pred_mg) ** 2))),
        'within15':   within15,
        'mean_error': float(np.mean(pred_mg - true_mg)),
        'n_masked':   int(m.sum()),
    }


def driver_response_test(windows_true: np.ndarray, y_pred_dict: dict,
                          masks: np.ndarray) -> dict:
    """
    Physiological plausibility test for imputation.

    For each window where a bolus or carb event falls inside the masked gap,
    compare the sign of the imputed CGM change (gap_start → gap_end) to the sign
    of the true CGM change. A physiologically correct model should read PI / RA
    (always visible, never masked) and reproduce the expected direction even when
    CGM itself is masked.

    Returns per-method accuracy for bolus-in-gap and carb-in-gap subsets.
    Linear interpolation has no access to driver signals → ~50% random accuracy.
    MTSM, trained to attend to PI/RA, should show materially higher accuracy.
    """
    IDX_B, IDX_C, IDX_G = 5, 6, 0   # bolus, carbs, CGM feature indices

    results = {}
    for method, y_pred in y_pred_dict.items():
        nb, nb_ok = 0, 0
        nc, nc_ok = 0, 0

        for i in range(len(windows_true)):
            m = masks[i].astype(bool)
            if not m.any():
                continue
            gs = int(np.where(m)[0][0])
            ge = int(np.where(m)[0][-1]) + 1

            has_bolus = bool(windows_true[i, gs:ge, IDX_B].any())
            has_carb  = bool(windows_true[i, gs:ge, IDX_C].any())
            if not (has_bolus or has_carb):
                continue

            true_dir = float(windows_true[i, ge - 1, IDX_G]
                             - windows_true[i, gs,     IDX_G])
            pred_dir = float(y_pred[i, ge - 1] - y_pred[i, gs])

            if has_bolus:
                nb += 1
                if np.sign(pred_dir) == np.sign(true_dir):
                    nb_ok += 1
            if has_carb:
                nc += 1
                if np.sign(pred_dir) == np.sign(true_dir):
                    nc_ok += 1

        results[method] = {
            'bolus_direction_acc': nb_ok / max(nb, 1),
            'carb_direction_acc':  nc_ok / max(nc, 1),
            'n_bolus_windows':     nb,
            'n_carb_windows':      nc,
        }
    return results


def plot_imputation_examples(windows_true: np.ndarray, y_pred_dict: dict,
                              masks: np.ndarray,
                              scaler_means: np.ndarray, scaler_stds: np.ndarray,
                              path: str, n_examples: int = 6) -> None:
    """
    n_examples side-by-side panels showing: true CGM (observed + hidden),
    each method's gap imputation, and PI/RA context on twin y-axis.
    Selects windows with the most dynamic CGM range inside the gap.
    """
    IDX_G, IDX_PI, IDX_RA = 0, 1, 2
    _col = {'fm': '#2563EB', 'raw': '#DC2626', 'linear': '#6B7280'}
    _lbl = {'fm': 'FM (zero-shot)', 'raw': 'Raw head (trained)', 'linear': 'Linear interp.'}
    t    = np.arange(288) * 5 / 60   # hours

    # Pick most dynamic windows
    true_z    = windows_true[:, :, IDX_G]
    m_bool    = masks.astype(bool)
    dyn_score = np.array([
        true_z[i, m_bool[i]].max() - true_z[i, m_bool[i]].min()
        if m_bool[i].any() else 0.0 for i in range(len(windows_true))
    ])
    sel = np.argsort(dyn_score)[-n_examples:][::-1]

    ncols = min(n_examples, 3)
    nrows = (n_examples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten()

    for panel, (ax, idx) in enumerate(zip(axes_flat[:n_examples], sel)):
        sm, ss = float(scaler_means[idx]), float(scaler_stds[idx])
        m      = m_bool[idx]
        gs, ge = int(np.where(m)[0][0]), int(np.where(m)[0][-1])

        true_mg_full = true_z[idx] * ss + sm
        ax_drv = ax.twinx()

        ax.axvspan(t[gs], t[ge], alpha=0.10, color='#9CA3AF', zorder=0)
        ax.axhspan(70, 180, alpha=0.05, color='green', zorder=0)

        # Observed part of true CGM
        obs_mg = true_mg_full.copy(); obs_mg[m] = np.nan
        ax.plot(t, obs_mg, color='#111827', lw=2, label='True (observed)', zorder=4)
        # Hidden part (reference, shown faintly)
        hid_mg = true_mg_full.copy(); hid_mg[~m] = np.nan
        ax.plot(t, hid_mg, color='#111827', lw=1.5, ls='--', alpha=0.35, zorder=3)

        # Imputation methods — only draw gap portion
        for method, y_pred in y_pred_dict.items():
            pred_mg = y_pred[idx] * ss + sm
            gap_mg  = pred_mg.copy(); gap_mg[~m] = np.nan
            ax.plot(t, gap_mg, color=_col.get(method, 'grey'), lw=2,
                    label=_lbl.get(method, method), zorder=5)

        # PI / RA on twin axis
        ax_drv.plot(t, windows_true[idx, :, IDX_PI], color='#7C3AED',
                    lw=1, alpha=0.6, label='PI')
        ax_drv.plot(t, windows_true[idx, :, IDX_RA], color='#059669',
                    lw=1, alpha=0.6, label='RA')
        ax_drv.set_ylabel('PI / RA (z)', fontsize=7, color='#6B7280')
        ax_drv.tick_params(labelsize=6)
        ax_drv.spines[['top']].set_visible(False)

        ax.set_ylabel('CGM (mg/dL)', fontsize=8)
        ax.set_xlabel('Time (h)', fontsize=8)
        ax.set_title(f'Example {panel + 1}  ({int(m.sum()) * 5} min gap)', fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.spines[['top', 'right']].set_visible(False)

    axes_flat[0].legend(fontsize=7, loc='upper left', ncol=2)
    for ax in axes_flat[n_examples:]:
        ax.set_visible(False)

    plt.suptitle('CGM Gap Imputation — Method Comparison', fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_imputation_by_gap(all_gap_metrics: dict, gap_labels: list, path: str) -> None:
    """3-panel figure: RMSE / MAE / Within-15 across gap lengths per method."""
    _col = {'fm': '#2563EB', 'raw': '#DC2626', 'linear': '#6B7280'}
    _lbl = {'fm': 'FM (zero-shot)', 'raw': 'Raw head (trained)', 'linear': 'Linear interp.'}
    methods = list(next(iter(all_gap_metrics.values())).keys())
    x = np.arange(len(gap_labels))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (key, ylabel) in zip(axes, [
            ('RMSE_mg', 'RMSE (mg/dL)'),
            ('MAE_mg',  'MAE (mg/dL)'),
            ('within15', 'Within-15 rule (%)')]):
        for method in methods:
            vals = [all_gap_metrics[gl][method].get(key, np.nan) for gl in gap_labels]
            ax.plot(x, vals, marker='o', lw=2,
                    color=_col.get(method, 'grey'), label=_lbl.get(method, method))
        ax.set_xticks(x)
        ax.set_xticklabels(gap_labels, fontsize=9)
        ax.set_xlabel('Gap length')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].legend(fontsize=9)
    plt.suptitle('Imputation Performance by Gap Length', fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_driver_response(windows_true: np.ndarray, y_pred_dict: dict,
                          masks: np.ndarray,
                          scaler_means: np.ndarray, scaler_stds: np.ndarray,
                          path: str, window_half: int = 24) -> None:
    """
    Mean CGM trajectory ±window_half steps around driver events that fall inside
    the masked gap. Compares each method's imputation vs true CGM. One row per
    driver type (bolus, carbs).
    """
    IDX_B, IDX_C, IDX_G = 5, 6, 0
    _col = {'fm': '#2563EB', 'raw': '#DC2626', 'linear': '#6B7280'}
    _lbl = {'fm': 'FM (zero-shot)', 'raw': 'Raw head (trained)', 'linear': 'Linear interp.'}

    t_ax = np.arange(-window_half, window_half + 1) * 5   # minutes

    traces = {driver: {'true': [], **{m: [] for m in y_pred_dict}}
              for driver in ('bolus', 'carbs')}

    for i in range(len(windows_true)):
        m  = masks[i].astype(bool)
        sm = float(scaler_means[i]); ss = float(scaler_stds[i])
        true_mg = windows_true[i, :, IDX_G] * ss + sm

        for driver, idx_d in (('bolus', IDX_B), ('carbs', IDX_C)):
            events = np.where(m & (windows_true[i, :, idx_d] > 0))[0]
            for ev in events:
                lo, hi = ev - window_half, ev + window_half + 1
                if lo < 0 or hi > 288:
                    continue
                traces[driver]['true'].append(true_mg[lo:hi])
                for method, y_pred in y_pred_dict.items():
                    pred_mg = y_pred[i] * ss + sm
                    traces[driver][method].append(pred_mg[lo:hi])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, driver in zip(axes, ('bolus', 'carbs')):
        td = traces[driver]
        if not td['true']:
            ax.text(0.5, 0.5, f'No {driver} events in gap', ha='center', va='center',
                    transform=ax.transAxes)
            continue
        true_arr = np.array(td['true'])
        ax.plot(t_ax, true_arr.mean(0), color='#111827', lw=2.5, label=f'True (n={len(true_arr)})')
        ax.fill_between(t_ax, true_arr.mean(0) - true_arr.std(0),
                               true_arr.mean(0) + true_arr.std(0), alpha=0.12, color='#111827')
        for method in y_pred_dict:
            if not td[method]:
                continue
            arr = np.array(td[method])
            ax.plot(t_ax, arr.mean(0), color=_col.get(method, 'grey'), lw=2,
                    label=_lbl.get(method, method))
        ax.axvline(0, color='grey', ls='--', lw=1, alpha=0.6, label=f'{driver.capitalize()} event')
        ax.set_xlabel('Time from event (min)', fontsize=9)
        ax.set_ylabel('CGM (mg/dL)', fontsize=9)
        ax.set_title(f'Mean response — {driver} events in gap', fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle('Physiological Causality — Imputed CGM Response to Driver Events', fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def save_imputation_table(all_gap_metrics: dict, gap_labels: list, path: str) -> None:
    """Save imputation metrics as CSV (one row per gap × method combination)."""
    import csv
    cols = ['gap', 'method', 'RMSE_mg', 'MAE_mg', 'R2_z', 'within15', 'mean_error', 'n_masked']
    rows = []
    for gl in gap_labels:
        for method, m in all_gap_metrics[gl].items():
            rows.append({
                'gap':        gl,
                'method':     method,
                'RMSE_mg':    f"{m.get('RMSE_mg',    float('nan')):.2f}",
                'MAE_mg':     f"{m.get('MAE_mg',     float('nan')):.2f}",
                'R2_z':       f"{m.get('R2_z',       float('nan')):.4f}",
                'within15':   f"{m.get('within15',   float('nan')):.1f}",
                'mean_error': f"{m.get('mean_error', float('nan')):.2f}",
                'n_masked':   str(m.get('n_masked', 0)),
            })
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_table(all_metrics: dict, path: str) -> None:
    """Save model-level comparison as CSV (one row per model, h0=t+5 and h23=t+120)."""
    cols = ['model',
            'R2_t5', 'R2_t120',
            'RMSE_t5', 'RMSE_t120',
            'MAE_t5',  'MAE_t120',
            'MARD_t5', 'within15_t5',
            'Clarke_A_t5', 'Clarke_A_t120',
            'bias_t5',
            'epochs', 'time_min']

    def _f(m, key, fmt='.4f'):
        v = m.get(key, float('nan'))
        return format(v, fmt) if v == v else 'nan'   # nan-safe

    rows = []
    for tag, m in all_metrics.items():
        rows.append({
            'model':          tag,
            'R2_t5':          _f(m, 'R2_h0'),
            'R2_t120':        _f(m, 'R2_h23'),
            'RMSE_t5':        _f(m, 'RMSE_h0',  '.2f'),
            'RMSE_t120':      _f(m, 'RMSE_h23', '.2f'),
            'MAE_t5':         _f(m, 'MAE_h0',   '.2f'),
            'MAE_t120':       _f(m, 'MAE_h23',  '.2f'),
            'MARD_t5':        _f(m, 'MARD_h0',  '.2f'),
            'within15_t5':    _f(m, 'within15_h0', '.2f'),
            'Clarke_A_t5':    _f(m, 'Clarke_5min_A'),
            'Clarke_A_t120':  _f(m, 'Clarke_120min_A'),
            'bias_t5':        _f(m, 'mean_error_h0', '.2f'),
            'epochs':         str(int(m['epochs_trained'])) if 'epochs_trained' in m else '-',
            'time_min':       _f(m, 'train_time_min', '.1f') if 'train_time_min' in m else '-',
        })

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
