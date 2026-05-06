import os
import time
import numpy as np
import tensorflow as tf

from src.encoder import load_encoder, build_encoder
from src.stage2.data import (load_all_patients, make_hypo_dataset,
                              make_eval_hypo_dataset, HYPO_AHEAD)
from src.stage2.models import build_hypo_risk_model
from src.stage2.train import train
from src.stage2.evaluate import (classification_metrics, save_metrics,
                                  print_metrics, plot_training_curves,
                                  plot_roc_curves, plot_pr_curves,
                                  save_classification_table)

# Two variants: FM (frozen MTSM encoder) vs Raw (same arch, random init)
VARIANTS = [('raw', None), ('fm', None)]

_MODE_SETS = {
    'raw': {'raw'},
    'fm':  {'fm'},
    'all': {'raw', 'fm'},
}


def _build_model(mode: str):
    if mode == 'fm':
        encoder = load_encoder(trainable=False)
    else:  # raw — same transformer architecture, random init, trains end-to-end
        encoder = build_encoder()
    return build_hypo_risk_model(encoder)


def _weibull_nll(y_true, y_pred):
    """Weibull negative log-likelihood with right censoring.

    y_true: (B, 2) — [time_to_event, delta]  delta=1 if event observed
    y_pred: (B, 2) — [log_lambda, log_k]
    """
    t     = y_true[:, 0]
    delta = y_true[:, 1]
    lam   = tf.exp(y_pred[:, 0])
    k     = tf.exp(y_pred[:, 1])
    t_lam = t / (lam + 1e-8)
    # event: log f(t) = log(k/lam) + (k-1)*log(t/lam) - (t/lam)^k
    log_ft = tf.math.log(k / (lam + 1e-8)) + (k - 1) * tf.math.log(t_lam + 1e-8) - t_lam ** k
    # censored: log S(t) = -(t/lam)^k
    log_st = -(t_lam ** k)
    ll = delta * log_ft + (1.0 - delta) * log_st
    return -tf.reduce_mean(ll)


def _weibull_risk(y_pred_raw: np.ndarray) -> np.ndarray:
    """P(T ≤ HYPO_AHEAD) = 1 - exp(-(HYPO_AHEAD/lambda)^k) — scalar risk score."""
    lam  = np.exp(y_pred_raw[:, 0])
    k    = np.exp(y_pred_raw[:, 1])
    risk = 1.0 - np.exp(-((HYPO_AHEAD / (lam + 1e-8)) ** k))
    return risk.ravel()


def run(args):
    print(f'\n=== Hypo Risk (nocturnal, Weibull) | run_id={args.run_id} | mode={args.mode} ===\n')

    out_dir = os.path.join('results', 'stage2', 'hypo_risk', args.run_id)
    os.makedirs(out_dir, exist_ok=True)

    patients = load_all_patients(args.data, max_patients=args.max_patients)
    splits   = make_hypo_dataset(patients, batch_size=getattr(args, 'batch_size', 128))

    allowed      = _MODE_SETS[args.mode]
    run_variants = [(m, a) for m, a in VARIANTS if m in allowed]

    histories = {}
    all_metrics, all_preds = {}, {}

    for mode, _ in run_variants:
        tag = f'{mode}_lstm'
        print(f'\n--- {tag.upper()} ---')

        lr      = getattr(args, 'lr',     1e-3) if mode == 'fm' else getattr(args, 'lr_raw', 1e-4)
        patience = 10 if mode == 'fm' else 15

        model = _build_model(mode)
        variant_splits = {
            **splits,
            'val': make_eval_hypo_dataset(splits['val_patients'], splits['batch_size']),
        }

        t0      = time.time()
        history = train(model, variant_splits,
                        run_id=tag,
                        results_dir=out_dir,
                        epochs=args.epochs,
                        lr=lr,
                        patience=patience,
                        loss=_weibull_nll)
        train_sec  = time.time() - t0
        epochs_run = len(history.history['loss'])
        print(f'  Training time: {train_sec/60:.1f} min  '
              f'({epochs_run} epochs, {train_sec/epochs_run:.1f} s/epoch)')

        y_true_raw = np.concatenate([y.numpy() for _, y in splits['test']])
        y_pred_raw = model.predict(splits['test'], verbose=0)

        y_true_binary = y_true_raw[:, 1]           # delta (event indicator)
        risk_scores   = _weibull_risk(y_pred_raw)  # P(hypo within 2h)

        metrics = classification_metrics(y_true_binary, risk_scores)
        metrics['train_time_min'] = round(train_sec / 60, 2)
        metrics['epochs_trained'] = epochs_run
        metrics['sec_per_epoch']  = round(train_sec / epochs_run, 1)

        print_metrics({k: v for k, v in metrics.items()
                       if k not in ('train_time_min', 'epochs_trained', 'sec_per_epoch')})

        save_metrics(metrics, os.path.join(out_dir, f'metrics_{tag}.json'))
        histories[tag]   = history
        all_metrics[tag] = metrics
        all_preds[tag]   = (y_true_binary, risk_scores)

    # Naive baseline: negative last CGM z-score as risk proxy (lower CGM → higher risk)
    print('\n--- NAIVE BASELINE (last CGM) ---')
    naive_true, naive_scores = [], []
    for x_batch, y_batch in splits['test']:
        naive_true.append(y_batch.numpy()[:, 1])
        naive_scores.append(-x_batch.numpy()[:, -1, 0])   # neg z-score of last CGM step
    naive_true   = np.concatenate(naive_true)
    naive_scores = np.concatenate(naive_scores)
    naive_metrics = classification_metrics(naive_true, naive_scores)
    print_metrics(naive_metrics)
    save_metrics(naive_metrics, os.path.join(out_dir, 'metrics_naive.json'))
    all_metrics['naive'] = naive_metrics
    all_preds['naive']   = (naive_true, naive_scores)

    if histories:
        plot_training_curves(histories, os.path.join(out_dir, 'training_curves.png'))

    plot_roc_curves(all_preds, os.path.join(out_dir, 'roc_curves.png'))
    plot_pr_curves(all_preds, splits['pos_rate'],
                   os.path.join(out_dir, 'pr_curves.png'))
    save_classification_table(all_metrics,
                               os.path.join(out_dir, 'comparison_table.csv'))

    print(f'\n  Saved results to {out_dir}')
    _print_comparison(all_metrics)


def _print_comparison(results: dict):
    print('\n=== Variant Comparison ===')
    keys = ['auroc', 'auprc', 'sens_at_90spec', 'f1_optimal', 'prevalence']
    header = f'  {"Metric":<20}' + ''.join(f'{tag:>16}' for tag in results)
    print(header)
    print('  ' + '-' * (20 + 16 * len(results)))
    for k in keys:
        row = f'  {k:<20}' + ''.join(
            f'{results[tag].get(k, float("nan")):>16.4f}' for tag in results)
        print(row)
