import gc
import json
import os
import time
import numpy as np
import tensorflow as tf

from src.encoder import load_encoder, load_decoder
from src.stage2.data import (load_all_patients, make_hypo_dataset,
                              make_eval_hypo_dataset, HYPO_AHEAD,
                              HYPO_AHEAD_NOCTURNAL)
from src.stage2.models import (build_hypo_risk_model, build_hypo_risk_decoder,
                                build_raw_hypo_risk_model)
from src.stage2.train import train
from src.stage2.evaluate import (classification_metrics, stratified_auroc,
                                  save_metrics, print_metrics,
                                  plot_training_curves, plot_roc_curves,
                                  plot_pr_curves, plot_calibration,
                                  save_classification_table)

# Variants: FM encoder2 frozen/fine-tuned / FM decoder frozen/fine-tuned / Raw LSTM
VARIANTS = [
    ('raw',          None),
    ('fm',           None),
    ('fm_ft',        None),
    ('fm_decoder',   None),
    ('fm_decoder_ft', None),
]

_MODE_SETS = {
    'raw':           {'raw'},
    'fm':            {'fm'},
    'fm_ft':         {'fm_ft'},
    'fm_decoder':    {'fm_decoder'},
    'fm_decoder_ft': {'fm_decoder_ft'},
    'all':           {'raw', 'fm'},
    'all_decoder':   {'raw', 'fm', 'fm_decoder'},
    'thesis':        {'raw', 'fm', 'fm_ft', 'fm_decoder', 'fm_decoder_ft'},
    'de':            {'raw', 'fm', 'fm_ft', 'fm_decoder', 'fm_decoder_ft'},
}


def _build_model(mode: str):
    if mode == 'fm':
        return build_hypo_risk_model(load_encoder(trainable=False))
    elif mode == 'fm_ft':
        return build_hypo_risk_model(load_encoder(trainable=True))
    elif mode == 'fm_decoder':
        return build_hypo_risk_decoder(load_decoder(trainable=False))
    elif mode == 'fm_decoder_ft':
        return build_hypo_risk_decoder(load_decoder(trainable=True))
    else:  # raw — Conv1D + LSTM, no encoder
        return build_raw_hypo_risk_model()


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


def _weibull_risk(y_pred_raw: np.ndarray, horizon: int = HYPO_AHEAD) -> np.ndarray:
    """P(T ≤ horizon) = 1 - exp(-(horizon/lambda)^k) — scalar risk score."""
    lam  = np.exp(np.clip(y_pred_raw[:, 0], -20, 20))
    k    = np.exp(np.clip(y_pred_raw[:, 1], -20, 20))
    risk = 1.0 - np.exp(-((horizon / (lam + 1e-8)) ** k))
    return np.nan_to_num(risk.ravel(), nan=0.5, posinf=1.0, neginf=0.0)


def _run_eval_only(args, out_dir, splits, bedtime_only: bool = False):
    """Load saved weights for each variant, re-evaluate with extended metrics."""
    import tensorflow as tf
    horizon = HYPO_AHEAD_NOCTURNAL if bedtime_only else HYPO_AHEAD

    print('\n--- Collecting test data ---')
    x_list, y_list = [], []
    for x_batch, y_batch in splits['test']:
        x_list.append(x_batch.numpy())
        y_list.append(y_batch.numpy())
    x_all        = np.concatenate(x_list)             # (N, 288, 10)
    y_true_raw   = np.concatenate(y_list)             # (N, 2)
    last_cgm_z   = x_all[:, -1, 0]                    # last observed CGM z-score
    y_true_binary = y_true_raw[:, 1]                  # delta (event indicator)
    print(f'  Test windows: {len(x_all):,}  |  hypo rate: {y_true_binary.mean():.3f}')

    allowed      = _MODE_SETS[args.mode]
    run_variants = [(m, a) for m, a in VARIANTS if m in allowed]

    all_metrics, all_preds = {}, {}

    for mode, _ in run_variants:
        tag          = f'{mode}_lstm'
        weights_path = os.path.join(out_dir, f'weights_{tag}.weights.h5')
        if not os.path.exists(weights_path):
            print(f'\n--- {tag.upper()} --- no weights found, skipping')
            continue

        import keras
        keras.backend.clear_session()
        gc.collect()

        print(f'\n--- {tag.upper()} ---')
        model = _build_model(mode)
        model(tf.zeros((1, 288, 10)))
        model.load_weights(weights_path)

        y_pred_raw  = model.predict(x_all, batch_size=256, verbose=0)
        risk_scores = _weibull_risk(y_pred_raw, horizon=horizon)

        metrics = classification_metrics(y_true_binary, risk_scores)
        strat   = stratified_auroc(y_true_binary, risk_scores, last_cgm_z)
        metrics.update(strat)

        existing = os.path.join(out_dir, f'metrics_{tag}.json')
        if os.path.exists(existing):
            with open(existing) as f:
                old = json.load(f)
            for k in ('train_time_min', 'epochs_trained', 'sec_per_epoch'):
                if k in old:
                    metrics[k] = old[k]

        print_metrics({k: v for k, v in metrics.items()
                       if isinstance(v, float) and k not in
                       ('train_time_min', 'sec_per_epoch', 'threshold_optimal')})
        save_metrics(metrics, existing)
        all_metrics[tag] = metrics
        all_preds[tag]   = (y_true_binary, risk_scores)
        del model; gc.collect()

    # Naive baseline
    print('\n--- NAIVE BASELINE ---')
    naive_scores  = -last_cgm_z
    naive_metrics = classification_metrics(y_true_binary, naive_scores)
    naive_strat   = stratified_auroc(y_true_binary, naive_scores, last_cgm_z)
    naive_metrics.update(naive_strat)
    print_metrics({k: v for k, v in naive_metrics.items()
                   if isinstance(v, float) and k not in ('threshold_optimal',)})
    save_metrics(naive_metrics, os.path.join(out_dir, 'metrics_naive.json'))
    all_metrics['naive'] = naive_metrics
    all_preds['naive']   = (y_true_binary, naive_scores)

    plot_roc_curves(all_preds, os.path.join(out_dir, 'roc_curves.png'))
    plot_pr_curves(all_preds, splits['pos_rate'], os.path.join(out_dir, 'pr_curves.png'))
    plot_calibration(all_preds, os.path.join(out_dir, 'calibration.png'))
    save_classification_table(all_metrics, os.path.join(out_dir, 'comparison_table.csv'))

    print(f'\n  Saved to {out_dir}')
    _print_comparison(all_metrics)


def run(args):
    print(f'\n=== Hypo Risk (nocturnal, Weibull) | run_id={args.run_id} | mode={args.mode} ===\n')

    out_dir = os.path.join('results', 'stage2', 'hypo_risk', args.run_id)
    os.makedirs(out_dir, exist_ok=True)

    bedtime_only = getattr(args, 'bedtime_only', False)
    if bedtime_only:
        print('  Bedtime-only mode: filter 20:00–23:59, horizon = 8h (96 steps)')

    patients = load_all_patients(args.data, max_patients=args.max_patients)
    splits   = make_hypo_dataset(patients, batch_size=getattr(args, 'batch_size', 128),
                                  max_train_patients=getattr(args, 'max_train_patients', None),
                                  bedtime_only=bedtime_only)

    if getattr(args, 'eval_only', False):
        _run_eval_only(args, out_dir, splits, bedtime_only=bedtime_only)
        return

    allowed      = _MODE_SETS[args.mode]
    run_variants = [(m, a) for m, a in VARIANTS if m in allowed]

    histories = {}
    all_metrics, all_preds = {}, {}

    for mode, _ in run_variants:
        tag = f'{mode}_lstm'
        if os.path.exists(os.path.join(out_dir, f'metrics_{tag}.json')):
            print(f'\n--- {tag.upper()} --- already done, skipping')
            continue

        import keras
        keras.backend.clear_session()
        gc.collect()

        print(f'\n--- {tag.upper()} ---')

        if mode in ('fm', 'fm_decoder'):
            lr, patience = getattr(args, 'lr',     1e-3), 10
        elif mode in ('fm_ft', 'fm_decoder_ft'):
            lr, patience = getattr(args, 'lr_ft',  1e-4), 15
        else:  # raw
            lr, patience = getattr(args, 'lr_raw', 1e-4), 15

        model = _build_model(mode)
        variant_splits = {
            **splits,
            'val': make_eval_hypo_dataset(splits['val_patients'], splits['batch_size'],
                                          bedtime_only=bedtime_only),
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

        test_x, test_y = [], []
        for xb, yb in splits['test']:
            test_x.append(xb.numpy())
            test_y.append(yb.numpy())
        x_test        = np.concatenate(test_x)
        y_true_raw    = np.concatenate(test_y)
        y_pred_raw    = model.predict(x_test, batch_size=256, verbose=0)

        horizon       = HYPO_AHEAD_NOCTURNAL if bedtime_only else HYPO_AHEAD
        y_true_binary = y_true_raw[:, 1]
        risk_scores   = _weibull_risk(y_pred_raw, horizon=horizon)
        last_cgm_z    = x_test[:, -1, 0]

        metrics = classification_metrics(y_true_binary, risk_scores)
        metrics.update(stratified_auroc(y_true_binary, risk_scores, last_cgm_z))
        metrics['train_time_min'] = round(train_sec / 60, 2)
        metrics['epochs_trained'] = epochs_run
        metrics['sec_per_epoch']  = round(train_sec / epochs_run, 1)

        print_metrics({k: v for k, v in metrics.items()
                       if k not in ('train_time_min', 'epochs_trained', 'sec_per_epoch')})

        save_metrics(metrics, os.path.join(out_dir, f'metrics_{tag}.json'))
        histories[tag]   = history
        all_metrics[tag] = metrics
        all_preds[tag]   = (y_true_binary, risk_scores)
        del model
        gc.collect()

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
