import os
import time
import numpy as np

from src.encoder import load_encoder
from src.stage2.data import (load_all_patients, make_forecasting_dataset,
                              make_eval_dataset, naive_forecast, HORIZON_LABELS,
                              load_patient, LOOKAHEAD, N_HORIZONS, IDX_CGM,
                              CONTIGUITY_THRESHOLD)
from src.stage2.models import (build_forecasting_lstm, build_forecasting_lstm_query,
                               build_forecasting_lstm_hcls, build_raw_forecasting_lstm)
from src.stage2.train import train
from src.stage2.evaluate import (regression_metrics, clarke_zones, save_metrics,
                                  print_metrics, plot_training_curves,
                                  plot_prediction_scatter, plot_clarke_grid,
                                  plot_horizon_comparison, plot_forecast_traces,
                                  save_comparison_table)

# Variants: Raw / FM+AttentionPool / FM+QueryCrossAttention / FM+hcls
VARIANTS = [
    ('raw',      'lstm'),
    ('fm',       'lstm'),
    ('fm_query', 'lstm'),
    ('fm_hcls',  'lstm'),
]

_MODE_SETS = {
    'raw':      {'raw'},
    'fm':       {'fm'},
    'fm_query': {'fm_query'},
    'fm_hcls':  {'fm_hcls'},
    'all':      {'raw', 'fm', 'fm_query', 'fm_hcls'},
}


def _build_model(mode: str, arch: str) -> object:
    if mode == 'fm':
        encoder = load_encoder(trainable=False)
        return build_forecasting_lstm(encoder)
    elif mode == 'fm_query':
        encoder = load_encoder(trainable=False)
        return build_forecasting_lstm_query(encoder)
    elif mode == 'fm_hcls':
        encoder = load_encoder(trainable=False)
        return build_forecasting_lstm_hcls(encoder)
    else:  # raw — pure LSTM, no encoder
        return build_raw_forecasting_lstm()


def _collect_samples(test_patients, n=12, seed=42, max_patients=5):
    """Load n contiguous windows from the first few test patients for forecast visualisation."""
    rng        = np.random.default_rng(seed)
    candidates = []
    for path, no_age in test_patients[:max_patients]:
        try:
            windows, mean, std = load_patient(path, no_age)
        except Exception:
            continue
        for i in range(len(windows) - LOOKAHEAD):
            delta_z = abs(float(windows[i, -1, IDX_CGM])
                          - float(windows[i + LOOKAHEAD, 0, IDX_CGM]))
            if delta_z > CONTIGUITY_THRESHOLD:
                continue
            context_mg = (windows[i, :, IDX_CGM] * std + mean).astype(np.float32)
            future_mg  = (windows[i + LOOKAHEAD, :N_HORIZONS, IDX_CGM] * std + mean
                          ).astype(np.float32)
            candidates.append((windows[i], context_mg, future_mg))
    if not candidates:
        return []
    idx = rng.choice(len(candidates), size=min(n, len(candidates)), replace=False)
    return [candidates[i] for i in sorted(idx)]


def run(args):
    print(f'\n=== Forecasting | run_id={args.run_id} | mode={args.mode} ===\n')

    out_dir = os.path.join('results', 'stage2', 'forecasting', args.run_id)
    os.makedirs(out_dir, exist_ok=True)

    patients = load_all_patients(args.data, max_patients=args.max_patients)
    splits   = make_forecasting_dataset(patients, batch_size=getattr(args, 'batch_size', 128))

    allowed      = _MODE_SETS[args.mode]
    run_variants = [(m, a) for m, a in VARIANTS if m in allowed]

    histories, all_metrics, trained_models = {}, {}, {}

    for mode, arch in run_variants:
        tag = f'{mode}_{arch}'
        print(f'\n--- {tag.upper()} ---')

        if mode in ('fm', 'fm_query'):
            lr, patience = getattr(args, 'lr',     1e-3), 10
        else:  # raw
            lr, patience = getattr(args, 'lr_raw', 1e-4), 15

        model     = _build_model(mode, arch)
        # Fresh val dataset per variant — avoids TF from_generator state exhaustion
        # when the same dataset object is reused across multiple model.fit() calls.
        variant_splits = {
            **splits,
            'val': make_eval_dataset(splits['val_patients'], splits['batch_size']),
        }
        t0        = time.time()
        history   = train(model, variant_splits,
                          run_id=tag,
                          results_dir=out_dir,
                          epochs=args.epochs,
                          lr=lr,
                          patience=patience)
        train_sec  = time.time() - t0
        epochs_run = len(history.history['loss'])
        print(f'  Training time: {train_sec/60:.1f} min  '
              f'({epochs_run} epochs, {train_sec/epochs_run:.1f} s/epoch)')

        y_test = np.concatenate([y for _, y in splits['test']], axis=0)
        y_pred = model.predict(splits['test'], verbose=0)

        metrics = regression_metrics(y_test, y_pred)
        metrics['train_time_min'] = round(train_sec / 60, 2)
        metrics['epochs_trained'] = epochs_run
        metrics['sec_per_epoch']  = round(train_sec / epochs_run, 1)
        for i, label in enumerate(HORIZON_LABELS):
            zones = clarke_zones(y_test[:, i], y_pred[:, i])
            for z, v in zones.items():
                metrics[f'Clarke_{label}_{z}'] = v

        print_metrics({k: v for k, v in metrics.items() if 'Clarke' not in k})
        for i, label in enumerate(HORIZON_LABELS):
            zones = {z: metrics[f'Clarke_{label}_{z}'] for z in 'ABCDE'}
            print(f'  Clarke t+{label}: ' +
                  '  '.join(f'{z}={zones[z]:.1%}' for z in 'ABCDE'))

        save_metrics(metrics, os.path.join(out_dir, f'metrics_{tag}.json'))
        plot_prediction_scatter(y_test, y_pred,
                                os.path.join(out_dir, f'scatter_{tag}.png'))
        plot_clarke_grid(y_test, y_pred,
                         os.path.join(out_dir, f'clarke_{tag}.png'))

        histories[tag]      = history
        all_metrics[tag]    = metrics
        trained_models[tag] = model

    # Naive baseline — last-value carry-forward
    print('\n--- NAIVE BASELINE ---')
    naive_pred, naive_true = naive_forecast(splits['test_patients'])
    naive_metrics = regression_metrics(naive_true, naive_pred)
    for i, label in enumerate(HORIZON_LABELS):
        zones = clarke_zones(naive_true[:, i], naive_pred[:, i])
        for z, v in zones.items():
            naive_metrics[f'Clarke_{label}_{z}'] = v
    print_metrics({k: v for k, v in naive_metrics.items() if 'Clarke' not in k})
    save_metrics(naive_metrics, os.path.join(out_dir, 'metrics_naive.json'))
    all_metrics['naive'] = naive_metrics

    if histories:
        plot_training_curves(histories, os.path.join(out_dir, 'training_curves.png'))

    plot_horizon_comparison(all_metrics, HORIZON_LABELS,
                            os.path.join(out_dir, 'horizon_comparison.png'))

    # Forecast trace visualisation — same windows, all models side by side
    print('\n  Collecting sample windows for forecast traces...')
    samples = _collect_samples(splits['test_patients'])
    if samples and trained_models:
        plot_forecast_traces(samples, trained_models,
                             os.path.join(out_dir, 'forecast_traces.png'))
        print(f'  Saved forecast_traces.png  ({len(samples)} windows)')

    # Comparison table
    save_comparison_table(all_metrics, os.path.join(out_dir, 'comparison_table.csv'))
    print(f'\n  Saved comparison_table.csv')

    if len(all_metrics) > 1:
        _print_comparison(all_metrics)


def _print_comparison(results: dict):
    print('\n=== Variant Comparison ===')
    scalar_keys = [k for k in next(iter(results.values())) if 'Clarke' not in k]
    header = f'  {"Metric":<22}' + ''.join(f'{tag:>18}' for tag in results)
    print(header)
    print('  ' + '-' * (22 + 18 * len(results)))
    for k in scalar_keys:
        row = f'  {k:<22}' + ''.join(
            f'{results[tag].get(k, float("nan")):>18.4f}' for tag in results)
        print(row)
