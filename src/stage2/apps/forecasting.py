import gc
import os
import time
import numpy as np

from src.encoder import load_encoder, load_decoder, load_ntp_head
from src.stage2.data import (load_all_patients, make_forecasting_dataset,
                              make_eval_dataset, make_ar_eval_data, naive_forecast,
                              HORIZON_LABELS, load_patient, LOOKAHEAD, N_HORIZONS,
                              IDX_CGM, CONTIGUITY_THRESHOLD)
from src.stage2.models import (build_forecasting_lstm, build_forecasting_lstm_decoder,
                               build_raw_forecasting_lstm, predict_ar_decoder)
from src.stage2.train import train
from src.stage2.evaluate import (regression_metrics, clarke_zones, save_metrics,
                                  print_metrics, plot_training_curves,
                                  plot_prediction_scatter, plot_clarke_grid,
                                  plot_horizon_comparison, plot_forecast_traces,
                                  save_comparison_table)

VARIANTS = [
    ('raw',          'lstm'),
    ('fm',           'lstm'),
    ('fm_ft',        'lstm'),
    ('fm_decoder',   'lstm'),   # decoder frozen — AR rollout, no LSTM head
    ('fm_decoder_ft','lstm'),   # decoder fine-tuned — h_last + LSTM head
]

_MODE_SETS = {
    'raw':           {'raw'},
    'fm':            {'fm'},
    'fm_ft':         {'fm_ft'},
    'fm_decoder':    {'fm_decoder'},
    'fm_decoder_ft': {'fm_decoder_ft'},
    'thesis':        {'raw', 'fm', 'fm_ft', 'fm_decoder', 'fm_decoder_ft'},
}


def _build_model(mode: str, arch: str) -> object:
    if mode == 'fm':
        return build_forecasting_lstm(load_encoder(trainable=False))
    elif mode == 'fm_ft':
        return build_forecasting_lstm(load_encoder(trainable=True))
    elif mode == 'fm_decoder_ft':
        return build_forecasting_lstm_decoder(load_decoder(trainable=True))
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
    splits   = make_forecasting_dataset(patients, batch_size=getattr(args, 'batch_size', 128),
                                         max_train_patients=getattr(args, 'max_train_patients', None))

    allowed      = _MODE_SETS[args.mode]
    run_variants = [(m, a) for m, a in VARIANTS if m in allowed]

    histories, all_metrics, trained_models = {}, {}, {}

    for mode, arch in run_variants:
        tag = f'{mode}_{arch}'
        if os.path.exists(os.path.join(out_dir, f'metrics_{tag}.json')):
            print(f'\n--- {tag.upper()} --- already done, skipping')
            continue

        import keras
        keras.backend.clear_session()
        gc.collect()

        print(f'\n--- {tag.upper()} ---')

        # ── Autoregressive decoder — no training, no LSTM head ───────────────
        if mode == 'fm_decoder':
            decoder  = load_decoder(trainable=False)
            ntp_head = load_ntp_head()
            ar_data  = make_ar_eval_data(splits['test_patients'])
            MAX_AR = 10_000
            if len(ar_data['windows']) > MAX_AR:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(ar_data['windows']), MAX_AR, replace=False)
                ar_data = {k: v[idx] for k, v in ar_data.items()}
                print(f'  AR eval subsampled to {MAX_AR:,} windows')
            t0 = time.time()
            y_pred = predict_ar_decoder(
                decoder, ntp_head,
                ar_data['windows'], ar_data['scaler_std'],
                N_HORIZONS, ar_data['last_cgm_mg'],
            )
            y_test = ar_data['y_mg']
            ar_sec = time.time() - t0
            print(f'  AR rollout: {len(y_test):,} samples  {ar_sec:.1f}s')

            metrics = regression_metrics(y_test, y_pred)
            metrics['train_time_min'] = 0.0
            metrics['epochs_trained'] = 0
            metrics['sec_per_epoch']  = 0.0
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
            all_metrics[tag] = metrics
            del decoder, ntp_head, ar_data
            gc.collect()
            continue  # skip LSTM train/eval below

        if mode == 'fm':
            lr, patience = getattr(args, 'lr',     1e-3), 10
        elif mode in ('fm_ft', 'fm_decoder_ft'):
            lr, patience = getattr(args, 'lr_ft',  1e-4), 15
        else:  # raw
            lr, patience = getattr(args, 'lr_raw', 1e-4), 15

        model = _build_model(mode, arch)
        eval_only = getattr(args, 'eval_only', False)

        if eval_only:
            weights_path = os.path.join(out_dir, f'weights_{tag}.weights.h5')
            if not os.path.exists(weights_path):
                print(f'  No weights at {weights_path} — skipping')
                continue
            model.compile(optimizer='adam', loss='huber')
            model.load_weights(weights_path)
            print(f'  Loaded weights from {weights_path}')
            train_sec, epochs_run = 0.0, 0
        else:
            # Fresh val dataset per variant with repeat() — fixes TF 2.17 from_generator
            # exhaustion bug where the same iterator is reused across epochs.
            variant_splits = {
                **splits,
                'val': make_eval_dataset(splits['val_patients'], splits['batch_size']).repeat(),
            }
            t0      = time.time()
            history = train(model, variant_splits,
                            run_id=tag,
                            results_dir=out_dir,
                            epochs=args.epochs,
                            lr=lr,
                            patience=patience)
            train_sec  = time.time() - t0
            epochs_run = len(history.history['loss'])
            histories[tag] = history
            print(f'  Training time: {train_sec/60:.1f} min  '
                  f'({epochs_run} epochs, {train_sec/epochs_run:.1f} s/epoch)')

        # Single-pass eval: two separate iterations of the same from_generator
        # dataset can yield different sample counts due to TF prefetch buffering.
        _ys, _ps = [], []
        for _xb, _yb in make_eval_dataset(splits['test_patients'], splits['batch_size']):
            _ys.append(_yb.numpy())
            _ps.append(model.predict_on_batch(_xb))
        y_test = np.concatenate(_ys, axis=0)
        y_pred = np.concatenate(_ps, axis=0)

        metrics = regression_metrics(y_test, y_pred)
        metrics['train_time_min'] = round(train_sec / 60, 2)
        metrics['epochs_trained'] = epochs_run
        metrics['sec_per_epoch']  = round(train_sec / epochs_run, 1) if epochs_run else 0.0
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

        all_metrics[tag] = metrics
        del model
        gc.collect()

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
