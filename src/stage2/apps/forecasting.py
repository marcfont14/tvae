import os
import numpy as np

from src.encoder import load_encoder, build_encoder
from src.stage2.data import load_all_patients, make_forecasting_dataset, HORIZON_LABELS
from src.stage2.models import build_forecasting_transformer, build_forecasting_lstm
from src.stage2.train import train
from src.stage2.evaluate import (regression_metrics, clarke_zones, save_metrics,
                                  print_metrics, plot_training_curves,
                                  plot_prediction_scatter, plot_clarke_grid)

VARIANTS = [
    ('fm', 'transformer'),
    ('fm', 'lstm'),
    ('ts', 'transformer'),
    ('ts', 'lstm'),
]


def _build_model(mode: str, arch: str) -> object:
    encoder = load_encoder(trainable=False) if mode == 'fm' else build_encoder()
    if arch == 'transformer':
        return build_forecasting_transformer(encoder)
    return build_forecasting_lstm(encoder)


def run(args):
    print(f'\n=== Forecasting | run_id={args.run_id} | mode={args.mode} ===\n')

    out_dir = os.path.join('results', 'stage2', 'forecasting', args.run_id)
    os.makedirs(out_dir, exist_ok=True)

    patients = load_all_patients(args.data, max_patients=args.max_patients)
    splits   = make_forecasting_dataset(patients, batch_size=getattr(args, 'batch_size', 128))

    # Determine which variants to run
    if args.mode == 'both':
        run_variants = VARIANTS
    else:
        run_variants = [(m, a) for m, a in VARIANTS if m == args.mode]

    histories, all_metrics = {}, {}

    for mode, arch in run_variants:
        tag = f'{mode}_{arch}'
        print(f'\n--- {tag.upper()} ---')

        lr      = (getattr(args, 'lr',    1e-3) if mode == 'fm'
                   else getattr(args, 'lr_ts', 1e-4))
        patience = 10 if mode == 'fm' else 15

        model   = _build_model(mode, arch)
        history = train(model, splits,
                        run_id=f'{args.run_id}_{tag}',
                        results_dir=os.path.join('results', 'stage2', 'forecasting'),
                        epochs=args.epochs,
                        lr=lr,
                        patience=patience)

        y_test = np.concatenate([y for _, y in splits['test']], axis=0)
        y_pred = model.predict(splits['test'], verbose=0)

        metrics = regression_metrics(y_test, y_pred)
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

        histories[tag]   = history
        all_metrics[tag] = metrics

    if histories:
        plot_training_curves(histories, os.path.join(out_dir, 'training_curves.png'))

    if len(all_metrics) > 1:
        _print_comparison(all_metrics)


def _print_comparison(results: dict):
    print('\n=== Variant Comparison ===')
    scalar_keys = [k for k in next(iter(results.values())) if 'Clarke' not in k]
    header = f'  {"Metric":<22}' + ''.join(f'{tag:>18}' for tag in results)
    print(header)
    print('  ' + '-' * (22 + 18 * len(results)))
    for k in scalar_keys:
        row = f'  {k:<22}' + ''.join(f'{results[tag][k]:>18.4f}' for tag in results)
        print(row)
