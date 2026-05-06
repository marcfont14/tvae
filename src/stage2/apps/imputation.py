import os
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.stage2.data import (
    load_all_patients,
    make_imputation_dataset, make_eval_imputation_tf, make_eval_imputation_numpy,
    IMPUTATION_GAP_LENGTHS, IMPUTATION_GAP_LABELS,
)
from src.stage2.models import build_mtsm_imputation_model, build_raw_imputation_model
from src.stage2.evaluate import (
    linear_interpolate, imputation_metrics, driver_response_test,
    plot_imputation_examples, plot_imputation_by_gap, plot_driver_response,
    save_imputation_table, save_metrics,
)


# ── Custom masked loss / metric ───────────────────────────────────────────────

def _masked_mse(y_true, y_pred):
    """y_true: (B, 288, 2) = [target_z, mask]; y_pred: (B, 288)."""
    target   = y_true[:, :, 0]
    mask     = y_true[:, :, 1]
    sq_err   = tf.square(target - y_pred) * mask
    n_masked = tf.reduce_sum(mask, axis=1) + 1e-8
    return tf.reduce_mean(tf.reduce_sum(sq_err, axis=1) / n_masked)


def _masked_mae_metric(y_true, y_pred):
    target = y_true[:, :, 0]
    mask   = y_true[:, :, 1]
    return tf.reduce_sum(tf.abs(target - y_pred) * mask) / (tf.reduce_sum(mask) + 1e-8)


# ── Raw model training ────────────────────────────────────────────────────────

def _train_raw(model, splits, out_dir, epochs=50, lr=1e-3, patience=15):
    """
    Train raw imputation model with a fresh val dataset each epoch to avoid
    TF from_generator state exhaustion (same fix as forecasting run06).
    Saves best weights to out_dir/raw_weights.weights.h5.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=_masked_mse,
        metrics=[_masked_mae_metric],
    )
    best_val_loss  = float('inf')
    patience_count = 0
    best_weights   = None
    history        = {'loss': [], 'val_loss': [], 'val_masked_mae': []}

    for epoch in range(epochs):
        h = model.fit(splits['train'], epochs=1,
                      steps_per_epoch=splits['steps_per_epoch'], verbose=0)
        train_loss = float(h.history['loss'][0])

        val_ds  = make_eval_imputation_tf(
            splits['val_patients'], splits['batch_size'],
            splits['gap_min'], splits['gap_max'], seed=42,
        )
        val_res  = model.evaluate(val_ds, verbose=0)
        val_loss = float(val_res[0])
        val_mae  = float(val_res[1])

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_masked_mae'].append(val_mae)
        print(f'  Epoch {epoch + 1:3d}/{epochs}  '
              f'loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_mae={val_mae:.4f}',
              flush=True)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss  = val_loss
            patience_count = 0
            best_weights   = model.get_weights()
            model.save_weights(os.path.join(out_dir, 'raw_weights.weights.h5'))
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f'  Early stopping at epoch {epoch + 1}', flush=True)
                break

    if best_weights is not None:
        model.set_weights(best_weights)
    return history, epoch + 1


# ── Main run ──────────────────────────────────────────────────────────────────

def run(args):
    print(f'\n=== Imputation | run_id={args.run_id} ===\n')
    out_dir = os.path.join('results', 'stage2', 'imputation', args.run_id)
    os.makedirs(out_dir, exist_ok=True)

    patients = load_all_patients(args.data, max_patients=args.max_patients)
    splits   = make_imputation_dataset(patients, batch_size=getattr(args, 'batch_size', 128))

    # ── 1. FM zero-shot (frozen encoder2 + pre-trained reconstruction head) ──
    print('\n--- FM (zero-shot, encoder2) ---')
    mtsm = build_mtsm_imputation_model()
    print(f'  Params: {mtsm.count_params():,}  (frozen — no task-specific training)')

    # ── 2. Raw head (same Dense(64)→Dense(1), no encoder, trained from scratch)
    print('\n--- Raw head (trained from scratch, no encoder) ---')
    raw = build_raw_imputation_model()
    print(f'  Params: {raw.count_params():,}')

    t0 = time.time()
    history, epochs_run = _train_raw(
        raw, splits, out_dir,
        epochs=getattr(args, 'epochs', 50),
        lr=getattr(args, 'lr_raw', 1e-3),
        patience=15,
    )
    train_sec = time.time() - t0
    print(f'  Training: {train_sec / 60:.1f} min  ({epochs_run} epochs)')

    with open(os.path.join(out_dir, 'raw_training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # ── 3. Evaluate all methods at each gap length ────────────────────────────
    print('\n--- Evaluation by gap length ---')
    all_gap_metrics = {}

    for gap_len, gap_label in zip(IMPUTATION_GAP_LENGTHS, IMPUTATION_GAP_LABELS):
        print(f'\n  Gap: {gap_label} ({gap_len} steps = {gap_len * 5} min)', flush=True)

        ev       = make_eval_imputation_numpy(splits['test_patients'], gap_len)
        W_orig   = ev['windows_orig']    # (N, 288, 10)
        W_masked = ev['windows_masked']  # (N, 288, 10)
        masks    = ev['masks']           # (N, 288)
        s_mean   = ev['scaler_means']
        s_std    = ev['scaler_stds']
        true_z   = W_orig[:, :, 0]      # CGM z-score

        mtsm_z = mtsm.predict(W_masked, batch_size=256, verbose=0)   # (N, 288)
        raw_z  = raw.predict(W_masked,  batch_size=256, verbose=0)   # (N, 288)
        lin_z  = linear_interpolate(W_masked, masks)                  # (N, 288)

        gap_metrics = {}
        for tag, pred_z in [('fm', mtsm_z), ('raw', raw_z), ('linear', lin_z)]:
            m = imputation_metrics(true_z, pred_z, masks, s_mean, s_std)
            gap_metrics[tag] = m
            print(f'    {tag:8s}  RMSE={m["RMSE_mg"]:.2f} mg/dL  '
                  f'MAE={m["MAE_mg"]:.2f}  R²={m["R2_z"]:.4f}  '
                  f'within15={m["within15"]:.1f}%', flush=True)

        all_gap_metrics[gap_label] = gap_metrics

        with open(os.path.join(out_dir, f'metrics_{gap_label}.json'), 'w') as f:
            json.dump(gap_metrics, f, indent=2)

        # Visual examples at 6h gap (centre of MTSM training distribution)
        if gap_label == '6h':
            y_pred_dict = {'fm': mtsm_z, 'raw': raw_z, 'linear': lin_z}

            plot_imputation_examples(
                W_orig, y_pred_dict, masks, s_mean, s_std,
                os.path.join(out_dir, 'imputation_examples.png'),
            )

            # Physiological causality test
            print('\n  Driver response test (2h gap):', flush=True)
            drv = driver_response_test(W_orig, y_pred_dict, masks)
            for tag, d in drv.items():
                print(f'    {tag:8s}  bolus_acc={d["bolus_direction_acc"]:.3f} '
                      f'(n={d["n_bolus_windows"]:4d})  '
                      f'carb_acc={d["carb_direction_acc"]:.3f} '
                      f'(n={d["n_carb_windows"]:4d})', flush=True)
            with open(os.path.join(out_dir, 'driver_response.json'), 'w') as f:
                json.dump(drv, f, indent=2)

            # Mean response curve around driver events
            plot_driver_response(
                W_orig, y_pred_dict, masks, s_mean, s_std,
                os.path.join(out_dir, 'driver_response.png'),
            )

    # ── 4. Summary outputs ────────────────────────────────────────────────────
    plot_imputation_by_gap(
        all_gap_metrics, IMPUTATION_GAP_LABELS,
        os.path.join(out_dir, 'imputation_by_gap.png'),
    )
    save_imputation_table(
        all_gap_metrics, IMPUTATION_GAP_LABELS,
        os.path.join(out_dir, 'comparison_table.csv'),
    )

    # ── 5. Print comparison table ─────────────────────────────────────────────
    print('\n=== Imputation RMSE (mg/dL) by gap length ===')
    methods = ['fm', 'raw', 'linear']
    hdr = f'  {"Method":<10}' + ''.join(f'{gl:>10}' for gl in IMPUTATION_GAP_LABELS)
    print(hdr)
    print('  ' + '-' * (10 + 10 * len(IMPUTATION_GAP_LABELS)))
    for method in methods:
        row = f'  {method:<10}' + ''.join(
            f'{all_gap_metrics[gl][method]["RMSE_mg"]:>10.2f}'
            for gl in IMPUTATION_GAP_LABELS
        )
        print(row)

    print(f'\n  Results saved to {out_dir}/')
