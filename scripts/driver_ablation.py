"""
Driver ablation: measure PI/RA continuous curves vs binary flags contribution
to zero-shot gap imputation.

Conditions (feature zeroing applied to W_masked before FM forward pass):
  full       — all features intact (baseline)
  pi_ra_only — binary flags zeroed (indices 5, 6)
  flags_only — continuous PI and RA zeroed (indices 1, 2)
  cgm_only   — both PI/RA and flags zeroed (indices 1, 2, 5, 6)

Run from /mnt/workspace/tvae:
  python -u scripts/driver_ablation.py --data data/processed/adults_global_norm
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stage2.data import (
    load_all_patients, make_imputation_dataset, make_eval_imputation_numpy,
    IMPUTATION_GAP_LENGTHS, IMPUTATION_GAP_LABELS,
    IDX_PI, IDX_RA, IDX_BOLUS, IDX_CARBS,
)
from src.stage2.evaluate import imputation_metrics, linear_interpolate
from src.stage2.models import build_mtsm_imputation_model


ABLATIONS = {
    'full':       [],
    'pi_ra_only': [IDX_BOLUS, IDX_CARBS],
    'flags_only': [IDX_PI, IDX_RA],
    'cgm_only':   [IDX_PI, IDX_RA, IDX_BOLUS, IDX_CARBS],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/processed/adults_global_norm')
    parser.add_argument('--max_patients', type=int, default=None)
    args = parser.parse_args()

    print('\n=== Driver Ablation — Zero-Shot Imputation ===\n')
    print(f'  Data: {args.data}')
    print(f'  Conditions: {list(ABLATIONS.keys())}\n')

    patients = load_all_patients(args.data, max_patients=args.max_patients)
    splits   = make_imputation_dataset(patients, batch_size=128)

    print('Loading FM imputation model...', flush=True)
    mtsm = build_mtsm_imputation_model()
    print(f'  Params: {mtsm.count_params():,}  (frozen)\n')

    # Results: gap_label -> condition -> R2_z
    results = {gl: {} for gl in IMPUTATION_GAP_LABELS}

    for gap_len, gap_label in zip(IMPUTATION_GAP_LENGTHS, IMPUTATION_GAP_LABELS):
        print(f'--- Gap: {gap_label} ---', flush=True)
        ev       = make_eval_imputation_numpy(splits['test_patients'], gap_len)
        W_orig   = ev['windows_orig']     # (N, 288, 10)
        W_masked = ev['windows_masked']   # (N, 288, 10) — CGM zeroed in gap
        masks    = ev['masks']
        s_mean   = ev['scaler_means']
        s_std    = ev['scaler_stds']
        true_z   = W_orig[:, :, 0]

        for cond, zero_idxs in ABLATIONS.items():
            W_in = W_masked.copy()
            for idx in zero_idxs:
                W_in[:, :, idx] = 0.0
            pred_z = mtsm.predict(W_in, batch_size=256, verbose=0)
            m      = imputation_metrics(true_z, pred_z, masks, s_mean, s_std)
            results[gap_label][cond] = m
            print(f'  {cond:<14}  RMSE={m["RMSE_mg"]:5.2f} mg/dL  R²={m["R2_z"]:+.4f}', flush=True)

        print()

    # Summary table
    print('=== Summary: R² by condition and gap ===\n')
    col_w = 12
    header = f'  {"Condition":<14}' + ''.join(f'{gl:>{col_w}}' for gl in IMPUTATION_GAP_LABELS)
    print(header)
    print('  ' + '-' * (14 + col_w * len(IMPUTATION_GAP_LABELS)))
    for cond in ABLATIONS:
        row = f'  {cond:<14}' + ''.join(
            f'{results[gl][cond]["R2_z"]:>{col_w}.4f}' for gl in IMPUTATION_GAP_LABELS
        )
        print(row)

    print('\n=== Summary: RMSE (mg/dL) by condition and gap ===\n')
    print(header)
    print('  ' + '-' * (14 + col_w * len(IMPUTATION_GAP_LABELS)))
    for cond in ABLATIONS:
        row = f'  {cond:<14}' + ''.join(
            f'{results[gl][cond]["RMSE_mg"]:>{col_w}.2f}' for gl in IMPUTATION_GAP_LABELS
        )
        print(row)

    print()


if __name__ == '__main__':
    main()
