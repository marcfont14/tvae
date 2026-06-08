"""
Individual feature ablation on zero-shot imputation.
Extends driver_ablation.py to per-feature resolution.

Conditions: zero out one feature group at a time, measure R² drop vs full.

Run from /mnt/workspace/tvae:
  python -u scripts/feature_ablation.py --data data/processed/adults_global_norm
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stage2.data import (
    load_all_patients, make_imputation_dataset, make_eval_imputation_numpy,
    IMPUTATION_GAP_LENGTHS, IMPUTATION_GAP_LABELS,
)
from src.stage2.evaluate import imputation_metrics
from src.stage2.models import build_mtsm_imputation_model

# Feature index map
FEATURES = {
    'CGM':     [0],
    'PI':      [1],
    'RA':      [2],
    'time':    [3, 4],
    'bolus':   [5],
    'carbs':   [6],
    'therapy': [7, 8, 9],
}

# Ablation conditions: name -> list of indices to zero out
ABLATIONS = {
    'full':         [],
    'no_PI':        [1],
    'no_RA':        [2],
    'no_PI+RA':     [1, 2],
    'no_flags':     [5, 6],
    'no_time':      [3, 4],
    'no_therapy':   [7, 8, 9],
    'drivers_only': [3, 4, 5, 6, 7, 8, 9],   # keep only CGM + PI + RA
    'CGM_only':     [1, 2, 3, 4, 5, 6, 7, 8, 9],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/processed/adults_global_norm')
    parser.add_argument('--max_patients', type=int, default=None)
    args = parser.parse_args()

    print('\n=== Feature Ablation — Zero-Shot Imputation ===\n')

    patients = load_all_patients(args.data, max_patients=args.max_patients)
    splits   = make_imputation_dataset(patients, batch_size=128)

    print('Loading FM imputation model...', flush=True)
    mtsm = build_mtsm_imputation_model()
    print(f'  Params: {mtsm.count_params():,}  (frozen)\n')

    results = {gl: {} for gl in IMPUTATION_GAP_LABELS}

    for gap_len, gap_label in zip(IMPUTATION_GAP_LENGTHS, IMPUTATION_GAP_LABELS):
        print(f'--- Gap: {gap_label} ---', flush=True)
        ev      = make_eval_imputation_numpy(splits['test_patients'], gap_len)
        W_orig  = ev['windows_orig']
        W_masked = ev['windows_masked']
        masks   = ev['masks']
        s_mean  = ev['scaler_means']
        s_std   = ev['scaler_stds']
        true_z  = W_orig[:, :, 0]

        baseline_r2 = None
        for cond, zero_idxs in ABLATIONS.items():
            W_in = W_masked.copy()
            for idx in zero_idxs:
                W_in[:, :, idx] = 0.0
            import tensorflow as tf
            pred_z = mtsm.predict(W_in, batch_size=256, verbose=0)
            m = imputation_metrics(true_z, pred_z, masks, s_mean, s_std)
            results[gap_label][cond] = m
            if cond == 'full':
                baseline_r2 = m['R2_z']
                print(f'  {cond:<16}  R²={m["R2_z"]:+.4f}  (baseline)', flush=True)
            else:
                delta = m['R2_z'] - baseline_r2
                print(f'  {cond:<16}  R²={m["R2_z"]:+.4f}  Δ={delta:+.4f}', flush=True)
        print()

    # Summary table: R² drop vs full
    print('=== R² drop when feature group is zeroed out ===')
    print('(negative = performance loss; larger magnitude = more important)\n')
    col_w = 10
    header = f'  {"Condition":<16}' + ''.join(f'{gl:>{col_w}}' for gl in IMPUTATION_GAP_LABELS)
    print(header)
    print('  ' + '-' * (16 + col_w * len(IMPUTATION_GAP_LABELS)))
    for cond in ABLATIONS:
        if cond == 'full':
            row = f'  {cond:<16}' + ''.join(
                f'{results[gl][cond]["R2_z"]:>{col_w}.4f}' for gl in IMPUTATION_GAP_LABELS
            )
        else:
            full_r2s = [results[gl]['full']['R2_z'] for gl in IMPUTATION_GAP_LABELS]
            row = f'  {cond:<16}' + ''.join(
                f'{results[gl][cond]["R2_z"] - full_r2s[i]:>{col_w}.4f}'
                for i, gl in enumerate(IMPUTATION_GAP_LABELS)
            )
        print(row)
    print()


if __name__ == '__main__':
    main()
