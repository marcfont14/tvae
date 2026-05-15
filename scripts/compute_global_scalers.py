"""
compute_global_scalers.py
=========================
Computes global mean and std for CGM, PI, and RA across all 1037 patients.

Reads per-patient scaler_mean/scaler_std stored in existing .npz files and
applies the law of total variance to compute pooled population-level statistics
without reloading any window data.

Output: results/outlier_analysis/global_scaler_full.npy
  [cgm_mean, cgm_std, pi_mean, pi_std, ra_mean, ra_std]

Run from /mnt/workspace/tvae:
  python -u scripts/compute_global_scalers.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from tqdm import tqdm

DATA_DIR = 'data/processed/adults'
OUT_PATH = 'results/outlier_analysis/global_scaler_full.npy'


def pooled_stats(means: np.ndarray, stds: np.ndarray):
    """
    Pooled mean and std across patients using the law of total variance.
    Assumes equal weight per patient.

    Var(X) = E[Var(X|patient)] + Var(E[X|patient])
           = mean(stdi^2) + var(meani)
    """
    pooled_mean = float(np.mean(means))
    pooled_var  = float(np.mean(stds ** 2) + np.var(means))
    pooled_std  = float(np.sqrt(pooled_var))
    return pooled_mean, pooled_std


def main():
    npz_files = sorted(Path(DATA_DIR).glob('*.npz'))
    print(f'Found {len(npz_files)} patient files in {DATA_DIR}')

    cgm_means, cgm_stds = [], []
    pi_means,  pi_stds  = [], []
    ra_means,  ra_stds  = [], []

    for path in tqdm(npz_files, desc='Reading scalers'):
        try:
            d = np.load(path, allow_pickle=True)
            m = d['scaler_mean']   # [cgm_mean, pi_mean, ra_mean]
            s = d['scaler_std']    # [cgm_std,  pi_std,  ra_std]
            cgm_means.append(float(m[0])); cgm_stds.append(float(s[0]))
            pi_means.append(float(m[1]));  pi_stds.append(float(s[1]))
            ra_means.append(float(m[2]));  ra_stds.append(float(s[2]))
        except Exception as e:
            print(f'  WARNING: skipped {path.name} — {e}')

    cgm_means = np.array(cgm_means); cgm_stds = np.array(cgm_stds)
    pi_means  = np.array(pi_means);  pi_stds  = np.array(pi_stds)
    ra_means  = np.array(ra_means);  ra_stds  = np.array(ra_stds)

    cgm_mean, cgm_std = pooled_stats(cgm_means, cgm_stds)
    pi_mean,  pi_std  = pooled_stats(pi_means,  pi_stds)
    ra_mean,  ra_std  = pooled_stats(ra_means,  ra_stds)

    print(f'\n=== Global scalers ({len(cgm_means)} patients) ===')
    print(f'  CGM: mean={cgm_mean:.4f}  std={cgm_std:.4f}')
    print(f'  PI:  mean={pi_mean:.4f}   std={pi_std:.4f}')
    print(f'  RA:  mean={ra_mean:.4f}   std={ra_std:.4f}')

    # Sanity check: CGM should match existing global_scaler.npy
    existing = 'results/outlier_analysis/global_scaler.npy'
    if os.path.exists(existing):
        gs = np.load(existing)
        print(f'\n  Existing CGM scaler: mean={gs[0]:.4f}  std={gs[1]:.4f}')
        print(f'  Recomputed CGM:      mean={cgm_mean:.4f}  std={cgm_std:.4f}')

    scaler = np.array([cgm_mean, cgm_std, pi_mean, pi_std, ra_mean, ra_std],
                      dtype=np.float32)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.save(OUT_PATH, scaler)
    print(f'\nSaved to {OUT_PATH}')


if __name__ == '__main__':
    main()
