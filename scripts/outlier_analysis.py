"""
Per-patient CGM distribution analysis.
Visualises the distribution of per-patient CGM means and stds,
identifies outliers, and computes the global mean+std for preprocessing.
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_patient_stats(data_dir: str):
    files = sorted(Path(data_dir).glob('*.npz'))
    patient_ids, cgm_means, cgm_stds, n_windows = [], [], [], []
    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            patient_ids.append(str(d['patient_id']))
            cgm_means.append(float(d['scaler_mean'][0]))
            cgm_stds.append(float(d['scaler_std'][0]))
            n_windows.append(int(d['windows'].shape[0]))
        except Exception as e:
            print(f'  WARNING: skipping {f.name}: {e}')
    return (np.array(patient_ids), np.array(cgm_means),
            np.array(cgm_stds), np.array(n_windows))


def iqr_bounds(x, k=1.5):
    q1, q3 = np.percentile(x, 25), np.percentile(x, 75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Loading patient stats from {args.data}...')
    patient_ids, cgm_means, cgm_stds, n_windows = load_patient_stats(args.data)
    N = len(cgm_means)
    print(f'  {N} patients loaded')

    # --- summary statistics ---
    print(f'\nCGM mean (mg/dL):')
    print(f'  min={cgm_means.min():.1f}  p5={np.percentile(cgm_means,5):.1f}  '
          f'p25={np.percentile(cgm_means,25):.1f}  median={np.median(cgm_means):.1f}  '
          f'p75={np.percentile(cgm_means,75):.1f}  p95={np.percentile(cgm_means,95):.1f}  '
          f'max={cgm_means.max():.1f}')

    print(f'\nCGM std (mg/dL):')
    print(f'  min={cgm_stds.min():.1f}  p5={np.percentile(cgm_stds,5):.1f}  '
          f'p25={np.percentile(cgm_stds,25):.1f}  median={np.median(cgm_stds):.1f}  '
          f'p75={np.percentile(cgm_stds,75):.1f}  p95={np.percentile(cgm_stds,95):.1f}  '
          f'max={cgm_stds.max():.1f}')

    # --- outlier detection (IQR) ---
    mean_lo, mean_hi = iqr_bounds(cgm_means, k=3.0)
    std_lo,  std_hi  = iqr_bounds(cgm_stds,  k=3.0)

    outlier_mask = (
        (cgm_means < mean_lo) | (cgm_means > mean_hi) |
        (cgm_stds  < std_lo)  | (cgm_stds  > std_hi)
    )
    n_outliers = outlier_mask.sum()
    print(f'\nOutliers (3×IQR): {n_outliers} / {N}')
    if n_outliers > 0:
        print(f'  mean bounds: [{mean_lo:.1f}, {mean_hi:.1f}] mg/dL')
        print(f'  std  bounds: [{std_lo:.1f}, {std_hi:.1f}] mg/dL')
        for pid, m, s, nw in zip(patient_ids[outlier_mask],
                                  cgm_means[outlier_mask],
                                  cgm_stds[outlier_mask],
                                  n_windows[outlier_mask]):
            flag = []
            if m < mean_lo or m > mean_hi: flag.append(f'mean={m:.1f}')
            if s < std_lo  or s > std_hi:  flag.append(f'std={s:.1f}')
            print(f'    patient {pid:>6}  windows={nw:4d}  {", ".join(flag)}')

    # --- global stats (excluding outliers) ---
    clean_mask = ~outlier_mask
    clean_means = cgm_means[clean_mask]
    clean_stds  = cgm_stds[clean_mask]
    clean_nw    = n_windows[clean_mask]

    # Window-weighted global mean and std
    total_windows = clean_nw.sum()
    global_mean = np.sum(clean_means * clean_nw) / total_windows

    # Pooled std: sqrt( mean(var_i + mean_i²) - global_mean² )
    global_var = np.sum((clean_stds**2 + clean_means**2) * clean_nw) / total_windows - global_mean**2
    global_std = np.sqrt(global_var)

    # Simple (unweighted) alternatives for comparison
    simple_mean = clean_means.mean()
    simple_std  = clean_means.std()

    print(f'\nGlobal stats (excluding {n_outliers} outliers, N={clean_mask.sum()} patients):')
    print(f'  Window-weighted mean: {global_mean:.2f} mg/dL')
    print(f'  Window-weighted std:  {global_std:.2f} mg/dL')
    print(f'  Simple mean of patient means: {simple_mean:.2f} mg/dL')
    print(f'  Simple std  of patient means: {simple_std:.2f} mg/dL')

    # Save for preprocessing
    np.save(os.path.join(args.out_dir, 'global_scaler.npy'),
            np.array([global_mean, global_std]))
    print(f'\n  Saved global_scaler.npy  →  mean={global_mean:.2f}, std={global_std:.2f}')

    # Also save outlier patient IDs
    if n_outliers > 0:
        outlier_path = os.path.join(args.out_dir, 'outlier_patients.txt')
        with open(outlier_path, 'w') as fh:
            for pid in patient_ids[outlier_mask]:
                fh.write(pid + '\n')
        print(f'  Saved outlier_patients.txt')

    # --- plots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Per-patient CGM distribution — 1037 T1D adults', fontsize=13)

    # 1. Histogram of CGM means
    ax = axes[0, 0]
    ax.hist(cgm_means, bins=60, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.axvline(global_mean, color='red', lw=1.5, label=f'global mean = {global_mean:.1f}')
    ax.axvline(mean_lo, color='orange', lw=1, ls='--', label=f'3×IQR bounds')
    ax.axvline(mean_hi, color='orange', lw=1, ls='--')
    ax.set_xlabel('Per-patient CGM mean (mg/dL)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of patient means')
    ax.legend(fontsize=8)

    # 2. Histogram of CGM stds
    ax = axes[0, 1]
    ax.hist(cgm_stds, bins=60, color='seagreen', edgecolor='white', linewidth=0.3)
    ax.axvline(global_std, color='red', lw=1.5, label=f'pooled std = {global_std:.1f}')
    ax.axvline(std_lo, color='orange', lw=1, ls='--', label='3×IQR bounds')
    ax.axvline(std_hi, color='orange', lw=1, ls='--')
    ax.set_xlabel('Per-patient CGM std (mg/dL)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of patient stds')
    ax.legend(fontsize=8)

    # 3. Scatter: mean vs std, coloured by outlier status
    ax = axes[1, 0]
    ax.scatter(cgm_means[clean_mask], cgm_stds[clean_mask],
               s=6, alpha=0.5, color='steelblue', label=f'clean ({clean_mask.sum()})')
    if n_outliers > 0:
        ax.scatter(cgm_means[outlier_mask], cgm_stds[outlier_mask],
                   s=30, color='red', marker='x', zorder=5, label=f'outlier ({n_outliers})')
    ax.set_xlabel('Per-patient CGM mean (mg/dL)')
    ax.set_ylabel('Per-patient CGM std (mg/dL)')
    ax.set_title('Mean vs std per patient')
    ax.legend(fontsize=8)

    # 4. Sorted means (rank plot) to spot extreme values clearly
    ax = axes[1, 1]
    sorted_means = np.sort(cgm_means)
    ax.plot(sorted_means, color='steelblue', lw=1)
    ax.axhline(mean_lo, color='orange', lw=1, ls='--', label='3×IQR bounds')
    ax.axhline(mean_hi, color='orange', lw=1, ls='--')
    ax.axhline(global_mean, color='red', lw=1.5, ls='-', label=f'global mean')
    ax.set_xlabel('Patient rank')
    ax.set_ylabel('CGM mean (mg/dL)')
    ax.set_title('Sorted patient means')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, 'cgm_distribution.png')
    plt.savefig(plot_path, dpi=150)
    print(f'  Saved {plot_path}')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    default='data/processed/adults')
    parser.add_argument('--out_dir', default='results/outlier_analysis')
    main(parser.parse_args())
