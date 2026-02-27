"""
preprocessing.py
================
Full preprocessing pipeline for the TVAE project.

Input:  data/interim/combined_filtered.parquet
Output: data/processed/<patient_id>.npz

Each .npz contains:
    - windows:      (N_windows, 288, 10) float32
    - scaler_mean:  (3,) float32  — mean  for [CGM, PI, RA]
    - scaler_std:   (3,) float32  — std   for [CGM, PI, RA]
    - patient_id:   str
    - modality:     str

Feature order in window tensor (axis=2):
    0  CGM           normalised
    1  PI            normalised
    2  RA            normalised
    3  hour_sin      in [-1, 1]
    4  hour_cos      in [-1, 1]
    5  bolus_logged  binary
    6  carbs_logged  binary
    7  AID           one-hot
    8  SAP           one-hot
    9  MDI           one-hot

Usage:
    python -m src.preprocessing
"""

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from pathlib import Path
from scipy.integrate import odeint
from tqdm import tqdm

from src.settings import Settings


# ── 1. Quality filter ─────────────────────────────────────────────────────────

def get_valid_ids(parquet_path: Path, cfg: Settings) -> set[str]:
    """
    Stream through the parquet and compute per-patient quality metrics.
    Returns set of patient IDs that pass all quality filters.
    """
    p = cfg.preprocessing
    stats = {}  # pid -> {'cgm_count': int, 'cgm_null': int, 'cgm_std': float, 'has_carbs': bool}

    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=500_000, columns=['id', 'CGM', 'carbs']):
        df = batch.to_pandas()
        for pid, grp in df.groupby('id'):
            cgm = grp['CGM']
            carbs = grp['carbs']
            if pid not in stats:
                stats[pid] = {
                    'cgm_count': 0,
                    'cgm_null':  0,
                    'cgm_sum':   0.0,
                    'cgm_sum2':  0.0,
                    'has_carbs': False,
                }
            s = stats[pid]
            valid = cgm.dropna()
            s['cgm_count'] += len(cgm)
            s['cgm_null']  += cgm.isna().sum()
            s['cgm_sum']   += valid.sum()
            s['cgm_sum2']  += (valid ** 2).sum()
            s['has_carbs'] = s['has_carbs'] or (carbs > 0).any()

    valid_ids = set()
    for pid, s in stats.items():
        n = s['cgm_count'] - s['cgm_null']
        if n < 2:
            continue
        missing_pct = 100 * s['cgm_null'] / s['cgm_count']
        mean = s['cgm_sum'] / n
        std  = np.sqrt(s['cgm_sum2'] / n - mean ** 2)
        if (std >= p.cgm_std_min and
                missing_pct <= p.cgm_missing_max and
                s['has_carbs']):
            valid_ids.add(str(pid))

    print(f"Quality filter: {len(valid_ids)}/{len(stats)} patients passed")
    return valid_ids


# ── 2. CGM cleaning (Polars) ──────────────────────────────────────────────────

def clean_cgm(df: pl.DataFrame, cfg: Settings) -> pl.DataFrame:
    """
    Clip physiological limits and interpolate small gaps.
    Uses the research group's approach adapted to our settings.
    """
    p = cfg.preprocessing

    # Clip physiological limits
    glucose = pl.col('CGM')
    glucose = pl.when(glucose < p.cgm_lower).then(None).otherwise(glucose)
    glucose = pl.when(glucose > p.cgm_upper).then(None).otherwise(glucose)
    df = df.with_columns(glucose.alias('CGM'))

    # Detect null blocks and interpolate small gaps
    df = (
        df
        .with_columns(pl.col('CGM').is_null().alias('_is_null'))
        .with_columns(
            (
                (pl.col('_is_null') != pl.col('_is_null').shift())
                .fill_null(True)
                .cast(pl.Int8)
                .cum_sum()
                .over('id')
            ).alias('_block_id')
        )
        .with_columns(
            pl.when(pl.col('_is_null'))
            .then(pl.len().over(['id', '_block_id']))
            .otherwise(None)
            .alias('_null_block_size')
        )
        .with_columns(pl.col('CGM').interpolate().over('id').alias('_cgm_interp'))
        .with_columns(
            pl.when(
                pl.col('_is_null') & (pl.col('_null_block_size') <= p.max_gap_points)
            )
            .then(pl.col('_cgm_interp'))
            .otherwise(pl.col('CGM'))
            .alias('CGM')
        )
        .drop(['_is_null', '_block_id', '_null_block_size', '_cgm_interp'])
    )

    return df


# ── 3. Driver features ────────────────────────────────────────────────────────

def add_driver_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Fill NaN drivers with 0 and add binary logged flags.
    """
    df = df.with_columns([
        pl.col('bolus').fill_null(0.0),
        pl.col('basal').fill_null(0.0),
        pl.col('carbs').fill_null(0.0),
    ])
    df = df.with_columns([
        (pl.col('bolus') > 0).cast(pl.Float32).alias('bolus_logged'),
        (pl.col('carbs') > 0).cast(pl.Float32).alias('carbs_logged'),
    ])
    return df


# ── 4. Hovorka model ──────────────────────────────────────────────────────────

def _pi_system(y, t, params, u_values):
    """3-compartment ODE for plasma insulin."""
    s1, s2, ifa = y
    kdia = 1 / params.tmaxI
    u_t  = u_values[int(t / params.dt)] if int(t / params.dt) < len(u_values) else 0.0
    ds1  = u_t - kdia * s1
    ds2  = kdia * (s1 - s2)
    difa = (s2 / (params.tmaxI * params.VI)) - params.Ke * ifa
    return [ds1, ds2, difa]


def _ra_system(y, t, params, u_values):
    """2-compartment ODE for rate of carb absorption."""
    d1, d2 = y
    u_t = u_values[int(t / params.dt)] if int(t / params.dt) < len(u_values) else 0.0
    dd1 = params.A_G * u_t - (1 / params.tau_D) * d1
    dd2 = (1 / params.tau_D) * d1 - (1 / params.tau_D) * d2
    return [dd1, dd2]


def compute_hovorka(df: pl.DataFrame, cfg: Settings) -> pl.DataFrame:
    """
    Compute PI (Plasma Insulin) and RA (Rate of Absorption) via Hovorka ODEs.
    Operates on a single-patient DataFrame.
    """
    p_pi = cfg.preprocessing.params_pi
    p_ra = cfg.preprocessing.params_ra
    n    = len(df)
    t    = np.arange(0, n * p_pi.dt, p_pi.dt)

    bolus = df['bolus'].to_numpy()
    basal = df['basal'].to_numpy()
    carbs = df['carbs'].to_numpy()

    # PI — bolus component
    sol_bolus = odeint(_pi_system, [0, 0, 0], t,
                       args=(p_pi, bolus / p_pi.dt), hmax=p_pi.dt)
    # PI — basal component
    sol_basal = odeint(_pi_system, [0, 0, 0], t,
                       args=(p_pi, basal / p_pi.dt), hmax=p_pi.dt)
    # PI = bolus_Ifa + basal_Ifa (sign convention from params)
    pi = p_pi.pi_sign * (sol_bolus[:, 2] + sol_basal[:, 2])

    # RA
    sol_ra = odeint(_ra_system, [0, 0], t,
                    args=(p_ra, carbs), hmax=p_ra.dt)
    ra = sol_ra[:, 1] / p_ra.tau_D

    return df.with_columns([
        pl.Series('PI', pi.astype(np.float32)),
        pl.Series('RA', ra.astype(np.float32)),
    ])


# ── 5 & 6. Temporal features + modality one-hot ───────────────────────────────

def add_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add hour_sin and hour_cos from the date column."""
    hour = pl.col('date').dt.hour() + pl.col('date').dt.minute() / 60.0
    df = df.with_columns([
        (2 * np.pi * hour / 24).sin().cast(pl.Float32).alias('hour_sin'),
        (2 * np.pi * hour / 24).cos().cast(pl.Float32).alias('hour_cos'),
    ])
    return df


def add_modality_onehot(df: pl.DataFrame) -> pl.DataFrame:
    """One-hot encode insulin_delivery_modality → AID, SAP, MDI columns."""
    modality = pl.col('insulin_delivery_modality')
    df = df.with_columns([
        (modality == 'AID').cast(pl.Float32).alias('AID'),
        (modality == 'SAP').cast(pl.Float32).alias('SAP'),
        (modality == 'MDI').cast(pl.Float32).alias('MDI'),
    ])
    return df


# ── 7. Normalisation ──────────────────────────────────────────────────────────

def normalise_patient(df: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """
    Z-score normalise CGM, PI, RA per patient.
    Returns normalised df and scaler parameters (mean, std) for [CGM, PI, RA].
    """
    cols = ['CGM', 'PI', 'RA']
    means = np.array([df[c].mean() for c in cols], dtype=np.float32)
    stds  = np.array([df[c].std()  for c in cols], dtype=np.float32)
    stds  = np.where(stds == 0, 1.0, stds)  # avoid division by zero

    df = df.with_columns([
        ((pl.col(c) - float(means[i])) / float(stds[i])).alias(c)
        for i, c in enumerate(cols)
    ])
    return df, means, stds


# ── 8. Sliding windows ────────────────────────────────────────────────────────

FEATURE_COLS = ['CGM', 'PI', 'RA', 'hour_sin', 'hour_cos',
                'bolus_logged', 'carbs_logged', 'AID', 'SAP', 'MDI']


def extract_windows(df: pl.DataFrame, cfg: Settings) -> np.ndarray:
    """
    Slide a window over the patient time series and return valid windows.
    Shape: (N_valid_windows, window_size, n_features)
    """
    p   = cfg.preprocessing
    arr = df.select(FEATURE_COLS).to_numpy().astype(np.float32)  # (T, 10)
    T   = len(arr)
    ws  = p.window_size
    st  = p.stride

    windows = []
    for start in range(0, T - ws + 1, st):
        window = arr[start:start + ws]  # (288, 10)
        # Check missing CGM (index 0) — discard if > threshold
        cgm_null_frac = np.isnan(window[:, 0]).mean()
        if cgm_null_frac > p.max_missing_window:
            continue
        # Replace remaining nulls with 0 (post-interpolation residuals)
        window = np.nan_to_num(window, nan=0.0)
        windows.append(window)

    if len(windows) == 0:
        return np.empty((0, ws, len(FEATURE_COLS)), dtype=np.float32)

    return np.stack(windows, axis=0)  # (N, 288, 10)


# ── 9. Serialisation ──────────────────────────────────────────────────────────

def save_patient(
    patient_id: str,
    windows: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    modality: str,
    output_dir: Path,
) -> None:
    """Save processed patient data as .npz"""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f'{patient_id}.npz',
        windows=windows,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        patient_id=np.array([patient_id]),
        modality=np.array([modality]),
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_preprocessing(cfg: Settings | None = None) -> None:
    """
    Full preprocessing pipeline. Streams through the parquet patient by patient.
    """
    if cfg is None:
        cfg = Settings()

    input_path  = cfg.paths.combined_parquet
    output_dir  = cfg.paths.data_processed

    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")

    # Step 1 — quality filter
    print("\n[1/4] Computing quality filter...")
    valid_ids = get_valid_ids(input_path, cfg)

    # Steps 2-9 — process each patient
    print("\n[2/4] Processing patients...")
    pf = pq.ParquetFile(input_path)

    # Buffer to accumulate rows per patient across batches
    patient_buffer: dict[str, list] = {}

    def process_patient(pid: str, rows: list) -> dict:
        """Run full pipeline for one patient. Returns stats."""
        df_pd = pl.from_pandas(
            __import__('pandas').concat(rows).reset_index(drop=True)
        )
        df_pd = df_pd.sort('date')

        # 2. Clean CGM
        df_pd = clean_cgm(df_pd, cfg)

        # 3. Driver features
        df_pd = add_driver_features(df_pd)

        # 4. Hovorka
        df_pd = compute_hovorka(df_pd, cfg)

        # 5 & 6. Temporal + modality
        df_pd = add_temporal_features(df_pd)
        df_pd = add_modality_onehot(df_pd)

        # 7. Normalise
        df_pd, means, stds = normalise_patient(df_pd)

        # 8. Windows
        windows = extract_windows(df_pd, cfg)

        # Get modality string
        modality = df_pd['insulin_delivery_modality'][0] if 'insulin_delivery_modality' in df_pd.columns else 'unknown'
        if modality is None:
            modality = 'unknown'

        # 9. Save
        if len(windows) > 0:
            save_patient(pid, windows, means, stds, str(modality), output_dir)

        return {'pid': pid, 'n_windows': len(windows)}

    stats_list = []
    cols_needed = ['id', 'date', 'CGM', 'bolus', 'basal', 'carbs',
                   'insulin_delivery_modality']

    for batch in tqdm(pf.iter_batches(batch_size=500_000, columns=cols_needed),
                      desc='Streaming batches'):
        df_batch = batch.to_pandas()
        df_batch = df_batch[df_batch['id'].astype(str).isin(valid_ids)]

        for pid, grp in df_batch.groupby('id'):
            pid = str(pid)
            if pid not in patient_buffer:
                patient_buffer[pid] = []
            patient_buffer[pid].append(grp)

    print(f"\n[3/4] Running pipeline for {len(patient_buffer)} patients...")
    for pid, rows in tqdm(patient_buffer.items(), desc='Processing patients'):
        try:
            s = process_patient(pid, rows)
            stats_list.append(s)
        except Exception as e:
            print(f"  WARNING: Failed {pid} — {e}")

    # Summary
    total_windows = sum(s['n_windows'] for s in stats_list)
    patients_with_data = sum(1 for s in stats_list if s['n_windows'] > 0)
    print(f"\n[4/4] Done.")
    print(f"  Patients processed:    {patients_with_data}/{len(patient_buffer)}")
    print(f"  Total windows:         {total_windows:,}")
    print(f"  Output directory:      {output_dir}")


if __name__ == '__main__':
    run_preprocessing()