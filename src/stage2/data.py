import os
import numpy as np
from typing import Optional

STRIDE     = 72    # steps between consecutive windows in .npz (Stage 1: 6h)
WINDOW_LEN = 288

# LOOKAHEAD: window[i+LOOKAHEAD] starts exactly 1 step after window[i] ends
LOOKAHEAD  = WINDOW_LEN // STRIDE   # = 4

# 6 horizons: t+5, t+10, ..., t+30 min  (local indices 0..5 in window[i+LOOKAHEAD])
N_HORIZONS     = 6
HORIZON_LABELS = ['5min', '10min', '15min', '20min', '25min', '30min']

# Feature indices in the window tensor
IDX_CGM, IDX_PI, IDX_RA = 0, 1, 2
IDX_BOLUS, IDX_CARBS     = 5, 6


def load_patient(path: str, no_age: bool = True):
    d = np.load(path)
    windows = d['windows'].astype(np.float32)
    if no_age:
        windows = windows[:, :, :10]
    return windows, float(d['scaler_mean'][0]), float(d['scaler_std'][0])


def load_all_patients(data_dir: str, max_patients: Optional[int] = None,
                      no_age: bool = True):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.npz'))
    if max_patients:
        files = files[:max_patients]
    patients = []
    for i, f in enumerate(files):
        patients.append(load_patient(os.path.join(data_dir, f), no_age))
        if (i + 1) % 100 == 0 or (i + 1) == len(files):
            print(f'  Loaded {i + 1}/{len(files)} patients', flush=True)
    return patients


def _patient_split(patients, val_split, test_split, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(patients))
    n_test = max(1, int(len(patients) * test_split))
    n_val  = max(1, int(len(patients) * val_split))
    return (
        [patients[i] for i in idx[n_test + n_val:]],
        [patients[i] for i in idx[n_test:n_test + n_val]],
        [patients[i] for i in idx[:n_test]],
    )


def _forecasting_gen(patients):
    """Yields (window, y_cgm_mg) — no future drivers."""
    for windows, scaler_mean, scaler_std in patients:
        for i in range(len(windows) - LOOKAHEAD):
            # t+5..t+30 min: first 6 steps of the next contiguous window
            labels_z  = windows[i + LOOKAHEAD][0:N_HORIZONS, IDX_CGM]
            cgm_mg    = (labels_z * scaler_std + scaler_mean).astype(np.float32)
            yield windows[i], cgm_mg


def make_forecasting_dataset(patients, val_split: float = 0.1, test_split: float = 0.1,
                              batch_size: int = 128):
    """
    Returns train/val/test tf.data.Dataset objects split by patient.

    Each dataset yields (window, y_cgm_mg):
      window   : (288, 10)  — 24h multimodal input
      y_cgm_mg : (6,)       — CGM in mg/dL at t+5…t+30 min
    """
    import tensorflow as tf

    train_p, val_p, test_p = _patient_split(patients, val_split, test_split)
    print(f'  Patients — train:{len(train_p)}  val:{len(val_p)}  test:{len(test_p)}',
          flush=True)

    sig = (
        tf.TensorSpec((WINDOW_LEN, 10), tf.float32),
        tf.TensorSpec((N_HORIZONS,),    tf.float32),
    )

    def make_ds(split_patients, shuffle):
        ds = tf.data.Dataset.from_generator(
            lambda p=split_patients: _forecasting_gen(p), output_signature=sig
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=10_000, reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return {
        'train': make_ds(train_p, shuffle=True),
        'val':   make_ds(val_p,   shuffle=False),
        'test':  make_ds(test_p,  shuffle=False),
    }
