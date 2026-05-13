import os
import numpy as np
from typing import Optional

STRIDE     = 72    # steps between consecutive windows in .npz (Stage 1: 6h)
WINDOW_LEN = 288

# LOOKAHEAD: window[i+LOOKAHEAD] starts exactly 1 step after window[i] ends
# (only holds when no windows were filtered between i and i+LOOKAHEAD).
LOOKAHEAD  = WINDOW_LEN // STRIDE   # = 4

# 24 horizons: t+5, t+10, ..., t+120 min  (local indices 0..23 in window[i+LOOKAHEAD])
N_HORIZONS     = 24
HORIZON_LABELS = [f'{(i+1)*5}min' for i in range(N_HORIZONS)]

# Feature indices in the window tensor
IDX_CGM, IDX_PI, IDX_RA  = 0, 1, 2
IDX_HSIN, IDX_HCOS        = 3, 4
IDX_BOLUS, IDX_CARBS      = 5, 6

# Contiguity gate: max allowed z-score jump between last step of window[i] and
# first step of window[i+LOOKAHEAD].  Maximum physiological CGM rate ≈ 4 mg/dL/min
# → 20 mg/dL over 5 min; with typical scaler_std ≈ 30–35 mg/dL that is ~0.65 z.
# Threshold of 1.0 accepts all valid 5-min transitions and rejects the vast
# majority of 6 h+ gaps (which produce jumps >> 1 z-score unit).
CONTIGUITY_THRESHOLD = 1.0

HYPO_THRESHOLD         = 70.0   # mg/dL
HYPO_AHEAD             = 24     # timesteps = 2 h at 5-min resolution
HYPO_AHEAD_NOCTURNAL   = 96     # timesteps = 8 h (full night, bedtime_only mode)

# Imputation gap lengths matching MTSM training distribution (steps × 5min = duration)
# MTSM was trained with MASK_MIN_LEN=60 (5h) and MASK_MAX_LEN=96 (8h).
# 48-step (4h) gap tests slight out-of-distribution generalization.
IMPUTATION_GAP_LENGTHS = [48, 60, 72, 96]   # 4h, 5h, 6h, 8h
IMPUTATION_GAP_LABELS  = ['4h', '5h', '6h', '8h']

# Training gap range — same as MTSM Stage 1 masking to ensure raw model
# is trained on the same distribution as the encoder.
IMPUTATION_TRAIN_GAP_MIN = 60   # 5h (steps)
IMPUTATION_TRAIN_GAP_MAX = 96   # 8h (steps)


def load_patient(path: str, no_age: bool = True):
    d = np.load(path)
    windows = d['windows'].astype(np.float32)
    if no_age:
        windows = windows[:, :, :10]
    return windows, float(d['scaler_mean'][0]), float(d['scaler_std'][0])


def load_all_patients(data_dir: str, max_patients: Optional[int] = None,
                      no_age: bool = True):
    """Returns a list of (path, no_age) tuples; data is loaded lazily per patient."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.npz'))
    if max_patients:
        files = files[:max_patients]
    paths = [(os.path.join(data_dir, f), no_age) for f in files]
    print(f'  Found {len(paths)} patient files', flush=True)
    return paths


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
    """
    Yields ({'window': (288,10), 'last_cgm': (1,)}, y_cgm_mg (6,)).

    Contiguity gate: pairs where the CGM z-score jump between the last step of
    window[i] and the first step of window[i+LOOKAHEAD] exceeds
    CONTIGUITY_THRESHOLD are skipped — they arise from filtered windows breaking
    the stride assumption and would corrupt training with mislabelled targets.

    last_cgm is the last observed CGM in mg/dL; the model uses it as a skip-
    connection anchor so predictions are guaranteed to start from that value.
    """
    for path, no_age in patients:
        try:
            windows, scaler_mean, scaler_std = load_patient(path, no_age)
        except Exception as e:
            print(f'  WARNING: skipping corrupt file {os.path.basename(path)}: {e}',
                  flush=True)
            continue
        for i in range(len(windows) - LOOKAHEAD):
            # Contiguity gate
            delta_z = abs(float(windows[i, -1, IDX_CGM])
                          - float(windows[i + LOOKAHEAD, 0, IDX_CGM]))
            if delta_z > CONTIGUITY_THRESHOLD:
                continue

            last_cgm_mg = float(windows[i, -1, IDX_CGM]) * scaler_std + scaler_mean
            labels_z    = windows[i + LOOKAHEAD][0:N_HORIZONS, IDX_CGM]
            cgm_mg      = (labels_z * scaler_std + scaler_mean).astype(np.float32)

            yield (
                {'window':   windows[i],
                 'last_cgm': np.array([last_cgm_mg], dtype=np.float32)},
                cgm_mg,
            )


def make_forecasting_dataset(patients, val_split: float = 0.1, test_split: float = 0.1,
                              batch_size: int = 128,
                              max_train_patients: Optional[int] = None):
    """
    Returns train/val/test tf.data.Dataset objects split by patient.

    Each dataset yields ({'window': (288,10), 'last_cgm': (1,)}, y_cgm_mg (6,)):
      window    : 24 h multimodal input in z-score
      last_cgm  : last observed CGM value in mg/dL (skip-connection anchor)
      y_cgm_mg  : CGM in mg/dL at t+5…t+30 min (absolute, not delta)
    max_train_patients caps only the training split; val/test remain at full size.
    """
    import tensorflow as tf

    train_p, val_p, test_p = _patient_split(patients, val_split, test_split)
    if max_train_patients is not None and max_train_patients < len(train_p):
        train_p = train_p[:max_train_patients]
        print(f'  Data efficiency: capped train to {len(train_p)} patients', flush=True)
    print(f'  Patients — train:{len(train_p)}  val:{len(val_p)}  test:{len(test_p)}',
          flush=True)

    sig = (
        {'window':   tf.TensorSpec((WINDOW_LEN, 10), tf.float32),
         'last_cgm': tf.TensorSpec((1,),             tf.float32)},
        tf.TensorSpec((N_HORIZONS,), tf.float32),
    )

    def make_ds(split_patients, shuffle, repeat=False):
        ds = tf.data.Dataset.from_generator(
            lambda p=split_patients: _forecasting_gen(p), output_signature=sig
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=2_000, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size)
        if repeat:
            ds = ds.repeat()
        return ds.prefetch(2)

    print('  Counting training steps (one pass)...', flush=True)
    steps_per_epoch = sum(1 for _ in make_ds(train_p, shuffle=False))
    print(f'  steps_per_epoch: {steps_per_epoch}', flush=True)

    print('  Counting val steps (one pass)...', flush=True)
    val_steps = sum(1 for _ in make_ds(val_p, shuffle=False))
    print(f'  val_steps: {val_steps}', flush=True)

    return {
        'train':           make_ds(train_p, shuffle=True, repeat=True),
        'val':             make_ds(val_p,   shuffle=False, repeat=True),
        'test':            make_ds(test_p,  shuffle=False),
        'steps_per_epoch': steps_per_epoch,
        'val_steps':       val_steps,
        'val_patients':    val_p,
        'test_patients':   test_p,
        'batch_size':      batch_size,
    }


def make_ar_eval_data(patients) -> dict:
    """
    Collect all test windows with per-patient scalers for AR decoder evaluation.
    Returns numpy arrays — not a tf.data pipeline — so predict_ar_decoder can
    do serial rollout with access to scaler_std for z→mg/dL conversion.
    Applies the same contiguity gate as _forecasting_gen for comparable metrics.
    """
    all_windows, all_last_mg, all_last_z, all_std, all_y_mg = [], [], [], [], []
    for path, no_age in patients:
        try:
            windows, scaler_mean, scaler_std = load_patient(path, no_age)
        except Exception as e:
            print(f'  WARNING: skipping {os.path.basename(path)}: {e}', flush=True)
            continue
        for i in range(len(windows) - LOOKAHEAD):
            delta_z = abs(float(windows[i, -1, IDX_CGM])
                          - float(windows[i + LOOKAHEAD, 0, IDX_CGM]))
            if delta_z > CONTIGUITY_THRESHOLD:
                continue
            last_mg = float(windows[i, -1, IDX_CGM]) * scaler_std + scaler_mean
            labels_z = windows[i + LOOKAHEAD, 0:N_HORIZONS, IDX_CGM]
            cgm_mg   = (labels_z * scaler_std + scaler_mean).astype(np.float32)
            all_windows.append(windows[i])
            all_last_mg.append(last_mg)
            all_last_z.append(float(windows[i, -1, IDX_CGM]))
            all_std.append(scaler_std)
            all_y_mg.append(cgm_mg)
    print(f'  AR eval: {len(all_windows):,} windows loaded', flush=True)
    return {
        'windows':     np.stack(all_windows).astype(np.float32),
        'last_cgm_mg': np.array(all_last_mg,  dtype=np.float32),
        'last_cgm_z':  np.array(all_last_z,   dtype=np.float32),
        'scaler_std':  np.array(all_std,       dtype=np.float32),
        'y_mg':        np.stack(all_y_mg).astype(np.float32),
    }


def make_eval_dataset(patients, batch_size: int = 128):
    """
    Create a fresh, non-repeating eval dataset from a patient list.
    Call this per-variant to avoid TF from_generator state exhaustion
    when the same dataset object is reused across multiple model.fit() calls.
    """
    import tensorflow as tf
    sig = (
        {'window':   tf.TensorSpec((WINDOW_LEN, 10), tf.float32),
         'last_cgm': tf.TensorSpec((1,),             tf.float32)},
        tf.TensorSpec((N_HORIZONS,), tf.float32),
    )
    ds = tf.data.Dataset.from_generator(
        lambda p=patients: _forecasting_gen(p), output_signature=sig
    )
    return ds.batch(batch_size).prefetch(2)


def _hypo_gen(patients, bedtime_only: bool = False):
    """
    Yields (window (288,10), label (2,)) for nocturnal hypo-risk (Weibull survival).

    bedtime_only=False (default):
      Nocturnal filter — horizon starts between 22:00 and 06:00; HYPO_AHEAD=24 (2h).
    bedtime_only=True:
      Bedtime filter — horizon starts between 20:00 and 23:59; HYPO_AHEAD_NOCTURNAL=96
      (8h). Covers the full nocturnal period from bedtime to morning.

    Label = [time_to_event, delta] where:
      time_to_event: first step (1-indexed) where CGM < 70 mg/dL, or ahead if none
      delta:         1.0 if hypo observed, 0.0 if censored
    """
    ahead = HYPO_AHEAD_NOCTURNAL if bedtime_only else HYPO_AHEAD
    for path, no_age in patients:
        try:
            windows, mean, std = load_patient(path, no_age)
        except Exception as e:
            print(f'  WARNING: skipping {os.path.basename(path)}: {e}', flush=True)
            continue
        for i in range(len(windows) - LOOKAHEAD):
            delta_z = abs(float(windows[i, -1, IDX_CGM])
                          - float(windows[i + LOOKAHEAD, 0, IDX_CGM]))
            if delta_z > CONTIGUITY_THRESHOLD:
                continue
            hsin = float(windows[i + LOOKAHEAD, 0, IDX_HSIN])
            hcos = float(windows[i + LOOKAHEAD, 0, IDX_HCOS])
            hour = np.arctan2(hsin, hcos) / (2 * np.pi) * 24 % 24
            if bedtime_only:
                if not (20 <= hour < 24):   # evening 20:00–23:59 → predict full night
                    continue
            else:
                if not (hour >= 22 or hour < 6):   # original nocturnal filter
                    continue
            future_mg  = windows[i + LOOKAHEAD, :ahead, IDX_CGM] * std + mean
            hypo_steps = np.where(future_mg < HYPO_THRESHOLD)[0]
            if hypo_steps.size > 0:
                t_event = float(hypo_steps[0] + 1)   # 1-indexed steps
                event   = 1.0
            else:
                t_event = float(ahead)                # censored at horizon end
                event   = 0.0
            yield windows[i], np.array([t_event, event], dtype=np.float32)


def make_hypo_dataset(patients, val_split: float = 0.1, test_split: float = 0.1,
                       batch_size: int = 128,
                       max_train_patients: Optional[int] = None,
                       bedtime_only: bool = False):
    """
    Returns train/val/test tf.data.Dataset objects for nocturnal hypo-risk (Weibull).
    Label is [time_to_event, delta] — no pos_weight needed for Weibull NLL.
    max_train_patients caps only the training split; val/test remain at full size.
    """
    import tensorflow as tf

    train_p, val_p, test_p = _patient_split(patients, val_split, test_split)
    if max_train_patients is not None and max_train_patients < len(train_p):
        train_p = train_p[:max_train_patients]
        print(f'  Data efficiency: capped train to {len(train_p)} patients', flush=True)
    print(f'  Patients — train:{len(train_p)}  val:{len(val_p)}  test:{len(test_p)}',
          flush=True)

    sig = (
        tf.TensorSpec((WINDOW_LEN, 10), tf.float32),
        tf.TensorSpec((2,),             tf.float32),
    )

    def make_ds(split_patients, shuffle, repeat=False):
        ds = tf.data.Dataset.from_generator(
            lambda p=split_patients: _hypo_gen(p, bedtime_only=bedtime_only),
            output_signature=sig
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=2_000, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size)
        if repeat:
            ds = ds.repeat()
        return ds.prefetch(2)

    print('  Scanning labels (all splits)...', flush=True)
    label_batches = []
    for _, y in make_ds(train_p, shuffle=False):
        label_batches.append(y.numpy())
    steps_per_epoch = len(label_batches)
    train_labels = np.concatenate(label_batches)

    val_labels  = np.concatenate([y.numpy() for _, y in make_ds(val_p,  shuffle=False)])
    test_labels = np.concatenate([y.numpy() for _, y in make_ds(test_p, shuffle=False)])
    all_labels  = np.concatenate([train_labels, val_labels, test_labels])

    def _split_summary(labels, name):
        n = len(labels); pos = int(labels[:, 1].sum())
        print(f'    {name:<6} windows={n:>6}  hypos={pos:>4}  rate={pos/max(n,1):.3f}',
              flush=True)
        return n, pos

    n_tr, p_tr = _split_summary(train_labels, 'train')
    n_va, p_va = _split_summary(val_labels,   'val')
    n_te, p_te = _split_summary(test_labels,  'test')
    n_tot = n_tr + n_va + n_te; p_tot = p_tr + p_va + p_te
    print(f'    {"TOTAL":<6} windows={n_tot:>6}  hypos={p_tot:>4}  rate={p_tot/max(n_tot,1):.3f}',
          flush=True)

    pos_rate = float(train_labels[:, 1].mean())   # delta column (train only, for loss)
    print(f'  steps_per_epoch: {steps_per_epoch}', flush=True)

    return {
        'train':           make_ds(train_p, shuffle=True, repeat=True),
        'val':             make_ds(val_p,   shuffle=False),
        'test':            make_ds(test_p,  shuffle=False),
        'steps_per_epoch': steps_per_epoch,
        'val_patients':    val_p,
        'test_patients':   test_p,
        'batch_size':      batch_size,
        'pos_rate':        pos_rate,
    }


def make_eval_hypo_dataset(patients, batch_size: int = 128, bedtime_only: bool = False):
    """Fresh non-repeating hypo eval dataset — call per variant to avoid exhaustion."""
    import tensorflow as tf
    sig = (
        tf.TensorSpec((WINDOW_LEN, 10), tf.float32),
        tf.TensorSpec((2,),             tf.float32),
    )
    ds = tf.data.Dataset.from_generator(
        lambda p=patients: _hypo_gen(p, bedtime_only=bedtime_only),
        output_signature=sig
    )
    return ds.batch(batch_size).prefetch(2)


def naive_forecast(test_patients) -> tuple:
    """
    Last-value carry-forward baseline.
    Returns (y_pred, y_true) as (N, N_HORIZONS) mg/dL arrays.
    Applies the same contiguity gate as the training generator so that metrics
    are computed on the same sample distribution.
    """
    preds, trues = [], []
    for path, no_age in test_patients:
        try:
            windows, scaler_mean, scaler_std = load_patient(path, no_age)
        except Exception:
            continue
        for i in range(len(windows) - LOOKAHEAD):
            delta_z = abs(float(windows[i, -1, IDX_CGM])
                          - float(windows[i + LOOKAHEAD, 0, IDX_CGM]))
            if delta_z > CONTIGUITY_THRESHOLD:
                continue
            last_cgm_mg = float(windows[i, -1, IDX_CGM]) * scaler_std + scaler_mean
            labels_z    = windows[i + LOOKAHEAD, 0:N_HORIZONS, IDX_CGM]
            cgm_mg      = labels_z * scaler_std + scaler_mean
            preds.append(np.full(N_HORIZONS, last_cgm_mg, dtype=np.float32))
            trues.append(cgm_mg.astype(np.float32))
    return np.array(preds), np.array(trues)


# ── App 1 — Imputation ────────────────────────────────────────────────────────

def _imputation_gen(patients, gap_min=IMPUTATION_TRAIN_GAP_MIN,
                     gap_max=IMPUTATION_TRAIN_GAP_MAX, seed=None):
    """
    Yields (x_masked (288,10), y (288,2)) for imputation model training.
    y[:, 0] = true CGM z-score; y[:, 1] = mask (1 = masked, 0 = observed).
    No contiguity gate — every window is used; gap position/length is random.
    """
    rng = np.random.default_rng(seed)
    for path, no_age in patients:
        try:
            windows, _, _ = load_patient(path, no_age)
        except Exception as e:
            print(f'  WARNING: skipping {os.path.basename(path)}: {e}', flush=True)
            continue
        for i in range(len(windows)):
            w        = windows[i].copy()
            gap_len  = int(rng.integers(gap_min, gap_max + 1))
            gap_s    = int(rng.integers(0, WINDOW_LEN - gap_len))
            mask     = np.zeros(WINDOW_LEN, dtype=np.float32)
            mask[gap_s:gap_s + gap_len] = 1.0
            target   = w[:, IDX_CGM].copy()
            w[gap_s:gap_s + gap_len, IDX_CGM] = 0.0   # mask token = 0
            yield w, np.stack([target, mask], axis=-1)  # (288,2)


def make_imputation_dataset(patients, val_split=0.1, test_split=0.1,
                             batch_size=128,
                             gap_min=IMPUTATION_TRAIN_GAP_MIN,
                             gap_max=IMPUTATION_TRAIN_GAP_MAX):
    """
    Train/val/test split for raw imputation model training.
    Train: random gaps each pass (augmentation); Val/test: fixed seed for reproducibility.
    Returns dict with train tf.data.Dataset (repeating), val/test patient lists,
    steps_per_epoch, and gap config.
    """
    import tensorflow as tf
    train_p, val_p, test_p = _patient_split(patients, val_split, test_split)
    print(f'  Patients — train:{len(train_p)}  val:{len(val_p)}  test:{len(test_p)}',
          flush=True)

    sig = (
        tf.TensorSpec((WINDOW_LEN, 10), tf.float32),
        tf.TensorSpec((WINDOW_LEN, 2),  tf.float32),
    )

    def _make_ds(split_patients, shuffle, repeat=False, seed=None):
        ds = tf.data.Dataset.from_generator(
            lambda p=split_patients, s=seed: _imputation_gen(p, gap_min, gap_max, seed=s),
            output_signature=sig,
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=2_000, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size)
        if repeat:
            ds = ds.repeat()
        return ds.prefetch(2)

    print('  Counting training steps (one pass)...', flush=True)
    steps_per_epoch = sum(1 for _ in _make_ds(train_p, shuffle=False, seed=0))
    print(f'  steps_per_epoch: {steps_per_epoch}', flush=True)

    return {
        'train':           _make_ds(train_p, shuffle=True, repeat=True, seed=None),
        'val_patients':    val_p,
        'test_patients':   test_p,
        'steps_per_epoch': steps_per_epoch,
        'batch_size':      batch_size,
        'gap_min':         gap_min,
        'gap_max':         gap_max,
    }


def make_eval_imputation_tf(patients, batch_size=128,
                             gap_min=IMPUTATION_TRAIN_GAP_MIN,
                             gap_max=IMPUTATION_TRAIN_GAP_MAX, seed=42):
    """
    Fresh non-repeating imputation tf.data dataset for per-epoch validation.
    Call this each epoch to avoid TF generator exhaustion.
    """
    import tensorflow as tf
    sig = (
        tf.TensorSpec((WINDOW_LEN, 10), tf.float32),
        tf.TensorSpec((WINDOW_LEN, 2),  tf.float32),
    )
    ds = tf.data.Dataset.from_generator(
        lambda p=patients, s=seed: _imputation_gen(p, gap_min, gap_max, seed=s),
        output_signature=sig,
    )
    return ds.batch(batch_size).prefetch(2)


def make_eval_imputation_numpy(patients, gap_len, max_windows=1000, seed=42):
    """
    Fixed centered-gap evaluation dataset returned as numpy arrays.
    All methods receive the SAME masked inputs for a fair comparison.

    gap_len: gap in steps (e.g. 6=30min, 12=1h, 24=2h, 48=4h).
    Gap is centered: starts at (WINDOW_LEN - gap_len) // 2.

    Returns dict with:
      windows_orig   (N, 288, 10) — original unmasked
      windows_masked (N, 288, 10) — CGM zeroed in gap
      masks          (N, 288)     — binary (1=gap)
      scaler_means/stds (N,)      — per-window inverse-transform params
      gap_start, gap_end, gap_len
    """
    rng       = np.random.default_rng(seed)
    gap_start = (WINDOW_LEN - gap_len) // 2
    gap_end   = gap_start + gap_len

    all_windows, all_means, all_stds = [], [], []
    for path, no_age in patients:
        try:
            windows, mean, std = load_patient(path, no_age)
        except Exception:
            continue
        for w in windows:
            all_windows.append(w)
            all_means.append(mean)
            all_stds.append(std)

    if len(all_windows) > max_windows:
        idx         = rng.choice(len(all_windows), max_windows, replace=False)
        all_windows = [all_windows[i] for i in idx]
        all_means   = [all_means[i]   for i in idx]
        all_stds    = [all_stds[i]    for i in idx]

    windows_orig          = np.stack(all_windows).astype(np.float32)
    windows_masked        = windows_orig.copy()
    windows_masked[:, gap_start:gap_end, IDX_CGM] = 0.0
    masks                 = np.zeros((len(windows_orig), WINDOW_LEN), dtype=np.float32)
    masks[:, gap_start:gap_end] = 1.0

    return {
        'windows_orig':   windows_orig,
        'windows_masked': windows_masked,
        'masks':          masks,
        'scaler_means':   np.array(all_means, dtype=np.float32),
        'scaler_stds':    np.array(all_stds,  dtype=np.float32),
        'gap_start':      gap_start,
        'gap_end':        gap_end,
        'gap_len':        gap_len,
    }
