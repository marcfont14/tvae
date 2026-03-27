
# Preprocessing Pipeline

Input: `data/interim/combined_filtered.parquet` (1575 patients, ~126M rows, 5-min resolution)
Output: `data/processed/all/` or `data/processed/adults/` — one `.npz` per patient with windowed tensors

Two processed datasets are maintained:
- `data/processed/all/` — 1441 patients, all ages (1–80), 1.46M windows
- `data/processed/adults/` — 988 patients, age ≥ 18, 951K windows

---

## Steps

### 1. Quality Filter

Exclude patients where:

- CGM std < 15 mg/dL (flat signal, sensor malfunction)
- CGM missing > 50% (too sparse)
- 0 carb events logged

Optional age filter via CLI:
- `--min-age` excludes patients below threshold
- `--max-age` excludes patients above threshold

Expected output: ~1441 patients (all), ~988 patients (adults).

### 2. CGM Cleaning

- Clip values outside physiological range [39, 400] mg/dL → null
- Interpolate linearly gaps ≤ 12 consecutive steps (1h)
- Gaps > 12 steps remain null (handled at windowing stage)

### 3. Driver Features

- Fill NaN → 0 for bolus, basal, carbs
- Add binary flags:
    - `bolus_logged` = 1 if bolus > 0 at that step
    - `carbs_logged` = 1 if carbs > 0 at that step

### 4. Hovorka Model

Compute continuous physiological signals from discrete events:

- **PI (Plasma Insulin)** — 3-compartment ODE on bolus + basal: S1 → S2 → Ifa. Parameters: VI=0.12, Ke=0.138, tmaxI=55, dt=5. Represents active insulin in plasma at each timestep.

- **RA (Rate of Absorption)** — 2-compartment ODE on carbs: D1 → D2 → RA. Parameters: tau_D=40, A_G=0.8, dt=5. Represents gut glucose absorption at each timestep.

Both computed per patient over the full time series before windowing.

**Integration method: Euler (forward).** `scipy.integrate.odeint` was evaluated but rejected due to numerical instability on long time series (>50k timesteps) and extreme bolus values, producing NaN outputs. Euler integration with dt=5min is sufficiently accurate for these ODE parameters and produces no NaN. Validated against odeint on 10 patients — MAE of order 1e-2, visually indistinguishable signals.

### 5. Temporal Features

- `hour_sin` = sin(2π × hour / 24) — cyclic encoding of time of day
- `hour_cos` = cos(2π × hour / 24)

### 6. Modality One-Hot

- `AID` = 1/0
- `SAP` = 1/0
- `MDI` = 1/0

One-hot of `insulin_delivery_modality` column. Static per patient, repeated for every timestep.

### 6b. Age Normalisation

- `age_norm` = age / 100

Continuous scalar, static per patient, repeated for every timestep. Not z-scored — dividing by 100 maps the observed range (1–80 years) to approximately [0, 0.8], preserving interpretability. Patients with missing age are assigned 0.0.

**Late fusion decision (run14+):** age_norm is stored in the .npz as feature 10 but is **dropped from the encoder input at training time** via `--no_age`. This slices the feature tensor to 10 features before the model. The motivation: passing age to the Transformer causes the encoder to use demographic shortcuts (more diagonal attention) rather than learning physiological dynamics. Age will be passed directly to Stage 2 task heads as a conditioning variable. No preprocessing changes required — the flag is applied in `experiment_mtsm.py` and `analyse_H.py`.

### 7. Normalisation

Z-score per patient (individual, not group):

- Compute mean and std from the patient's full time series
- Apply to CGM, PI, RA
- Binary features (bolus_logged, carbs_logged), one-hot (AID/SAP/MDI), hour_sin, hour_cos, and age_norm are not normalised

Save scaler parameters (mean, std) per patient for inverse transform at inference.

### 8. Sliding Windows

- Window size: 288 steps (24h)
- Stride: 72 steps (6h)
- Discard windows with > 20% null CGM after interpolation
- Never mix patients within a window

Output tensor shape per window: `(288, 11)`
Features in order: `[CGM, PI, RA, hour_sin, hour_cos, bolus_logged, carbs_logged, AID, SAP, MDI, age_norm]`

### 8b. Pathological Window Filtering (at load time)

An additional filter is applied in `load_windows()` inside `experiment_mtsm.py` **after** loading the .npz, before training. This is separate from the patient-level quality filter in step 1.

```python
has_driver = ((bolus + carbs) > 0).any(axis=1)   # window has ≥1 event
cgm_std    = cgm.std(axis=1)
cgm_ok     = (cgm_std > 0.3) & (cgm_std < 4.0)   # not flat, not corrupted
mask_keep  = has_driver & cgm_ok
```

Removes ~13.8% of windows across the adults dataset. Flat windows (no events, no glucose variability) provide no training signal; extreme-std windows are likely sensor artefacts.

### 9. Serialisation

Save per patient as `.npz`:
```
data/processed/
├── all/
│   ├── 1.npz
│   ├── 2.npz
│   ├── T_1.npz
│   └── ...
└── adults/
    ├── 1.npz
    ├── 2.npz
    └── ...
```

Each `.npz` contains:

- `windows`: array of shape `(N_windows, 288, 11)`
- `scaler_mean`: array of shape `(3,)` — mean for CGM, PI, RA
- `scaler_std`: array of shape `(3,)` — std for CGM, PI, RA
- `patient_id`: str
- `modality`: str
- `age`: float32 — raw age in years

---

## Feature Tensor Summary

| Index | Feature      | Normalised | Notes                          |
|-------|-------------|------------|-------------------------------|
| 0     | CGM         | yes        | z-score per patient            |
| 1     | PI          | yes        | Hovorka plasma insulin         |
| 2     | RA          | yes        | Hovorka rate of absorption     |
| 3     | hour_sin    | no         | already in [-1, 1]             |
| 4     | hour_cos    | no         | already in [-1, 1]             |
| 5     | bolus_logged| no         | binary 0/1                     |
| 6     | carbs_logged| no         | binary 0/1                     |
| 7     | AID         | no         | one-hot                        |
| 8     | SAP         | no         | one-hot                        |
| 9     | MDI         | no         | one-hot                        |
| 10    | age_norm    | no         | age/100, static per patient    |

---

## Estimated Output Size

- all: 1441 patients × ~1014 windows = ~1.46M windows
- adults: 988 patients × ~963 windows = ~951K windows
- Each window: 288 × 11 × 4 bytes (float32) = 12.7 KB
- Total all: ~18 GB uncompressed, ~6-9 GB compressed (.npz)