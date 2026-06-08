# Data

Raw data is not tracked in git. This document describes how to obtain the source datasets, the expected directory layout, and how to run the preprocessing pipeline.

---

## Source datasets

### METABONET

METABONET is a public T1D CGM dataset collected at the Hospital Clínic de Barcelona.

- **Access:** Request access through the data owners (Hospital Clínic / UdG research group). The dataset is not publicly downloadable without a data sharing agreement.
- **Files used:** `metabonet_public_train.parquet`, `metabonet_public_test.parquet`
- **Expected location:** `data/raw/`
- **Patient IDs:** numeric (e.g. `1.npz`, `42.npz`)

### T1DEXI

T1DEXI is a public T1D exercise study dataset.

- **Access:** Available through PhysioNet or the original publication repository. Requires registration.
  - Publication: Cosan et al., *Scientific Data* 2023
- **Files used:** Raw CSV files per participant from the T1DEXI data release
- **Expected location:** `data/raw/T1DEXI/`
- **Patient IDs after parsing:** prefixed with `T_` (e.g. `T_1.npz`, `T_42.npz`)

Only adult participants (age ≥ 18) from T1DEXI are used. Exercise sessions are excluded due to high feature missingness in METABONET.

---

## Expected `data/raw/` layout

```
data/raw/
├── metabonet_public_train.parquet
├── metabonet_public_test.parquet
└── T1DEXI/
    ├── <participant_id>/
    │   ├── CGM.csv
    │   ├── Bolus.csv
    │   ├── Basal.csv
    │   └── ...
    └── ...
```

---

## Preprocessing pipeline

Run these scripts in order from the repo root inside Docker:

```bash
# 1. Parse T1DEXI CSVs → parquet aligned to METABONET schema
python -u scripts/parse_t1dexi.py
# Output: data/raw/t1dexi_parsed.parquet

# 2. Merge METABONET train/test + T1DEXI → single parquet
python -u scripts/merge_datasets.py
# Output: data/raw/train.parquet, data/raw/test.parquet

# 3. Quality filter (CGM σ ≥ 15, missing ≤ 50%, ≥ 1 carb event)
python -u scripts/filter_dataset.py
# Output: data/raw/metabonet_train_filtered.parquet
#         data/interim/combined_filtered.parquet

# 4. Compute global PI/RA normalisation scalers
python -u scripts/outlier_analysis.py
python -u scripts/compute_global_scalers.py
# Output: results/outlier_analysis/global_scaler_full.npy
#         results/outlier_analysis/pretrain_patients.txt
#         results/outlier_analysis/test_patients.txt

# 5. Window + normalise → per-patient NPZ
#    (called internally by the Stage 1 training scripts;
#     also available standalone via src/preprocessing.py)
```

The windowing parameters are defined in `src/settings.py`:
- Window length: 288 steps (24 h at 5-min resolution)
- Stride: 72 steps (6 h)
- Step-change filter: removes windows where > 5% of steps have |ΔCGM_z| > 1.5

---

## Processed data format

Each patient produces one `.npz` file in `data/processed/adults_global_norm/`:

```python
data = np.load('data/processed/adults_global_norm/42.npz')
data['windows']      # shape (N, 288, 11) — N windows, 288 timesteps, 11 features
data['scaler_mean']  # per-patient CGM mean (for inverse transform)
data['scaler_std']   # per-patient CGM std
```

Feature index reference:

| Index | Feature | Normalisation |
|---|---|---|
| 0 | CGM (z-score) | Global: mean = 144.40 mg/dL, std = 57.11 mg/dL |
| 1 | PI — plasma insulin (Hovorka ODE) | Global: mean = −3.06, std = 4.22 |
| 2 | RA — carb absorption (Hovorka ODE) | Global: mean = 0.63, std = 2.48 |
| 3 | hour_sin | Circadian encoding |
| 4 | hour_cos | Circadian encoding |
| 5 | bolus flag | Binary event |
| 6 | carbs_logged flag | Binary event |
| 7–9 | AID / SAP / MDI | Zero-filled (excluded from thesis — ablation Δ ≤ 0.04) |
| 10 | age_norm | Dropped from encoder (`--no_age`) |

**Active features used in all experiments: columns 0–6 (7 features).**

Global scalers are stored in `results/outlier_analysis/global_scaler_full.npy` as a 6-element array `[cgm_mean, cgm_std, pi_mean, pi_std, ra_mean, ra_std]`.

---

## Patient split

The train/test split is fixed and stored in `results/outlier_analysis/`:

| File | Contents |
|---|---|
| `pretrain_patients.txt` | 934 patient IDs used for Stage 1 pre-training (train + val) |
| `test_patients.txt` | 103 held-out test patients |

Val patients (103) are drawn from the pretrain set at runtime; they are not excluded from `pretrain_patients.txt`.

Final cohort: **1,037 adult T1D patients** (METABONET + T1DEXI, age ≥ 18).
