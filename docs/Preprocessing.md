
# Preprocessing Pipeline

**Scope:** 988 adult T1D patients (age ≥ 18). Output: one `.npz` per patient containing sliding 24h windows of a 10-feature multimodal tensor, ready for MTSM pre-training. Age is stored in the `.npz` but excluded from the encoder — see Section 7.

---

## 1. Source Datasets

### 1.1 METABONET

Observational dataset from a European multi-centre study. Contains 831 patients with Type 1 Diabetes, spanning age 1–80. Data is provided as Parquet files (`data/raw/`) at 5-minute CGM resolution. Features available per patient: CGM, bolus insulin, basal insulin, carbohydrate intake, and insulin delivery modality (AID / SAP / MDI).

### 1.2 T1DEXI

Multi-site clinical dataset from the Jaeb Center for Health Research. Contains 744 patients total: 497 adults (age 18–40) and 247 pediatric patients (age 12–17). Provided as CSV files (`data/raw/`), aligned to 5-minute resolution. Same feature set as METABONET.

### 1.3 Integration

The two datasets are merged into a single combined Parquet file:

```
data/interim/combined_filtered.parquet
```

1,575 patients total, ~126 million rows. A quality filter is then applied per patient:

- CGM standard deviation ≥ 15 mg/dL (excludes flat/sensor-failure signals)
- CGM missing data ≤ 50% (excludes patients with sparse recording)
- At least 1 carbohydrate event logged (excludes patients with no dietary data)
- Age ≥ 18 (adults only — fixed scope for this thesis)

After filtering: **988 adult patients** retained across both datasets.

---

## 2. Physiological Feature Engineering

Raw logged events — insulin boluses and carbohydrate intake — are discrete impulses in time. A transformer operating on these raw signals cannot reason about the *current physiological state* because the effect of a bolus delivered 90 minutes ago is not visible in the raw log. To make this information explicit, we use the Hovorka (2004) compartmental model to convert discrete events into continuous physiological signals.

### 2.1 Plasma Insulin (PI) — Subcutaneous Absorption Model

Subcutaneous insulin absorption follows a three-compartment chain: insulin enters subcutaneous depot S₁, transfers to a slower depot S₂, and then appears in plasma as active insulin I(t).

$$\frac{dS_1}{dt} = u(t) - \frac{S_1(t)}{\tau_{maxI}}$$

$$\frac{dS_2}{dt} = \frac{S_1(t)}{\tau_{maxI}} - \frac{S_2(t)}{\tau_{maxI}}$$

$$\frac{dI}{dt} = \frac{S_2(t)}{\tau_{maxI} \cdot V_I} - K_e \cdot I(t)$$

Where u(t) is the total insulin input (bolus + basal, mU/min) and the output I(t) is the plasma insulin concentration — the **PI** feature. Parameters:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| τ_maxI | 55 min | Time to peak subcutaneous absorption |
| V_I | 0.12 L/kg | Insulin distribution volume |
| K_e | 0.138 min⁻¹ | Plasma elimination rate |

### 2.2 Carbohydrate Absorption (RA) — Gut Absorption Model

Ingested carbohydrates transit through the gut via a two-compartment chain before appearing in blood as a glucose flux.

$$\frac{dD_1}{dt} = A_G \cdot \text{meals}(t) - \frac{D_1(t)}{\tau_D}$$

$$\frac{dD_2}{dt} = \frac{D_1(t)}{\tau_D} - \frac{D_2(t)}{\tau_D}$$

$$RA(t) = \frac{D_2(t)}{\tau_D}$$

Where meals(t) is the carbohydrate intake rate (g/min) and the output RA(t) is the rate of glucose appearance in blood — the **RA** feature. Parameters:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| A_G | 0.8 | Carbohydrate bioavailability (dimensionless) |
| τ_D | 40 min | Gut transit time constant |

### 2.3 Numerical Integration

Both ODEs are solved using **forward Euler at dt = 5 min** (matching CGM resolution), applied over each patient's full time series before windowing. `scipy.integrate.odeint` was evaluated and rejected: it produces NaN outputs on long time series (>50,000 timesteps) with extreme bolus spikes. Euler integration with dt = 5 min is numerically stable for these parameter values. Validated against odeint on 10 patients — MAE of order 10⁻², visually indistinguishable signals.

### 2.4 Discrete Event Flags

Binary indicators `bolus_logged` and `carbs_logged` are kept alongside PI and RA. The ODEs smooth the physiological response over hours, but the *precise moment* of the event — a sharp impulse at t = 0 — is erased in the smoothing. The binary flags recover this temporal anchor. This distinction matters: ablation run13 (flags removed) showed that the encoder's H representation loses its organised structure even though reconstruction MAE barely changes.

---

## 3. Signal Processing

### 3.1 CGM Cleaning

1. Clip values outside [39, 400] mg/dL → null (physiologically impossible readings from sensor artefacts)
2. Interpolate null gaps of ≤ 12 consecutive steps (1 hour) linearly
3. Gaps longer than 1 hour remain null and are handled at the windowing stage

### 3.2 Temporal Encoding

Time of day is encoded cyclically to preserve the continuity at midnight (23:55 and 00:05 should be adjacent):

```
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

These two values together encode any time of day as a point on the unit circle, avoiding the discontinuity that would appear if raw hour (0–23) were used as a feature.

### 3.3 Modality One-Hot

Each patient's insulin delivery modality is encoded as a static one-hot vector across three categories: AID (automated insulin delivery), SAP (sensor-augmented pump), MDI (multiple daily injections). The vector is repeated at every timestep. This encodes the treatment context that governs the insulin dynamics pattern.

---

## 4. Normalisation

CGM, PI, and RA are z-scored **per patient** using statistics computed from the patient's full time series:

```
x_normalised = (x − μ_patient) / σ_patient
```

The scaler parameters (μ, σ) are saved alongside the windows for inverse transform at inference time. All other features — binary flags, one-hot vectors, and hour encodings — are not normalised; they are already bounded.

**Why per-patient:** T1D patients vary enormously in baseline glucose levels, insulin sensitivity, and carbohydrate absorption rates. A global normalisation would conflate this inter-patient variability with the within-patient dynamics that the model needs to learn. Per-patient z-scoring ensures the encoder sees physiological variation, not demographic differences.

---

## 5. Windowing

Normalised per-patient time series are segmented into overlapping 24-hour windows:

- **Window size:** 288 steps = 24 hours at 5-minute resolution
- **Stride:** 72 steps = 6 hours → approximately 4 windows per day per patient
- **Null threshold:** windows with > 20% null CGM after interpolation are discarded at extraction time

### 5.1 Pathological Window Filtering

A second, stricter filter is applied at training time (inside `experiment_mtsm.py`) rather than at preprocessing time, so that different training configurations can use different thresholds without reprocessing:

```python
has_driver = ((bolus + carbs) > 0).any(axis=1)   # at least one logged event
cgm_std    = cgm.std(axis=1)
cgm_ok     = (cgm_std > 0.3) & (cgm_std < 4.0)   # not flat, not corrupted
keep       = has_driver & cgm_ok
```

This removes ~13.8% of windows across the adults dataset. Windows with no logged events provide no MTSM training signal (the masking objective requires driver context to reconstruct CGM). Windows with extreme CGM standard deviation are likely sensor artefacts.

---

## 6. Feature Tensor

The encoder receives **10 features** per timestep. Age is stored in the `.npz` as feature index 10 but is excluded from the encoder input (see note below).

| Index | Feature | Normalised | Description |
|-------|---------|------------|-------------|
| 0 | CGM | z-score per patient | Continuous glucose monitor reading |
| 1 | PI | z-score per patient | Plasma insulin — Hovorka 3-compartment ODE |
| 2 | RA | z-score per patient | Carb absorption rate — Hovorka 2-compartment ODE |
| 3 | hour_sin | — | sin(2π × hour / 24) |
| 4 | hour_cos | — | cos(2π × hour / 24) |
| 5 | bolus_logged | — | Binary: insulin bolus event at this step |
| 6 | carbs_logged | — | Binary: carbohydrate event at this step |
| 7 | AID | — | One-hot: automated insulin delivery |
| 8 | SAP | — | One-hot: sensor-augmented pump |
| 9 | MDI | — | One-hot: multiple daily injections |

**Age exclusion:** `age_norm` (age / 100) is stored in the `.npz` as feature index 10 but is not passed to the encoder. Including age caused the transformer to exploit demographic shortcuts — attention patterns became more diagonal, meaning the model relied on patient-level identity rather than learning physiological dynamics. Age will be passed directly to Stage 2 downstream task heads as a conditioning variable (late fusion). The `.npz` always contains 11 features; the encoder input is always sliced to 10.

---

## 7. Output Format

One `.npz` file per patient, stored in `data/processed/adults/`:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `windows` | (N, 288, 11) | float32 | All valid windows (11 features including age) |
| `scaler_mean` | (3,) | float32 | Per-patient mean for CGM, PI, RA |
| `scaler_std` | (3,) | float32 | Per-patient std for CGM, PI, RA |
| `patient_id` | scalar | str | Original patient identifier |
| `modality` | scalar | str | AID / SAP / MDI |
| `age` | scalar | float32 | Raw age in years |

**Scale:** 988 patients × ~963 windows average = ~951K windows total. Each window: 288 × 11 × 4 bytes (float32) = 12.7 KB. Total dataset: ~12 GB compressed.
