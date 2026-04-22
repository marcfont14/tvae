# Stage 2 — Downstream Applications

**Status: Design complete. Implementation pending.**
**Encoder:** run21 (frozen). Weights at `results/mtsm/run21/encoder_weights.weights.h5`.

---

## 1. Overview

Stage 2 attaches lightweight task-specific heads to the frozen Stage 1 encoder. The encoder is never retrained, only the heads are trained on task-labelled data. This is the **foundation model claim**: one pre-trained encoder, multiple downstream applications.

**Core evaluation:** For each application, compare:
- **FM (Foundation Model):** frozen encoder + trained head
- **TS (Trained from Scratch):** random init encoder + trained head, same architecture

The thesis claim is FM ≥ TS, with the performance gap widening as labelled data decreases. This demonstrates the encoder has learned generalisable representations, not task-specific features.

**Model type definitions:**
- **Deterministic**: single point output; same input always yields same output at inference
- **Probabilistic**: outputs a distribution or survival curve; uncertainty is first-class
- **Generative**: samples counterfactual trajectories from a latent variable (CVAE)

**Applications at a glance:**

| #   | Application                  | Priority     | H format          | Type          | Est. time |
| --- | ---------------------------- | ------------ | ----------------- | ------------- | --------- |
| 1   | Gap Imputation               | Already done | Full H (288, 128) | Deterministic | 1–2 days  |
| 2   | Short-Horizon Forecasting    | High         | Full H (288, 128) | Deterministic | 1–2 weeks |
| 3   | Overnight Hypoglycaemia Risk | High         | Pooled h (128,)   | Probabilistic | 1 week    |
| 4   | Dynamic ISF/CR Profiling     | Medium       | Pooled h (128,)   | Deterministic | 2 weeks   |
| 5   | Digital Twin + LoRA          | Medium       | Full H (288, 128) | Generative    | 3–4 weeks |
| 6   | Next-day TIR Prediction      | Low          | Pooled h (128,)   | Deterministic | 2–3 days  |
| 7   | Anomaly / Artifact Detection | Low          | Pooled h (128,)   | Deterministic | 2–3 days  |

---

## 2. Encoder Output and How Stage 2 Uses It

### 2.1 Two Modes of H

The Stage 1 encoder produces **H ∈ R^(288 × 128)** for every 24h input window. Stage 2 uses H in two modes depending on the task:

| Mode                   | Shape      | Used for                                 | Temporal resolution                                         |
| ---------------------- | ---------- | ---------------------------------------- | ----------------------------------------------------------- |
| **Full sequence H**    | (288, 128) | Sequence-in, sequence-out tasks          | Preserved: each of 288 timesteps has its own representation |
| **Attention-pooled h** | (128,)     | Window-level classification / regression | Collapsed: single vector per window                         |

### 2.2 Attention Pooling: H → h

For window-level tasks, H is collapsed to a single vector via learnable attention pooling:

```
Learnable query vector: q ∈ R^128   (trained with the task head)

α = softmax( H · q / √128 )         attention weights: (288,)
h = Σ_t  α_t · H_t                  pooled vector: (128,)
```

This is preferred over mean pooling because α learns to concentrate weight on clinically relevant timesteps (falling glucose, bolus events, meal transitions). Mean pooling weights all 288 timesteps equally, including physiologically uninformative basal periods.

### 2.3 H is Per-Window, Not Per-Patient

The Stage 1 encoder is a **universal function**, same frozen weights for all patients. It maps any 24h window to H regardless of who the patient is. A patient with 30 days of data at 12h stride produces ~60 windows, each with its own H. 

At Stage 2 inference:
```
1. Take the relevant 24h window from the patient
2. Run through frozen Stage 1 encoder  →  H (288, 128)
3. Pass H to the Stage 2 head  →  prediction
```

Stage 1 runs fresh at every inference call. During Stage 2 **training**, H can be precomputed and cached for all windows (encoder is frozen, so H never changes).

**Which window each app uses:**

| App | Window |
|---|---|
| Gap Imputation | Window containing the gap |
| Forecasting | Current 24h window ending at prediction time |
| Hypo Risk | Bedtime window (21:00–23:00 start) |
| TIR Prediction | Current 24h window |
| ISF/CR Profiling | Current window + perturbed copy |
| Digital Twin | Current window (H is context for counterfactuals) |

---

## 3. Age Conditioning

Age is not encoded in H (excluded from encoder by design). For all Stage 2 heads that require age, it is passed as a scalar:

```
age_norm = age / 100    (stored as feature 10 in each .npz)
```

Late fusion: `concat([h, age_norm])` before the first Dense layer of the head. This keeps the encoder general and allows age to be used explicitly per task.

---

## 4. Application 1 — Gap Imputation

**Type:** Deterministic | **Est. time:** 1–2 days  
**Encoder integration:** Full H (288, 128) → reuse frozen Stage 1 reconstruction head directly (no new architecture)  
**Priority:** Done

### Input
Full H (288, 128). The window is presented with CGM masked at gap positions; PI, RA, flags, and circadian features remain visible.

### Architecture
Reuse the Stage 1 reconstruction head directly:

```
H (288, 128)
  ─── Dense(128 → 64, ReLU) ───    (288, 64)
  ─── Dense(64 → 1) ───            (288, 1)  →  ŷ_CGM per timestep
```

The head weights are from `results/mtsm/run21/model_weights.weights.h5` (full model including head). No fine-tuning. Gap positions are masked in the input; the encoder fills them from context (PI, RA, visible CGM, circadian).

This is what the encoder was trained to do (MTSM pre-training is exactly gap filling under masking). The zero-shot case evaluates whether the training objective generalises to realistic CGM gap distributions (sensor dropout, calibration interruptions) rather than the random training masks.

### Evaluation
| Metric | Description |
|---|---|
| RMSE / MAE | At gap timesteps only |
| Clarke Error Grid | Clinical accuracy zone analysis |
| TIR error | Time-in-range error: does imputed CGM reproduce TIR statistics? |

**Baselines:** linear interpolation, cubic spline, MICE (multiple imputation by chained equations).

---

## 5. Application 2 — Short-Horizon Forecasting (2–4h)

**Type:** Deterministic | **Est. time:** 1–2 weeks  
**Encoder integration:** H (288, 128) serves as K/V in a cross-attention decoder; future planned drivers are projected to Q  

**Priority:** High.

### Input
- **History:** full H (288, 128) — the encoder output for the current 24h window
- **Future drivers:** planned PI/RA/event features at forecast horizons (t+30, t+60, t+120 min)
- We predict 3 values for easier comparison with baseline 

### Architecture
Cross-attention decoder: H serves as keys and values (what the encoder knows); future planned drivers serve as queries (what the decoder is asking about).

```
Planned future drivers at [t+30, t+60, t+120]:
  ─── Dense(n_driver_features → 128) ───   Q: (3, 128)

H (288, 128)                               K: (288, 128)
                                           V: (288, 128)

Cross-attention:
  scores = softmax( Q · K^T / √128 )       (3, 288)
  context = scores · V                     (3, 128)

  ─── Dense(128 → 64, ReLU) ───
  ─── Dense(64 → 1) ───                    ŷ_CGM at t+30, t+60, t+120   (3,)
```

**Future driver modes:**
- **Oracle:** true future PI/RA read from the dataset (upper bound, not deployable)
- **Planned:** user-specified meal/bolus plan → Hovorka ODE → planned PI/RA (deployable)

The cross-attention decoder lets the model selectively read the parts of H most relevant to each forecast horizon, conditioned on what the planned drivers suggest will happen.

**Loss:** Huber(δ=1.0), which downweights large outliers relative to MSE.
**Stride:** 12h for Stage 2 (vs 6h in Stage 1) to reduce the 75% window overlap.

### Label Pipeline
Read raw CGM at absolute timestamps t+30, t+60, t+120 from the raw patient time series (outside the training window). Apply per-patient inverse z-score transform → compare in mg/dL.

### Evaluation
| Metric | Horizons |
|---|---|
| RMSE / MAE | 30 min, 60 min, 120 min |
| Clarke Error Grid | 30 min, 60 min, 120 min |

**Baselines:** GluFormer (Nature 2025), CGMformer (Nat Sci Rev 2025).

---

## 6. Application 3 — Overnight Hypoglycaemia Risk

**Type:** Probabilistic (Weibull survival) | **Est. time:** 1 week  
**Encoder integration:** H (288, 128) → attention pooling → h (128,); late fusion with age_norm  
**Priority:** High.

### Input
Bedtime windows only (21:00–23:00 start time). Attention-pooled h (128,) + age_norm (1,) via late fusion.

### Attention Pooling
```
α = softmax( H · q / √128 )    q learned with the head;  α: (288,)
h = Σ_t  α_t · H_t             h: (128,)
```

### Architecture —  Weibull Survival Head

```
h (128,)  +  age_norm (1,)
  ─── concat → Dense(129 → 64, ReLU) → Dense(64 → 32, ReLU) ───

  ─── Dense(32 → 1, softplus) ───    λ (scale parameter)
  ─── Dense(32 → 1, softplus) ───    k (shape parameter)

Survival function:  S(t) = exp( −(t / λ)^k )
Predicted time-to-hypo: E[T] = λ · Γ(1 + 1/k)
```

Weibull head gives a full survival curve, clinically richer than a binary flag. Loss: Weibull negative log-likelihood with right-censoring (windows where no hypo occurred before 08:00).

### Label Pipeline
1. Filter to bedtime windows (last CGM timestamp between 21:00–23:00)
2. Read look-ahead CGM from raw time series for the next 9h
3. Apply inverse per-patient z-score transform → mg/dL
4. Binary label: min(CGM over next T hours) < 70 mg/dL (3.9 mmol/L)

### Evaluation
| Metric | Description |
|---|---|
| AUROC | Binary classifier discrimination |
| Sensitivity / Specificity | At 90% sensitivity operating point |
| C-statistic | Survival head discrimination |
| Calibration | Reliability diagram — does predicted P match observed frequency? |

---

## 7. Application 4 — Dynamic ISF/CR Profiling

**Type:** Deterministic | **Est. time:** 2 weeks  
**Encoder integration:** Siamese — frozen encoder runs twice (original + perturbed window); pooled h and h' are concatenated and fed to the regression head  
**Priority:** Medium. No ground truth labels — indirect validation only.

### Input
Twin runs of the frozen encoder on the original and perturbed window. Both produce pooled h vectors.

### Architecture — Siamese Perturbation Network

```
Original window (288, 10)          Perturbed window (288, 10)
  ─── [Frozen Encoder] ───           ─── [Frozen Encoder] ───
  H (288, 128)                       H' (288, 128)
  ─── Attention pool ───             ─── Attention pool ───
  h (128,)                           h' (128,)

concat([h, h']) → (256,)
  ─── Dense(256 → 128, ReLU) ───
  ─── Dense(128 → 64, ReLU) ───
  ─── Dense(64 → 1) ───             ΔGlucose_max  (ISF or CR estimate)
```

**Perturbation:** synthetic +1U insulin or +10g carbs injected at t=0; Hovorka ODE re-run to compute perturbed PI and RA trajectory for the full window. The encoder sees the same CGM, flags, and circadian features in both — only PI/RA differ.

The Siamese structure lets the head learn the *differential* between physiological states, isolating the glucose response attributable to the specific perturbation.

### Evaluation
No ground truth ISF/CR labels exist. Indirect validation only:

| Metric | Description |
|---|---|
| ISF correlation | Predicted ISF vs pump-programmed ISF settings (T1DEXI metadata) |
| CR correlation | Predicted CR vs pump-programmed CR settings |
| Bolus correction correlation | Predicted ISF vs retrospective bolus correction patterns |
| Within-patient consistency | Variance of ISF estimates across windows for the same patient |

---

## 8. Application 5 — Digital Twin + LoRA

**Type:** Generative (CVAE ) | **Est. time:** 3–4 weeks  
**Encoder integration:** Stage A — H as K/V in CVAE cross-attention decoder; Stage B — LoRA adapters on Q/V projections in all 5 encoder layers  
**Priority:** Lower 

### Stage A: Probabilistic CVAE Decoder

Generates counterfactual CGM trajectories given a planned intervention. The CVAE latent variable z captures patient-specific metabolic variability not encoded in H.

```
───── Inference network (training only) ─────
AttentionPool(H) + h_future
  ─── Dense → ───    μ ∈ R^d_z,  log σ ∈ R^d_z    (d_z = 16–32)
               └──→  z ~ N(μ, σ²)

───── Decoder (training + inference) ─────
Future planned drivers c_t at each timestep t:
  q_t = Linear(c_t) + Linear(z)                    (288, 128)

Cross-attention over H (K/V):
  context_t = softmax( q_t · H^T / √128 ) · H      (288, 128)
  ─── Dense(128 → 1) per timestep ───               ŷ_t  (counterfactual CGM)

Loss: Huber(ŷ, x_future) + β · KL( N(μ, σ²) ‖ N(0, I) )
```

At inference: sample z ~ N(0, I) n times → n counterfactual CGM trajectories given the planned intervention.

### Evaluation
| Metric | Description |
|---|---|
| RMSE / MAE | Mean trajectory vs ground-truth future CGM |
| Coverage (90%) | % of ground-truth values falling within the 5th–95th percentile of sampled trajectories |
| Calibration | Does predicted uncertainty match empirical error distribution? |
| Counterfactual validity | Simulated meal/bolus perturbations produce physiologically plausible glucose responses |

### Stage B: Patient-Specific LoRA Fine-Tuning

LoRA adapts the encoder's attention patterns to a specific patient without modifying the shared weights:

```
W_effective = W_frozen + B · A

  A ∈ R^(d_in × r),  B ∈ R^(r × d_out),  r ≪ min(d_in, d_out)
```

- Apply to Q and V projection matrices in all 5 Transformer layers only
- Fine-tune on 14–30 days per-patient data (~120 windows at 12h stride)
- Storage overhead: ~4–8 KB per patient (B and A matrices only)
- Frozen weights unchanged — shared encoder structure preserved

**Why LoRA handles patient heterogeneity:** the CVAE decoder learns how to decode H for the average patient. LoRA shifts the encoder's attention patterns toward the patient's specific region of H-space — where they typically sit given their physiology, therapy, and sensitivity — without moving the shared representational structure.

---

## 9. Application 6 — Next-day TIR Prediction

**Type:** Deterministic | **Est. time:** 2–3 days  
**Encoder integration:** H (288, 128) → attention pooling → h (128,) → Dense head

### Input
Attention-pooled h (128,) from the current 24h window.

### Architecture

```
h (128,)
  ─── Dense(128 → 64, ReLU) ───
  ─── Dense(64 → 3) ───            [%TIR, %hypo (<3.9), %hyper (>10.0)]  (3,)
```

Three-output regression with softmax normalisation: the three fractions must sum to 1. Loss: MSE or Dirichlet NLL if treating outputs as proportions.

Alternatively: single output %TIR only (MSE loss) for simplicity.

### Label Pipeline
For each training window, look one full stride ahead in the same patient's .npz and compute:
```python
cgm_next = windows[i+1, :, 0]  # next 24h window CGM (z-score)
cgm_mg = cgm_next * scaler_std[0] + scaler_mean[0]  # inverse z-score
tir = ((cgm_mg >= 70) & (cgm_mg <= 180)).mean()
hypo = (cgm_mg < 70).mean()
hyper = (cgm_mg > 180).mean()
```

Label is free from existing .npz files — no external annotation required.

### Evaluation
| Metric | Description |
|---|---|
| MAE / RMSE | Predicted vs actual %TIR |
| Pearson r | Correlation of predicted and actual TIR across test set |
| Clinical zones | % predictions within ±5%, ±10%, ±15% TIR |

**Baselines:** predict mean population TIR (naïve), predict TIR of the current window (persistence).

---

## 10. Evaluation Framework

For each application:

| Comparison | Description |
|---|---|
| **FM** | Frozen Stage 1 encoder + trained Stage 2 head |
| **TS** | Random-init encoder (same architecture) + trained end-to-end |

Expected result: FM ≥ TS overall, with the gap widening as labelled data decreases. The foundation model claim requires FM to match or beat TS even with limited labels — because the frozen encoder has already learned generalisable physiological structure from 988 patients × 70 epochs of unlabelled pre-training.

**External baselines per application:**

| Application       | External baselines                                                    |
| ----------------- | --------------------------------------------------------------------- |
| Gap Imputation    | Linear interpolation, cubic spline, MICE                              |
| Forecasting       | GluFormer (Nature 2025), CGMformer (Nat Sci Rev 2025)                 |
| Hypo Risk         | Threshold-based (CGM slope, min overnight CGM)                        |
| ISF/CR            | Pump-programmed settings (as reference, not a model)                  |
| TIR Prediction    | Current-window TIR (persistence), population mean TIR (naïve)         |
| Anomaly Detection | Rule-based heuristics (rate-of-change threshold, flat-line detection) |

---

## 11. Implementation Priority

```
App 1 (Gap Imputation)    ← immediate (1–2 days)
    ↓
App 2 (Forecasting)       ← label pipeline required (1–2 weeks)
	↓
App 3 (Hypo Risk)         ← bedtime filter + look-ahead labels (1 week)
    ↓
App 4 (ISF/CR)            ← if time permits; no ground truth (2 weeks)
    ↓
App 5 (Digital Twin)      ← population CVAE after App 2; LoRA as thesis outlook (3–4 weeks)
	↓
App 6 (TIR Prediction)    ← label-free from .npz, same head pattern as App 3 (2–3 days) Optional
```
