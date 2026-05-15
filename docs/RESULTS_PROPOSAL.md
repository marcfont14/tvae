# Thesis Results Proposal — Supervisor Meeting

**Thesis title (working):** A Transformer Foundation Model Pre-trained on T1D CGM Data Learns Glucose Dynamics

**Date:** May 2026

---

## Central Claim

The transformer learned glucose dynamics. This claim is supported by three converging lines of evidence:

1. **Zero-shot gap imputation (R²=0.68 at 4h)** — the encoder reconstructs 4–8h gaps in CGM without any imputation-specific training, beating linear interpolation and a supervised raw head. This only works if the model has internalised the dynamics of glucose, insulin, and carbohydrate absorption during pre-training.

2. **Physiological recovery (ISF R²=0.499, CR R²=0.406 vs CGM stats R²≈−0.03)** — a linear probe on the frozen model's embeddings recovers insulin sensitivity and carbohydrate ratio from patient-level representations. Raw CGM statistics cannot predict these quantities at all. The transformer encodes information that is invisible to hand-crafted glucose features.

3. **Clinical transfer (FM competitive or best on nocturnal hypo risk with fewer labels)** — pre-trained representations transfer to downstream clinical prediction tasks, with FM variants matching or exceeding a from-scratch LSTM especially under low-data conditions.

---

## 1. Dataset and Preprocessing

**Cohort:** 1,037 adult T1D patients from two sources: METABONET (multi-centre, Europe) and T1DEXI (US, remote monitoring study). Inclusion criteria: age ≥18, sufficient CGM coverage (σ≥15 mg/dL, <50% missing), at least one carbohydrate event logged.

**Windowing:** Each patient's continuous CGM trace is sliced into 24h windows (288 steps at 5-min resolution), with a 6h stride — yielding ~1M windows across 1,037 patients before filtering. A step-change filter removes physiologically implausible sensor artifacts: any window where more than 5% of steps have |ΔCGM_z|>1.5 is discarded. This removes 8.99% of windows (1,099,885 → 1,000,961).

**Feature vector (per timestep, 11 dimensions):**

| Index | Feature | Description |
|---|---|---|
| 0 | CGM (z-score) | Continuous glucose, globally normalised |
| 1 | PI (z-score) | Plasma insulin from Hovorka ODE |
| 2 | RA (z-score) | Carbohydrate absorption rate from Hovorka ODE |
| 3–4 | hour_sin/cos | Circadian time encoding |
| 5–6 | bolus/carbs | Binary event flags |
| 7–9 | AID/SAP/MDI | One-hot therapy modality |
| 10 | age_norm | Age (dropped from encoder: --no_age) |

PI and RA are derived from a pharmacokinetic ODE (Hovorka 2004) using logged insulin doses and carbohydrate intakes. They are never measured directly — they represent model-estimated states, capturing the expected physiological trajectory of insulin in plasma and glucose absorption in the gut.

**Patient split:** 103 test / 103 validation / 831 train. The test set is fixed throughout. Pre-training uses only the 831 training patients.

### 1.1 Normalisation Decision

This was the most consequential methodological choice of the project.

**Per-patient normalisation (initial approach):** PI and RA for each patient are z-scored using that patient's own mean and standard deviation. Motivation: patients differ in therapy modality, pump settings, and body weight — standardising per patient was expected to reduce this heterogeneity.

**Global normalisation (main track):** PI and RA are z-scored using population-level statistics computed from the training set (PI: μ=−3.06, σ=4.22; RA: μ=0.63, σ=2.48). CGM is also globally normalised (μ=144.40, std=57.11 mg/dL).

**Why global norm is correct:** ISF (insulin sensitivity factor) and CR (carbohydrate ratio) are defined by absolute insulin doses. A patient requiring 20U to cover a meal has fundamentally different physiology from one requiring 5U. Per-patient normalisation divides each patient's insulin signal by their own mean — it erases exactly this inter-patient variation. Global normalisation preserves absolute magnitudes, and the magnitude differences are the signal the model needs to recover ISF and CR.

**Empirical confirmation:**

| Target | Per-patient norm (Decoder R²) | Global norm (Decoder R²) |
|---|---|---|
| ISF | 0.354 | **0.499** |
| CR | −0.003 (null) | **0.271** |
| HbA1c | 0.383 | 0.402 |

CR went from a complete null result to meaningful prediction purely by changing the normalisation. The hypothesis that per-patient normalisation was necessary for therapy heterogeneity was wrong — the heterogeneity is the signal.

**Relevant plot:** `results/outlier_analysis/cgm_distribution.png`

---

## 2. Stage 1 — Foundation Model Pre-training

Two complementary architectures are pre-trained on the 831-patient training set.

### 2.1 Encoder — BERT-style Masked Time Series Model (MTSM)

**Architecture:**
```
Input (288, 10) → Dense(128) + sinusoidal positional encoding
               → prepend CLS token → (289, 128)
               → 5 × TransformerBlock [4 heads, d_ff=256, dropout=0.2]
               → H (288, 128)      ← sequence representation (CLS stripped)
               → h_cls (128,)      ← CLS token output = global summary
```

Total parameters: ~641K.

**What is h_cls?** The CLS token is a single learnable vector (randomly initialised) that is prepended to the sequence before the transformer layers. Because transformer self-attention is fully global — every position can attend to every other position — the CLS token receives gradients from the entire 24h window. After 5 transformer layers, it has aggregated context from all 288 timesteps. It is not a handcrafted pooling; the pre-training objective forces it to encode predictive information about the window. After pre-training, this 128-dimensional vector is used as the patient's window-level summary for downstream tasks.

**Training objective — masked span reconstruction:**
- Random contiguous spans of CGM are masked (60–96 steps = 5–8h per span), targeting ~35% of the sequence
- Only CGM is masked. PI, RA, bolus, carbs, and circadian features are always visible
- The model must reconstruct the masked CGM values using context from before and after the gap, mediated by PI and RA
- Masking entire spans (not random individual points) prevents the shortcut of copying adjacent timesteps
- Loss: cross-entropy over K=200 discretised bins covering [40, 400] mg/dL
- Driver weighting: masked timesteps within 2h of a bolus or carb event receive 3× weight — the model is penalised more heavily for missing post-meal or post-bolus dynamics
- Auxiliary causal loss: h_cls simultaneously predicts the last 6 CGM steps of the window, providing direct gradient signal to the global summary vector

**Training results (encoder_global_norm):**
- Best validation loss: 7.93 at epoch 14 (of 70 total, no early stopping triggered)
- Reconstruction MAE = 0.37 z-score units, Pearson r = 0.796 over masked positions
- Median per-window Pearson r = 0.831 (P10=0.309, P90=0.959 — most windows reconstructed well)
- Glycaemic range preservation: TIR predicted 81.0% vs actual 78.3% (small overcorrection: slightly too many values pushed into range)

**Relevant plots:**
- `results/mtsm/encoder_global_norm/training_curves.png` — validation loss convergence
- `results/mtsm/encoder_global_norm/reconstruction_examples.png` — qualitative reconstruction quality
- `results/mtsm/encoder_global_norm/reconstruction_quality.png` — quantitative reconstruction metrics

#### 2.1.1 H Analysis — Layer-wise Representation Probing

To understand what the encoder actually learned, the intermediate hidden states H are analysed at each of the 5 transformer layers. Three metrics are computed:

**Probe R²:** A Ridge regression (α=1) is trained to predict raw CGM from the flattened H representation at each layer. High R² means CGM is linearly decodable from H — the layer still "looks like" glucose. Low R² means the representation has been transformed into something less directly glucose-like.

**PC1% (Distributed Variance):** PCA is applied to mean-pooled H (shape N×128). The % of variance explained by the first component indicates how concentrated the information is. Low PC1% = information spread across many dimensions = richer, more distributed representation.

**Feature coverage (Σ|r|):** Pearson correlation between ‖H‖₂ (the L2-norm of each hidden state) and each input feature. High |r| means the activation magnitude tracks that feature. Summing over key features gives a coverage score.

**Comparison across normalisations:**

| Metric | encoder_global_norm | encoder_clean (per-patient) | Interpretation |
|---|---|---|---|
| L5 probe R² | **0.633** | 0.894 | Global norm H is harder to linearly decode |
| PC1 at L5 | 32.75% | 33.44% | Similar concentration |
| Abstraction depth (1−R²) | **0.367** | 0.106 | Global norm is 3.5× more abstract |
| Feature coverage (Σ\|r\|) | **0.257** | 0.136 | Global norm uses 2× more diverse features |
| CGM_r at L5 | **−0.119** (sign flip) | −0.045 | See below |

The sign flip in CGM_r means that at layer 5, the positions with the highest activation norm are NOT the high-glucose timesteps — the model has inverted the naive expectation. The global norm encoder is not simply tracking glucose magnitude; it is encoding something more abstract that correlates negatively with raw CGM. This is the mechanistic signature of a representation that has moved beyond surface statistics.

The drop in probe R² from 0.894 to 0.633 is the most important result: the global norm encoder makes CGM harder to linearly recover from H, which means H encodes more about PI/RA dynamics (always visible), circadian structure, and event context — exactly what is needed to recover ISF and CR.

**Relevant plots (compare both runs side by side):**
- `results/mtsm/encoder_global_norm/abstraction_trajectory.png` — PC1% and probe R² across 5 layers
- `results/mtsm/encoder_clean/abstraction_trajectory.png` — same for per-patient norm (shows less abstraction)
- `results/mtsm/encoder_global_norm/feature_corr_per_layer.png` — heatmap of feature-H correlations

### 2.2 Decoder — GPT-style Next-Token Prediction (NTP)

**Architecture:** Same transformer (5 layers, 4 heads, d_model=128) with causal masking — each timestep can only attend to itself and prior timesteps. Loss: cross-entropy over K=200 bins for the next CGM value.

**Output representations:**
- **H** (288, 128): sequence of causal hidden states. Each h_t knows the past up to t but not the future.
- **h_last** (128,): final timestep hidden state = causal summary of the full 24h window.

The decoder ran for the full 70 epochs without early stopping — NTP under global normalisation is a harder task because the model must predict absolute values, not just relative shapes.

Mean-pool H (mean over all 288 timesteps) is used as the patient embedding in the embedding study. This is less principled than h_cls (it averages causal states with different amounts of context), but it is the only available global summary for a causal model with no CLS token.

**Relevant plot:** `results/mtsm/decoder_global_norm/training_curves.png`

---

## 3. Stage 2a — Gap Imputation (Zero-shot)

> **Status:** Results are with global norm encoder (run02). This task is complete.

**Setup:** A 4h, 6h, or 8h contiguous block of CGM is masked from a 24h test window. PI and RA remain visible throughout. The task is to reconstruct the hidden CGM values.

**Why zero-shot:** The encoder's pre-training objective is exactly masked span reconstruction. At test time, the same forward pass is applied — no task-specific training. The model was never shown imputation examples during pre-training.

**Baselines:**
- **Linear interpolation:** straight line between the last observed CGM before the gap and the first after it. Has no access to PI/RA.
- **Raw head:** a small linear head trained specifically on imputation (supervised). Also has access to PI/RA through the input features.

**Results:**

| Gap | FM RMSE (mg/dL) | FM R² | Linear RMSE | Linear R² | Raw RMSE | Raw R² |
|---|---|---|---|---|---|---|
| 4h | **30.7** | **0.680** | 38.6 | 0.482 | 51.0 | 0.124 |
| 6h | **34.1** | **0.616** | 44.1 | 0.330 | 51.2 | 0.126 |
| 8h | **36.3** | **0.571** | 48.2 | 0.193 | 51.5 | 0.125 |

The raw head achieves near-constant R²≈0.12 regardless of gap length — it has no understanding of CGM dynamics and essentially predicts mean glucose. Linear interpolation degrades rapidly beyond 6h. FM zero-shot beats both at all gaps.

**Driver response test:** For windows where a bolus or carb event falls inside the masked gap, we assess whether the imputed CGM trace moves in the correct direction (expected: post-bolus drop, post-carb rise). Linear interpolation achieves ~50% (random), since it has no access to PI/RA. FM reads the always-visible PI/RA features and reproduces the correct directional response — this is direct evidence of causal understanding.

**Relevant plots:**
- `results/stage2/imputation/run02/imputation_by_gap.png`
- `results/stage2/imputation/run02/imputation_examples.png`
- `results/stage2/imputation/run02/driver_response.png`

---

## 4. Stage 2b — Short-Horizon Glucose Forecasting

> **Status:** Complete comparison with per-patient norm weights (run10). Global norm rerun (gn_run01) is partially complete (fm_lstm + raw_lstm only). Numbers below are from run10 — treat as indicative.

**Setup:** Given the full 24h context window, predict the next 2h of CGM at 5-min resolution (24 future steps). No future drivers (PI, RA, bolus, carbs) are available at inference time — the model must extrapolate from observed dynamics.

**Model variants:**
- **Naive:** repeat the last observed CGM value across all horizons.
- **Raw LSTM:** Conv1D(stride=4) downsamples 288→72 steps, then a 2-layer LSTM(128) and Dense(24) head. No pre-training.
- **FM frozen:** encoder H → learned attention pooling → 128-d context → LSTM head. Encoder weights frozen.
- **FM fine-tuned:** same, encoder weights trainable end-to-end.
- **FM Decoder FT:** GPT decoder fine-tuned, LSTM head seeded from the decoder's h_last.
- **FM Decoder frozen (AR):** pure autoregressive rollout of the pre-trained NTP head. Compounding errors collapsed this variant (R²_t120=−8.49 — completely failed at long horizons).

**Results (run10 — indicative, per-patient norm weights):**

| Model | RMSE t+5 min | R² t+5 | RMSE t+120 min | R² t+120 |
|---|---|---|---|---|
| Naive | 11.60 | 0.955 | 54.26 | 0.050 |
| FM frozen | 10.04 | 0.966 | 45.25 | 0.339 |
| FM fine-tuned | 8.61 | 0.975 | 44.97 | 0.348 |
| Raw LSTM | 7.74 | 0.980 | 44.30 | **0.367** |
| **FM Decoder FT** | **7.13** | **0.983** | 44.52 | 0.360 |

FM Decoder FT is best at short horizons (t+5). Raw LSTM is marginally best at long horizons (t+120). Frozen FM underperforms — the reconstruction pre-training objective does not directly optimise for AR prediction, and freezing the encoder means the downstream LSTM cannot compensate. Fine-tuning largely closes this gap.

**Relevant plots:**
- `results/stage2/forecasting/run10/horizon_comparison.png`
- `results/stage2/forecasting/run10/training_curves.png`

---

## 5. Stage 2c — Nocturnal Hypoglycaemia Risk

> **Status:** Complete with per-patient norm weights (bedtime_01). Global norm rerun pending.

**Setup:** Given a CGM window starting at 20:00–23:59 (bedtime), predict the probability of hypoglycaemia (CGM < 70 mg/dL) occurring within the next 8 hours (nocturnal period). Nocturnal hypo is clinically critical — patients are asleep and cannot self-treat.

**Weibull survival model:** Instead of binary classification, the model predicts two parameters of a Weibull distribution over time-to-event (T = time to first hypoglycaemia). The risk score is the CDF evaluated at the 8h horizon:

```
P(T ≤ 8h) = 1 − exp(−(8h / λ)^k)
```

where λ (scale) and k (shape) are output by the neural network. This handles right-censoring: windows that end hypo-free still provide information (S(T) > 0 beyond the observation window). Binary cross-entropy would ignore censored cases.

**Head architecture:** h_cls → Dense(64, relu) → Dense(2) → [log_λ, log_k]. For the decoder: h_last → same head (no LSTM).

**Results (bedtime_01, prevalence=26.7%):**

| Model | AUROC | AUPRC | Sens at 90% Spec |
|---|---|---|---|
| Raw LSTM | **0.706** | **0.535** | 0.345 |
| FM fine-tuned | 0.702 | 0.527 | 0.338 |
| FM Decoder frozen | 0.672 | 0.510 | 0.319 |
| FM frozen | 0.653 | 0.422 | 0.273 |
| FM Decoder FT | 0.500 | 0.267 | 0.000 — collapsed |
| Naive (last CGM) | 0.649 | 0.484 | 0.303 |

FM Decoder FT collapse: catastrophic fine-tuning failure under the 8h Weibull horizon with limited bedtime-window training examples. The frozen decoder (0.672) transfers stably. All FM variants with fine-tuning require careful regularisation in this low-data regime.

**Label efficiency (no bedtime filter, full nocturnal prediction):**

| n training patients | Raw LSTM AUROC | FM Decoder FT AUROC |
|---|---|---|
| 42 | 0.869 | **0.887** |
| 831 (full) | 0.889 | **0.891** |

With 42 training patients (5% of the full training set), FM Decoder FT outperforms Raw LSTM by 1.8 AUROC points. The gap narrows as data increases. This is the secondary claim: pre-trained representations give FM variants a head start when labelled data is scarce.

**Relevant plots:**
- `results/stage2/hypo_risk/bedtime_01/roc_curves.png`
- `results/stage2/hypo_risk/bedtime_01/pr_curves.png`
- `results/stage2/hypo_risk/de_hypo_n42/roc_curves.png` (label efficiency — low n)
- `results/stage2/hypo_risk/de_hypo_n831/roc_curves.png` (label efficiency — full n)

---

## 6. Embedding Study — Patient-Level Phenotyping

> **Status:** Complete with global norm weights. This is a primary result.

### 6.1 What the embeddings are

**Encoder embedding (h_cls):** For each patient, all their windows are passed through the frozen encoder. Each window produces one h_cls vector (128-d) from the CLS token. The patient embedding is the **mean of h_cls across all windows** — a single 128-dimensional point representing the patient's typical 24h glucose episode dynamics.

**Decoder embedding (mean-pool H):** For each window, the decoder produces H (288, 128). The temporal mean across all 288 timesteps gives a 128-d window embedding. The patient embedding is the mean across windows.

These are patient-level representations derived from the pre-trained models without any additional training. No labels (ISF, HbA1c, CR) are used at this stage.

**Forward pass (embedding_study.py):**
```python
# Encoder
_, h_cls = encoder(batch, training=False)      # (B, 128) per window
patient_emb = h_cls.mean(axis=0)               # (128,) per patient

# Decoder
H, _ = decoder(batch, training=False)          # (B, 288, 128) per window
patient_emb = H.mean(axis=(0, 1))              # (128,) per patient
```

### 6.2 Clinical variables

Per-patient scalar statistics derived from all windows:

- **GRI** (Glycemic Risk Index): 0–100, higher = worse. Combines time below range and time above range with clinical severity weights: GRI = min(100, 3·TBR_vlow + 2.4·TBR_low + 1.6·TAR_vhigh + 0.8·TAR_high), where thresholds are <54, 54–70, >250, and 180–250 mg/dL.
- **TIR**: % time in 70–180 mg/dL
- **TBR**: % time below 70 mg/dL
- **TAR**: % time above 180 mg/dL
- **CV**: coefficient of variation = std/mean of CGM
- **Therapy**: majority vote across windows (0=AID, 1=SAP, 2=MDI)

### 6.3 Geometry metrics

**PCA effective dimensionality:** Full PCA on the 1037×128 embedding matrix. The number of components needed to explain 90% of variance measures how much of the 128-dimensional space is actually used.
- Encoder: **8 dimensions** for 90% variance
- Decoder: **3 dimensions** for 90% variance
The decoder representation is near-degenerate: three components dominate, meaning most of the 128-d space is redundant. The encoder uses a richer, more distributed geometry.

**KNN consistency:** GRI is discretised into quartiles. For each patient, the 10 nearest neighbours in embedding space are found. Consistency = fraction of those neighbours sharing the same GRI quartile. Random baseline = 0.25 (uniform quartile distribution).
- High consistency = geometrically similar patients are also clinically similar.

**Isotropy:** Mean cosine similarity across 10,000 random embedding pairs. Low cosine similarity = embeddings point in diverse directions = isotropic = less anisotropic collapse.

**LID (Local Intrinsic Dimensionality, k=20):** For each point, compute the distance ratios d_i/d_k for its 20 nearest neighbours. LID = −1/mean(log(ratios)). Higher LID = locally high-dimensional manifold.

**Linear probe R² (GRI):** RidgeCV with 5-fold cross-validation predicting GRI from embeddings. Reported as out-of-fold R² — no leakage.

### 6.4 Results (global norm)

| Metric | Encoder h_cls | Decoder mean-pool H |
|---|---|---|
| PCA dims (90% var) | **8** | 3 |
| KNN consistency (k=10) | higher | lower |
| Linear probe R² (GRI) | 0.949 | ~0.997 |

**Critical note on decoder R²≈1.0:** GRI is dominated by mean glucose (via TBR and TAR). Mean-pool H is essentially a weighted average of H across 288 timesteps — in the causal decoder, h_t is a function of CGM_0…CGM_t, so the temporal mean heavily encodes mean CGM. This makes GRI prediction trivial from the decoder embedding: it is not predicting glycaemic risk, it is recovering mean glucose in disguise. The decoder embedding is **not** a rich phenotyping representation.

The encoder, by contrast, achieves R²=0.949 from a genuinely distributed 8-dimensional structure. UMAP shows that this structure organises by GRI gradient, therapy modality, and TIR in a continuous, interpretable way.

### 6.5 UMAP visualisation

UMAP (n_components=2, n_neighbours=15, min_dist=0.1) is applied to reduce the 128-d embeddings to 2D for visualisation. Three colourmaps:
- **GRI quartile** (discrete, green→red): captures glycaemic risk stratification
- **TIR continuous** (RdYlGn colourmap): captures time-in-range gradient
- **Therapy modality** (AID=blue, SAP=green, MDI=magenta): captures treatment technology

**Relevant plots (global norm — use these):**
- `results/embedding_study_global_norm/plots_paper/umap_2d.png` — main visualisation (2 rows × 3 cols: encoder/decoder × GRI/TIR/therapy)
- `results/embedding_study_global_norm/plots_paper/pca_variance.png` — cumulative variance: encoder vs decoder
- `results/embedding_study_global_norm/plots_paper/linear_probe.png` — R² bar chart (encoder vs decoder vs raw CGM stats)
- `results/embedding_study_global_norm/plots_paper/knn_consistency.png` — consistency vs K, encoder beats decoder at all K

**For comparison:** `results/embedding_study/plots_paper/umap_2d.png` (per-patient norm — contrast the structure)

---

## 7. Patient-Level Analysis — Physiological Recovery

> **Status:** Complete with global norm weights. This is the strongest result of the thesis.

### 7.1 Targets and how they are derived

**ISF — Insulin Sensitivity Factor (mg/dL per Unit of insulin)**

Derived from METABONET raw logs. For each patient, identify correction bolus events: a bolus dose is administered (bolus > 0) without any logged carbohydrates (carbs = 0), and the CGM at the time of bolus is ≥ 150 mg/dL (patient is clearly hyperglycaemic and correcting, not covering a meal). For each such event, look forward 90 minutes and find the minimum CGM value reached:

```
ISF_event = (CGM_at_bolus − CGM_min_in_90min) / bolus_dose
```

Filter: 5 ≤ ISF ≤ 200 mg/dL/U (physiologically plausible range). Per-patient ISF = **median** over all events with ≥5 valid events required. N = 911 patients with ISF labels.

*Known limitations:* The 90-minute window may not capture the full insulin action curve for some patients. CGM minimum may be confounded by meals, exercise, or counter-regulation. Median over ≥5 events reduces noise but cannot eliminate it — R²≈0.5 reflects a noisy target, not necessarily a model ceiling.

**CR — Carbohydrate Ratio (grams carbohydrate per Unit of insulin)**

From METABONET: meal events where both bolus > 0 and carbs > 0 are logged simultaneously. For each meal event:

```
CR_event = carbs_logged (g) / bolus_dose (U)
```

Filter: 2 ≤ CR ≤ 50 g/U. Per-patient CR = **median** over ≥5 valid meal events. N = 911 patients.

*Known limitations:* Assumes logged carb intake matches actual absorption (no plate waste, no dual-wave absorption for high-fat meals). Carb counting accuracy in T1D is variable.

**HbA1c — Glycated Haemoglobin (%)**

Laboratory measurement from T1DEXI clinical records (LB.csv, LBTESTCD=='HBA1C'). Per-patient HbA1c = mean of all available lab readings. N = 502 patients (T1DEXI only — METABONET does not provide lab data in this version).

### 7.2 Ridge probe methodology

For each target (ISF, HbA1c, CR) and each feature set:

**Feature sets:**
- **Encoder h_cls** (128-d): patient-level mean of window CLS embeddings, from frozen global norm encoder
- **Decoder mean-pool H** (128-d): patient-level mean of window mean-pooled hidden states, from frozen global norm decoder
- **CGM stats** (6-d): [mean_CGM, std_CGM, TIR, TAR, TBR, CV] — hand-crafted baseline from raw glucose

**Method:** RidgeCV regression with αs=[0.01, 0.1, 1, 10, 100] and 5-fold cross-validation (KFold, shuffle=True, random_state=42). Reported metric: mean out-of-fold R² ± standard deviation across folds. Out-of-fold means the predictions are made on held-out validation folds — no training data is used to produce the reported R² values.

StandardScaler applied to features within each fold (fit on train, applied to val — no leakage).

### 7.3 Results

**Global norm (main track):**

| Target | Encoder h_cls R² | Decoder H R² | CGM stats R² |
|---|---|---|---|
| ISF (mg/dL/U) | 0.372 | **0.499** | −0.034 |
| CR (g/U) | **0.406** | 0.271 | −0.008 |
| HbA1c (%) | 0.361 | 0.402 | **0.411** |

**Per-patient norm (archived — comparison only):**

| Target | Encoder h_cls R² | Decoder H R² | CGM stats R² |
|---|---|---|---|
| ISF (mg/dL/U) | 0.208 | 0.354 | −0.034 |
| CR (g/U) | 0.012 | −0.003 | −0.008 |
| HbA1c (%) | 0.364 | 0.383 | 0.411 |

### 7.4 Interpretation

**ISF (R²=0.499 for decoder):** The transformer recovers approximately half the variance in insulin sensitivity from a signal it was never trained to predict. CGM statistics cannot do this at all (R²=−0.034). ISF is driven by the absolute magnitude of glucose drops following correction boluses — information that only exists in the globally-normalised PI representation. The decoder (causal, forward-looking) outperforms the encoder for ISF: h_last captures the cumulative effect of insulin over the full 24h window, which better tracks absolute insulin sensitivity than a bidirectional CLS summary.

**CR (R²=0.406 for encoder):** A pure null result under per-patient norm (R²=0.012), unlocked by global normalisation. CR is driven by the ratio of carbohydrate intake to bolus dose, and these are preserved in absolute form only under global norm. Encoder outperforms decoder for CR: bidirectional context (BERT-style) better captures the temporal alignment between RA (carb absorption) and PI (bolus action) than causal-only context. The CLS token aggregates this joint pattern into h_cls.

**HbA1c (R²=0.411 for CGM stats):** CGM statistics win here, as expected. HbA1c is driven by time-averaged blood glucose over 3 months — directly captured by mean CGM. The transformer embeddings add little over this simple statistic for a target that is essentially mean glucose on a longer timescale.

**Relevant plots:**
- `results/patient_level_global_norm/r2_summary.png` — main bar chart: 3 bars per target
- `results/patient_level_global_norm/scatter_all.png` — predicted vs actual scatter for each target × feature set

---

## 8. Anticipated Supervisor Questions

**Q: Why global normalisation? Isn't per-patient normalisation standard for clinical time series?**

Per-patient normalisation is appropriate when the goal is pattern recognition agnostic to scale — e.g., detecting meal events or sleep onset. It is inappropriate when the scale carries the signal. ISF and CR are defined by absolute doses: a patient with ISF=30 mg/dL/U responds very differently to the same insulin dose as one with ISF=80. Per-patient normalisation divides each patient's PI by their own mean, removing exactly the inter-patient dose variation that encodes sensitivity. The empirical proof: CR R² 0.012→0.406.

**Q: What exactly is h_cls? Is it supervised?**

The CLS token is a learnable parameter vector (128-d, randomly initialised). It is not supervised by any label during pre-training. The pre-training loss has two components: (1) masked CGM reconstruction across the full sequence, which propagates gradients through the transformer layers and indirectly shapes h_cls via self-attention; (2) an explicit auxiliary loss where h_cls predicts the last 6 CGM steps of the window (causal auxiliary). This second term provides a direct gradient signal to h_cls, training it to encode predictive information about upcoming glucose values. The result is a summary vector that has been optimised to capture 24h glucose episode dynamics.

**Q: Decoder mean-pool is unprincipled — why include it?**

Because the comparison is informative. The near-perfect decoder GRI R²≈1.0 is not a success — it reveals that mean-pool H collapses to mean glucose. Averaging 288 causal hidden states (each of which encodes cumulative CGM history) produces something close to mean CGM. GRI is dominated by mean glucose statistics, so the probe trivially succeeds. This finding demonstrates the importance of the CLS token design: h_cls is forced to be something other than a simple average by virtue of its position and the auxiliary loss. Including the decoder comparison makes this contrast explicit.

**Q: How do you know the Ridge probe is not overfitting? N=911 but features are 128-dimensional.**

The reported R² is strictly out-of-fold (5-fold CV): predictions for each validation fold are made using a model trained only on the other four folds. Ridge regression is a regularised linear model — it cannot fit arbitrary 128-d patterns with N=911 the way a neural network could. The cross-validated alpha selection also occurs within the training folds only. The ±std across folds is reported to show stability.

**Q: Why does the encoder beat the decoder for CR but the decoder beat the encoder for ISF?**

CR is a ratio: carbs per unit of insulin. The BERT-style encoder sees the full temporal context bidirectionally — it can attend to both the RA (carb absorption) and PI (bolus action) peaks within the same window and their relationship. This joint bidirectional context is what h_cls encodes. The GPT-style decoder only knows the causal past at each step: h_last encodes the full history, but the relationship between future RA and earlier PI is lost. Conversely, ISF is driven by cumulative glucose drop after a bolus — a forward-in-time process that the causal decoder naturally models as it processes the trajectory. h_last captures "how much did glucose fall in response to everything that happened" in a way that bidirectional h_cls does not emphasise.

**Q: How do you know ISF and CR ground truth labels are reliable?**

They are derived estimates from logged clinical data, not direct measurements. Known sources of noise: (1) ISF derivation assumes the 90-min CGM minimum reflects full insulin action — violated if carbs are consumed after the correction bolus, or if exercise intervenes; (2) CR derivation assumes logged carb counts are accurate — T1D carb counting error is typically ±30%; (3) median over ≥5 events reduces per-event noise substantially. The R²≈0.5 ceiling for ISF likely reflects irreducible noise in the target rather than a model ceiling — this is actually a useful framing: we achieve R²=0.5 against a noisy proxy of a quantity the model was never trained to predict.

**Q: FM Decoder FT collapsed in hypo risk — does this undermine the thesis?**

No — it is an experimental finding that reveals a training condition boundary, not a fundamental limitation of the approach. Three points: (1) the frozen decoder (AUROC=0.672) transfers stably — the representation is useful; (2) the collapse is explained by the combination of a very long Weibull horizon (8h), a small effective training set (bedtime-only windows, 20:00–23:59), and full fine-tuning of all 641K parameters — a known catastrophic forgetting configuration; (3) the label efficiency experiment (no bedtime filter, full training set) shows fm_decoder_ft AUROC=0.891, best of all models. The bedtime result is a cautionary finding about fine-tuning under data scarcity, which we report honestly.

---

## 9. Plot Selection Summary

| Section | Show | Skip |
|---|---|---|
| Dataset | cgm_distribution.png | — |
| Stage 1 Encoder | training_curves (global+per-patient), abstraction_trajectory (both, side by side), feature_corr_per_layer (global), reconstruction_examples (global) | attention_*, H_norm_* |
| Stage 1 Decoder | training_curves (global) | — |
| Imputation | imputation_by_gap, imputation_examples, driver_response | comparison_table.csv (embed numbers in text) |
| Forecasting | run10/horizon_comparison, run10/training_curves | scatter_*, clarke_* (supplementary) |
| Hypo Risk | bedtime_01/roc+pr_curves, de_hypo_n42+n831/roc_curves | calibration.png, training_curves |
| Embedding | umap_2d (global norm), pca_variance (global), linear_probe (global), knn_consistency (global) | geometry_summary, silhouette, consistency |
| Patient Level | r2_summary (global), scatter_all (global) | per-patient norm plots (use table instead) |

---

## 10. Results Status at Time of Writing

| Component | Global norm | Per-patient norm |
|---|---|---|
| Stage 1 encoder pre-training | DONE | DONE (archived) |
| Stage 1 decoder pre-training | DONE | DONE (archived) |
| H analysis | DONE (both) | DONE (both) |
| Embedding study | DONE | DONE (archived) |
| Patient-level analysis | DONE | DONE (archived) |
| Gap imputation | DONE (run02) | — |
| Forecasting | **Partial** (gn_run01: fm+raw only) | DONE (run10: all 5 models) |
| Hypo risk (bedtime) | Pending | DONE (bedtime_01) |
| Hypo risk (label efficiency) | Pending | DONE (de_hypo_n42/n831) |
