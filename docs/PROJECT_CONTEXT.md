# PROJECT_CONTEXT.md — Thesis Writing Guide
_Last updated: 2026-05-13. This is the authoritative context document for thesis writing._

---

## 1. What This Thesis Is

**Title (working):** T1DFormer — to be confirmed.

**Degree:** Undergraduate biomedical engineering thesis, University of Girona (Micelab). Supervised by Prof. Josep Vehí and Prof. Juan Barrios. Work period: February to June 2026.

**Central claim:** A Transformer pre-trained exclusively on Type 1 Diabetes (T1D) multivariate physiological time series learns the underlying dynamics of glucose metabolism — not just surface statistical patterns. This is demonstrated through three pieces of evidence: (1) zero-shot gap imputation (R²=0.68), showing the encoder reconstructs missing glucose from physiological context; (2) insulin sensitivity and carb ratio recovery from globally-normalised signals (decoder ISF R²=0.499, encoder CR R²=0.406, both vs CGM stats ≈−0.03), showing the models encode absolute physiological quantities they were never explicitly trained to predict; and (3) overnight hypoglycaemia risk prediction, where the fine-tuned decoder outperforms a raw LSTM trained from scratch.

**Secondary claim:** Pre-trained representations transfer to downstream clinical tasks, with the pre-trained backbone matching or outperforming scratch baselines.

**What is explicitly out of scope:** Long-term outcome prediction, real-time deployment, label efficiency, digital twin (stub only), ISF/CR as a downstream task head (it is an analysis, not a task).

---

## 2. Differentiators from Prior Work

This is critical to get right throughout the thesis. See Bibliography.md for full detail.

**GluFormer** (Lutsker et al., Nature 2025): GPT-style autoregressive, pre-trained on 10,812 **non-diabetic** adults (Human Phenotype Project, Israel). Downstream tasks are long-term outcome stratification over a 12-year follow-up (diabetes onset risk, cardiovascular mortality). This is NOT short-horizon glucose forecasting. CGM signal only, no physiological drivers.

**CGMformer** (Lu et al., National Science Review 2025): BERT-style MLM, pre-trained on T2D and prediabetic Chinese cohorts (964 hospital + 58,847 real-world). **No T1D patients.** Downstream tasks are postprandial glucose response prediction and T2D metabolic clustering. This is NOT hypoglycaemia risk prediction. CGM signal only, no physiological drivers.

**The key differentiators of this thesis:**
1. Pre-trained exclusively on T1D patients (934 non-test patients from METABONET + T1DEXI). Both prior models use non-T1D pre-training data.
2. Integrates ODE-derived physiological driver signals (plasma insulin PI and carbohydrate absorption RA via Hovorka model) at 5-minute resolution alongside CGM. Neither GluFormer nor CGMformer uses physiological drivers.
3. Targets short-term clinical decisions specific to T1D management (hours-ahead imputation, 2h forecasting, 8h overnight hypo risk), not long-term metabolic outcomes.
4. First systematic comparison of BERT-style vs GPT-style pre-training objectives on T1D multivariate physiological time series.

**Safe positioning sentence for the thesis:**
"GluFormer and CGMformer demonstrate that self-supervised pre-training on CGM data produces generalisable representations, but both were trained on non-diabetic or Type 2 diabetic populations, neither incorporates the physiological driver signals central to T1D management, and both target long-term metabolic outcomes rather than the short-term clinical decisions that define daily T1D care."

---

## 3. Dataset

1,037 adult T1D patients (age ≥18) from two studies: METABONET (wolff2026metabonet) and T1DEXI (Riddell2023).

**Inclusion funnel:** METABONET (1,183 + 309) + T1DEXI adult (497) → 1,989 raw records → after event filter, deduplication, quality filter (CGM σ ≥ 15, missing ≤ 50%, ≥1 carb event) → **1,037 final patients**.

**Patient split (strict to avoid leakage):** 103 test / 103 val / 831 train = 934 pretrain patients. The 103 test patients are never seen during Stage 1 pre-training or Stage 2 training. This is essential for the validity of the downstream evaluation.

**Feature vector (288 steps × 10 features per window):**
- Index 0: CGM z-score (globally normalised, mean=144.40 mg/dL, std=57.11 mg/dL)
- Index 1: PI z-score (plasma insulin, Hovorka ODE — globally z-scored, mean=−3.06, std=4.22)
- Index 2: RA z-score (carb absorption, Hovorka ODE — globally z-scored, mean=0.63, std=2.48)
- Index 3–4: hour_sin / hour_cos (circadian encoding)
- Index 5–6: bolus flag / carbs_logged flag (binary event flags)
- Index 7–9: AID / SAP / MDI (one-hot therapy modality)
- Index 10: age_norm (dropped from encoder input, --no_age flag)

**PI/RA normalisation — global norm (FINAL DECISION):** PI and RA are z-scored globally using population-level statistics. This preserves absolute insulin magnitudes across patients, which is essential for recovering physiological quantities like ISF and CR. Earlier per-patient z-scoring destroyed this information and produced ISF R²=0.354 as a lower bound and null CR results. Global norm raised ISF decoder R² to 0.499 and unlocked CR encoder R²=0.406. The therapy modality heterogeneity concern (that AID micro-doses and MDI boluses would be conflated) proved less important than preserving magnitude — patients with higher insulin sensitivity genuinely have lower PI values, and that signal is recoverable once global scaling is applied. Global scalers: CGM mean=144.40 std=57.11, PI mean=−3.06 std=4.22, RA mean=0.63 std=2.48. Stored in `results/outlier_analysis/global_scaler_full.npy`.

**Windowing:** 288 steps = 24h at 5-min resolution, stride = 72 steps (6h). Step-change filter removes windows with >5% of steps having |ΔCGMz| > 1.5 (sensor artifact). 1,099,885 → 1,000,961 windows (8.99% removed). Windows retain circadian context via hour_sin/cos encoding.

**Driver blindness:** Many real-world T1D datasets have incomplete bolus and meal logging (especially MDI patients). When events are not logged, PI and RA are zero — the encoder must infer glucose dynamics from CGM shape alone. This is an acknowledged limitation discussed in the thesis. The therapy modality one-hot (AID/SAP/MDI) provides partial compensation.

---

## 4. Dataset Integration and Harmonisation

Two independent clinical studies were integrated to form the 1,037-patient cohort.

**METABONET** (wolff2026metabonet): Multi-modal metabolic dataset with CGM, insulin pump and pen records, and dietary logs. Provided as two separate parquet files (train: 1,183 records, test: 309 records). After applying an event quality filter (patients with fewer than 10 bolus and carb events were discarded as insufficiently annotated), and after identifying and removing 126 duplicate patients shared across the two splits, METABONET contributed 914 unique patients.

**T1DEXI** (Riddell2023): Type 1 Diabetes Exercise Initiative. Multi-site study with CGM, insulin pump data, and exercise annotations. 497 adult records (age ≥ 18) contributed after age filtering.

**Harmonisation steps:** Both datasets were aligned to a common 5-minute timestamp grid, a common feature schema (CGM, bolus, basal, carbs, insulin_delivery_modality, age), and a unified patient ID scheme (METABONET patients prefixed with their source, T1DEXI patients prefixed with T\_). All records were merged into a single `combined_filtered.parquet` intermediate file before the preprocessing pipeline.

**Final cohort:** After the quality filter described in Section 3, 1,037 adult T1D patients remain. The therapy modality distribution reflects the real-world heterogeneity of T1D management: a mix of AID (automated insulin delivery / closed loop), SAP (sensor-augmented pump / open loop), and MDI (multiple daily injections), with AID being the most common in METABONET and MDI more prevalent in T1DEXI.

---

## 5. Preprocessing Pipeline

Each patient is processed independently through the following steps, producing one `.npz` file per patient.

**Step 1 — Quality filter:** Computed per-patient CGM statistics (standard deviation, missing fraction, carb event presence) from the raw parquet. A patient is included only if: CGM σ ≥ 15 mg/dL (sufficient glycaemic variability), missing CGM ≤ 50%, at least one carb event logged, and age ≥ 18. This filter reduces 1,989 raw records to 1,037.

**Step 2 — CGM cleaning:** CGM values outside physiological limits are clipped to null, then small contiguous null blocks (up to a configurable number of steps) are linearly interpolated. Longer gaps are left as NaN and handled at the windowing stage.

**Step 3 — Driver features:** Missing bolus, basal, and carb values are filled with zero (absent = no event). Binary logged flags are added: `bolus_logged = (bolus > 0)` and `carbs_logged = (carbs > 0)`. These flags inform the model whether a driver event was actually recorded, partially mitigating driver blindness.

**Step 4 — Hovorka ODE (plasma insulin and carb absorption):** Plasma insulin (PI) and rate of carbohydrate absorption (RA) are computed via forward Euler integration of the Hovorka pharmacokinetic model (Hovorka 2004). PI uses a 3-compartment model separating bolus and basal components; RA uses a 2-compartment gut absorption model. Forward Euler at dt=5 minutes was chosen over scipy odeint after odeint produced NaN on long series (>50,000 steps) with large bolus values. The resulting PI and RA signals are continuous physiological curves at 5-minute resolution, transforming discrete dosing events into smooth pharmacokinetic trajectories.

**Step 5 — Temporal encoding:** Hour of day is encoded as `hour_sin = sin(2π·h/24)` and `hour_cos = cos(2π·h/24)`, giving the model continuous circadian context without discontinuity at midnight.

**Step 6 — Modality one-hot:** Insulin delivery modality (AID / SAP / MDI) is one-hot encoded into three binary columns, allowing the model to condition on therapy type.

**Step 7 — Normalisation:** All three signals (CGM, PI, RA) are z-scored globally using population-level statistics computed via the law of total variance across all 1,037 patients. CGM: mean=144.40 mg/dL, std=57.11 mg/dL. PI: mean=−3.06, std=4.22. RA: mean=0.63, std=2.48. Stored in `results/outlier_analysis/global_scaler_full.npy`. Global normalisation preserves absolute between-patient differences — a patient on MDI with large boluses will have genuinely higher PI values than an AID patient, and this signal is what enables ISF and CR recovery. The processed data is in `data/processed/adults_global_norm/`.

**Step 8 — Sliding windows:** A window of 288 steps (24 hours at 5-minute resolution) is slid over each patient's time series with a stride of 72 steps (6 hours), producing multiple overlapping 24-hour snapshots per patient. Windows with more than 50% missing CGM are discarded. A step-change filter removes windows where more than 5% of consecutive 5-minute steps have a z-score change exceeding 1.5 — this removes sensor artefacts (e.g. compression artefacts, calibration jumps) that would corrupt the pre-training signal. After filtering: 1,099,885 → 1,000,961 windows (8.99% removed, 0 patients lost).

**Step 9 — Serialisation:** Each patient is saved as a compressed `.npz` file containing: `windows (N, 288, 11)`, `scaler_mean (3,)` and `scaler_std (3,)` for CGM/PI/RA, and metadata (patient_id, modality, age).

**Output feature tensor (288 steps × 11 features):**

| Index | Feature | Normalisation |
|---|---|---|
| 0 | CGM | Global z-score (mean=144.40, std=57.11) |
| 1 | PI (Hovorka plasma insulin) | Global z-score (mean=−3.06, std=4.22) |
| 2 | RA (Hovorka carb absorption) | Global z-score (mean=0.63, std=2.48) |
| 3 | hour_sin | [-1, 1] |
| 4 | hour_cos | [-1, 1] |
| 5 | bolus_logged | Binary flag |
| 6 | carbs_logged | Binary flag |
| 7 | AID | One-hot |
| 8 | SAP | One-hot |
| 9 | MDI | One-hot |
| 10 | age_norm | age/100 (dropped from encoder, --no_age) |

---

## 6. H Analysis — Encoder Representation Quality

After pre-training, the encoder's internal representation H was analysed to validate that it learned meaningful structure. The main model is **encoder_global_norm** (global PI/RA norm). encoder_clean (per-patient norm) is archived for comparison.

**Quantitative metrics — encoder_global_norm (H_enrichment_scores.json):**

| Metric | encoder_global_norm | encoder_clean | Interpretation |
|---|---|---|---|
| PC1 variance, L1 | 79.6% | 54.2% | Global norm: early layer more dominated by dominant direction |
| PC1 variance, L5 (final) | 32.8% | 33.4% | Both converge to similar distribution at final layer |
| Probe R², L1 | 0.995 | 0.991 | Both near-perfect at L1 — still raw signal |
| Probe R², L5 (final) | **0.633** | 0.894 | Global norm: H is much less directly predictive of CGM |
| Abstraction depth | **0.367** | 0.106 | Global norm: representation moved much further from raw input |
| Feature coverage | **0.257** | 0.136 | Global norm: nearly twice as many features represented in H |
| CGM_r at L5 | −0.119 | −0.045 | Sign flip present in both — final layer inverted vs raw CGM |
| PI_r at L5 | 0.105 | 0.044 | Global norm: stronger insulin coupling in final representation |
| Distributed variance | 67.3% | 66.6% | Similar — variance well spread across PCA dimensions |
| Recon MAE (z-score) | 0.374 | ~0.42 | Global norm slightly better reconstruction MAE |
| Pearson r | 0.796 | 0.823 | encoder_clean slightly better corr (CE loss harder with global norm) |

**Key findings for thesis:**

1. **Progressive abstraction across layers.** In both models, PC1 dominance falls substantially from L1 to L5, showing the transformer distributes information across many dimensions rather than compressing into a single axis.

2. **Sign flip at L5.** The final layer representation is negatively correlated with raw CGM in both models (encoder_global_norm: CGM_r = −0.119). H is not a smoothed version of the input — the encoder has built a genuinely transformed internal representation.

3. **Global norm encoder is far more abstract.** Abstraction depth 0.367 vs 0.106, feature coverage 0.257 vs 0.136. The representation has moved much further from the raw CGM signal and encodes substantially more features — including PI dynamics that are only visible when absolute magnitudes are preserved. This is the mechanistic explanation for the improved ISF and CR recovery downstream.

4. **Reconstruction quality.** Despite higher abstraction, the global norm encoder has better reconstruction MAE (0.374 vs ~0.42). The CE val_loss is slightly higher (7.93 vs 7.83), consistent with the harder task — the model now has to predict from inputs with more inter-patient variance.

**Plots (use `results/mtsm/encoder_global_norm/` for thesis figures):**
- `abstraction_trajectory.png` — PC1 and probe R² across layers (key validation figure)
- `attention_meal.png`, `attention_post_bolus.png`, `attention_hypo.png` — event-aligned attention
- `feature_corr_per_layer.png` — feature correlations across transformer depth
- `reconstruction_examples.png`, `reconstruction_timeseries.png` — visual reconstruction examples
- `H_norm_vs_drivers.png`, `H_norm_circadian.png` — H norm profiles

**Where this fits:** Chapter 6 (Results), Stage 1 evaluation. The abstraction trajectory and at least one attention plot are strong thesis figures. Note: some attention heatmaps show near-identity attention (one bright spot) — this is genuine selective attention, not a plotting bug, and is worth noting as evidence of event-focused contextualisation.

---

## 8. Architecture

### Stage 1 — Two Pre-trained Models (both on 934 non-test patients, global PI/RA norm)

**Encoder (encoder_global_norm) — BERT-style:**
```
Input (288, 10) → Dense(128) + sinusoidal positional encoding → prepend CLS token
→ Transformer ×5 [4 heads, d_model=128, d_ff=256, dropout=0.2]
→ H (288, 128): full sequence representation (feature tokens)
→ h_cls (128,): global summary (CLS token output)
```
~641K parameters. Pre-training loss: masked CGM span cross-entropy (K=200 bins over [40, 400] mg/dL), driver-weighted (3× gradient on timesteps within 2h post-bolus/carbs). No causal auxiliary loss. Weights: `results/mtsm/encoder_global_norm/encoder_weights.weights.h5`.

**Decoder (decoder_global_norm) — GPT-style:**
Same transformer architecture with causal masking. Pre-training loss: next-token prediction (NTP) cross-entropy (K=200 bins). h_last used as causal summary. Ran full 70 epochs (no early stop — harder NTP task with global norm preserving PI/RA magnitude variance). Weights: `results/mtsm/decoder_global_norm/encoder_weights.weights.h5`.

### Stage 2 — Downstream Task Heads

General pattern: the frozen pre-trained model produces H (288, 128) and h_cls (128,), which feed lightweight task-specific heads. Variants compared: frozen encoder (fm), fine-tuned encoder (fm_ft), frozen decoder (fm_decoder), fine-tuned decoder (fm_decoder_ft), and raw LSTM trained from scratch (raw).

**App 1 — Gap Imputation (zero-shot):** The encoder reconstructs masked spans directly. No task-specific training. This is the purest test of what was learned during pre-training.

**App 2 — Forecasting (2h, 24 steps at 5-min resolution):** AttentionPool(H) → h (128,) → horizon embedding (24,32) → LSTM decoder (128, unroll=True) → Dense(64) → Dense(1) → skip connection + last CGM. Huber loss.

**App 3 — Bedtime Hypo Risk (8h nocturnal, Weibull survival):** h_cls → Dense(64, relu) → Dense(2) → [log_λ, log_k]. Weibull NLL loss with right-censoring. Risk score = P(T ≤ 96 steps) = 1 − exp(−(96/λ)^k). Filter: windows ending 20:00–23:59 only.

---

## 9. Results

### Stage 1 — Pre-training (global norm)
| Model | Result |
|---|---|
| **encoder_global_norm** | val_loss=7.93 @ ep14 (best weights restored, ran 70 epochs). Recon MAE=0.37z, Pearson r=0.796. H: probe R²=0.633 @ L5, abstraction_depth=0.367, feature_coverage=0.257, distributed_var=67.3%. Sign flip: CGM_r=−0.119. |
| **decoder_global_norm** | Ran full 70 epochs. NTP task harder with global norm (absolute PI/RA variance). |

### App 1 — Gap Imputation (zero-shot, DONE — encoder_clean weights, still valid)
| Gap | FM R² | Linear R² | Raw R² |
|---|---|---|---|
| 4h | **0.680** | 0.482 | 0.12 |
| 6h | **0.616** | 0.330 | 0.13 |
| 8h | **0.571** | 0.193 | 0.13 |

FM zero-shot beats linear interpolation at all gaps. FM vs Raw: R² 0.68 vs 0.12. Strongest single result — directly validates "learned dynamics." Note: these results used encoder_clean weights; with encoder_global_norm the numbers may shift slightly but the qualitative finding is expected to hold.

### App 2 — Forecasting (PENDING RERUN with global norm weights)
Previous results (leaky encoder2 weights, indicative only):
| Model | RMSE t+5 | RMSE t+30 | RMSE t+120 | R² t+5 |
|---|---|---|---|---|
| Naive | 11.6 | 25.1 | 54.3 | 0.955 |
| FM frozen | 10.0 | 23.1 | 45.3 | 0.966 |
| FM fine-tuned | 8.6 | 21.6 | 45.0 | 0.975 |
| Raw LSTM | 7.7 | 21.0 | 44.3 | 0.980 |
| Decoder FT | **7.1** | 21.1 | 44.5 | **0.983** |

Decoder FT beats Raw at t+5. FM frozen underperforms — bidirectional encoder is architecturally mismatched for causal generation (expected, worth discussing). Rerun with global norm weights in progress.

### App 3 — Hypo Risk (PENDING RERUN with global norm weights)
**Bedtime_01** (20:00–23:59, 8h horizon, per-patient norm — archived as secondary finding):
raw_lstm wins (AUROC=0.706). fm_decoder_ft collapsed (AUROC=0.500) — catastrophic fine-tuning failure from weak 8h Weibull gradient + limited bedtime windows. Prevalence=26.7%. Nocturnal rerun with global norm weights pending.

### Embedding Study (DONE — global norm)
| Metric | Encoder h_cls | Decoder mean-pool H |
|---|---|---|
| PCA dims (90% var) | **8** | 3 |
| GRI probe R² | **0.949** | 0.997* |
| Raw CGM stats R² (GRI) | 0.984 | 0.984 |

*Decoder 3-dim collapse dominated by mean glucose. Not a rich representation.
Results in `results/embedding_study_global_norm/plots_paper/umap_3d_encoder.html`.

### Patient-Level Analysis (DONE — global norm) — CROWN JEWEL RESULTS
| Target | Encoder h_cls | Decoder H | CGM stats |
|---|---|---|---|
| ISF (mg/dL/U) | 0.372 | **0.499** | −0.034 |
| HbA1c (%) | 0.361 | 0.402 | **0.411** |
| CR (g/U) | **0.406** | 0.271 | −0.008 |

**ISF (insulin sensitivity factor):** Decoder H R²=0.499 vs CGM stats R²=−0.034. The decoder, trained only on normalised glucose and insulin signals, recovers absolute insulin sensitivity it was never explicitly trained to predict. This is the single most powerful result in the thesis.

**CR (carb-to-insulin ratio):** Encoder h_cls R²=0.406 vs CGM stats R²=−0.008. Previously a null result under per-patient norm. Global norm unlocked this — CR depends on absolute insulin dose, which is now preserved. Encoder beats decoder (0.406 vs 0.271) because h_cls integrates the full 24h window into a global patient summary, while decoder's sequential mean-pool averages out dose-specific dynamics.

**HbA1c:** CGM stats wins (0.411). Expected — HbA1c reflects long-term average glucose, which CGM stats capture directly. Not a meaningful comparison for the central claim.

ISF and CR labels computed from raw unnormalised parquet data (absolute mg/dL/U and g/U respectively). Results in `results/patient_level_global_norm/`.

---

## 10. Thesis Structure

### Written and revised:
- **Chapter 1 — Introduction** (03_introduction.tex): Motivation, Objectives, Institutions. Revised 2026-05-13 — GluFormer/CGMformer positioning corrected, patient count fixed (934 not 1037), decoder added to Stage 1 description, label efficiency framing removed from Objective 4.

### Not yet written (pending):
- Chapter 2 — State of the Art / Related Work
- Chapter 3 — Dataset and EDA
- Chapter 4 — Conceptual Engineering (system design, problem formulation)
- Chapter 5 — Detail Engineering (architecture, preprocessing, training)
- Chapter 6 — Results and Discussion
- Chapter 7 — Conclusions

Sections 4 and 5 need reformulation because the thesis scope changed: originally framed around label efficiency and BERT vs GPT comparison as the primary contribution; now framed around "the transformer learned glucose dynamics" with ISF recovery and zero-shot imputation as primary evidence.

---

## 11. Writing Style

Academic but narrative. The thesis should read like a story, not a technical report. Use flowing sentences, guide the reader through the reasoning before stating conclusions, avoid dry passive-voice constructions. **No em dashes ever.** Use commas, semicolons, or parentheses instead. Match the tone of the Introduction as written — that is the target voice.

---

## 12. Key Facts to Get Right

- Models were pre-trained on **934** patients, not 1,037. The 103 test patients are held out throughout.
- GluFormer is pre-trained on **non-diabetic adults**, not T1D. Its downstream task is long-term outcome stratification (12-year diabetes/CVD risk), NOT short-horizon glucose forecasting.
- CGMformer is pre-trained on **T2D and prediabetic Chinese cohorts**, not T1D. Its downstream task is postprandial glucose response and T2D clustering, NOT hypoglycaemia risk prediction.
- ISF R²=0.499 (decoder) and CR R²=0.406 (encoder) are from **global PI/RA normalisation**. Both CGM stats baselines are near −0.03. These are clean results with no caveats.
- The raw LSTM baseline is not purely raw data — it uses a Conv1D + LSTM trained from scratch on the same windows. The comparison is pre-trained representation vs task-specific learned representation.
- UMAP/HbA1c/ISF embedding analysis was done but NOT compelling enough to include in thesis (correlations r≈0.23–0.29). Do not reference as a positive result.
