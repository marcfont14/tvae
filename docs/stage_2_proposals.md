# Stage 2 — Downstream Applications on H

**Context:** Stage 1 produced a bidirectional Transformer encoder pre-trained with MTSM on 988 adult T1D patients. The encoder outputs H ∈ R^(288×128) — a contextualised, driver-aware representation of each 24h window. Stage 2 attaches lightweight task-specific heads to the frozen encoder and trains them on labelled downstream tasks. The central thesis claim: **one representation, multiple clinical applications**.

---

## Shared Design Decisions

Before the individual applications, several architectural decisions apply to all of Stage 2.

### Frozen vs Fine-tuned Encoder

The encoder is **frozen by default** across all applications. This is the foundation model claim — downstream tasks are solved purely by training the head on top of H, without touching the encoder weights. Partial fine-tuning (last 1–2 encoder layers) can be tested as an ablation to measure how much performance is left on the table by freezing. The expected result: minimal gain for simple tasks, modest gain for complex tasks, at the cost of losing the "shared representation" argument.

### Cached H vs On-the-Fly

For applications where the encoder is strictly frozen, H can be **pre-computed and cached** for all training windows. This makes head training fast (no encoder forward pass per batch). For applications where encoder fine-tuning is tested, H must be computed on-the-fly.

**Recommendation:** Cache H for all initial experiments. This also ensures strict separation — the head never accidentally sees encoder gradients.

### Attention Pooling

Many downstream heads need a fixed-size representation of the window rather than the full sequence H (288×128). **Attention pooling** is the standard mechanism:

```
h_pool = softmax(v^T tanh(W H^T + b)) H    →   R^128
```

where v, W, b are learned parameters. This is one attention head over the 288 timesteps, producing a single context-weighted vector. It is preferred over mean pooling (which discards temporal structure) and over using only H_T (which ignores the full window).

### Why Bidirectionality Is Not Leakage

The encoder is bidirectional, so H_t integrates context from the entire 24h window including timesteps after t. This is not data leakage for downstream tasks. The reason: **the forecast target is always outside the input window**.

For forecasting, the label is CGM at t+30/60/120 min — which comes from the **next** 24h window, not from the current one. H encodes the dynamics of the observed window: how glucose responded to insulin, the circadian position, the pattern of driver activity. None of this contains information about the next window's CGM values. The bidirectional attention makes H a *richer encoding of the observed history*, not a lookup of the future.

The analogy holds: a doctor reviews a full day's notes bidirectionally to form the best possible understanding of a patient's current state. That understanding doesn't contain tomorrow's values — it just produces a better summary of today.

Leakage would only exist in one specific case: if we tried to predict the **end of the same window** using only the **beginning** as observed history (e.g. predicting hours 22–24 from hours 0–22, while H was computed on the full 24h including hours 22–24). The Stage 2 forecasting setup avoids this by always labelling from the **next** window. This is a deliberate design choice that must be maintained throughout implementation.

### H Heterogeneity — The Central Generalisation Challenge

This is the most important structural concern across all Stage 2 applications, and it must be addressed explicitly.

**The problem:** H is computed from one specific window — a specific patient, a specific 24h slice. The distribution of windows is highly heterogeneous:

- **Driver density:** a window with 3 meals and 4 correction boluses produces a very different H than a basal-only overnight window where PI ≈ 0 and RA = 0 throughout. The H norm analysis confirms this — windows without drivers produce flat, low-magnitude H; postprandial windows produce high, event-reactive H.
- **Patient-level systematic differences:** a patient with severe insulin resistance produces a consistently different H than a high-sensitivity patient, even for identical observed glucose trajectories. The encoder cannot fully separate "this patient is resistant" from "this patient just ate a large meal."
- **CGM variability regime:** a window from a well-controlled patient (CGM std < 0.5 z-score) occupies a different region of H space than a high-variability window (CGM std > 2.0 z-score).

When a downstream head trains on H from all these window types and learns a single mapping H → output, it faces the risk of conflating structurally different physiological situations. A basal-only H fed to a hypo prediction head may look superficially similar to a pre-hypo H with residual insulin-on-board. A high-variability H from a well-controlled patient at mealtime may resemble an H from a poorly controlled patient at baseline.

**Why H is still generalisable — and where the encoder helps:**

The encoder was trained on all window types simultaneously. It was forced to learn a representation space where basal windows, postprandial windows, high-variability windows, and different patients all coexist in a structured way. The continuous, non-clustered PCA distribution of H is evidence of this: window types and patient types lie along smooth gradients, not in disconnected islands. The head can learn to read these gradients.

Moreover, the H norm analysis shows that even "boring" basal windows are not uninformative — they encode circadian position, residual plasma insulin, the patient's characteristic overnight trajectory, and the absence of driver activity (which is itself informative: no drivers + declining glucose = very different from no drivers + stable glucose).

**The key insight:** H from any window always encodes *something* useful — the metabolic state, the driver history, the time of day. What varies is *how much* is encoded and *which aspects* dominate. The downstream head must be trained on enough diversity to handle this.

**Per-application mitigations** are detailed in each section below. The overarching principles are:

1. **Stratify training data by window type** for tasks where the relevant windows are a minority (e.g. hypo risk uses only bedtime windows; this also makes H more homogeneous).
2. **The cross-attention decoder for forecasting separates H from the driver query** — the head does not need to parse which "type" of H it received; it just queries H through the lens of the specific future driver scenario.
3. **Patient-level generalisation is the harder problem** and is the explicit motivation for App 5 (LoRA personalisation). The population head gives a reasonable approximation; LoRA adapts to the specific patient's H distribution.
4. **Evaluate per window-type stratum** — report forecasting performance separately for driver-rich vs basal-only windows. If performance collapses on basal windows, the head has overfit to event-driven dynamics.

### Window Stride and Effective Overlap for Stage 2

Stage 1 training uses 24h windows with a **6h stride**, giving 18h of overlap between consecutive windows (75% overlap). For pre-training this is fine — it maximises the number of training examples and every window is independent from the model's perspective.

For Stage 2, the 18h overlap creates two practical problems:

1. **Label leakage risk:** with 18h overlap, two consecutive windows share 18h of identical input. If consecutive (window, label) pairs are in the same training batch, the head can trivially learn to copy predictions between near-identical H representations. This is not the same as encoder leakage — the labels are from different future windows — but the head may overfit to the autocorrelation structure of H rather than learning the underlying physiology.

2. **IID assumption violation for evaluation:** test-set windows from the same patient are not independent when overlap is 18h. Evaluation metrics computed over these windows are optimistically biased — the model effectively "sees" most of the test window in the neighbouring training windows.

**Recommended stride for Stage 2 applications:**

| Application | Stride | Reason |
|---|---|---|
| Gap imputation (App 1) | 6h (unchanged) | No labels beyond the window; no leakage possible. More test samples helps. |
| Forecasting (App 2) | **12h** (every 2nd window) | Reduces H overlap to 50%; forecast labels are still fully outside the window. |
| Hypo risk (App 3) | N/A — use bedtime filter | One window per patient per night by time-of-day selection; stride is irrelevant. |
| ISF/CR (App 4) | 12h | Siamese pair is always the same window; stride affects training set diversity. |
| Digital twin (App 5) | **12h** | Decoder sees more distinct metabolic contexts during training. |

**Why 12h, not 24h (non-overlapping)?** 12h stride still produces ~2× more training examples than non-overlapping windows, while halving the problematic 18h overlap. The 12h of remaining overlap mostly covers the early-morning and late-evening boundaries of the window, which contain less driver activity and are less likely to cause correlated errors. Fully non-overlapping windows (24h stride) would reduce the training set to ~238K examples from the current 951K — a large drop in Stage 2 head training data for no clear benefit beyond clean IID sampling.

**Implementation:** no changes to the preprocessing pipeline. At Stage 2 training time, after loading H from cache (or computing on the fly), subsample windows by taking every 2nd window per patient. For bedtime hypo risk, apply the time-of-day filter instead.

---

## Application 1 — Gap Imputation (Free Result)

### Clinical Motivation

CGM sensors produce structural gaps from sensor warm-ups (2h on sensor insertion), compression artefacts during sleep, and signal interference. Standard practice is linear interpolation or last-value carry-forward. These fail to account for active insulin or carbohydrate absorption during the gap — if a patient bolused 30 minutes before the sensor dropout, linear interpolation will produce a flat line when glucose is actually falling.

Gap imputation is also the basis for **Time-in-Range (TIR) calculation**. Gaps corrupt TIR estimates. A physiologically-aware imputation directly improves clinical reporting.

### Why H Enables This

The MTSM pretext task is exactly gap imputation: mask a contiguous 5–8h span of CGM and reconstruct it from context including PI and RA. The Stage 1 encoder already learned to fill gaps using driver information. This application **requires no additional training** — it is a zero-shot capability of Stage 1.

### Architecture

```
Input window with gap (CGM = 0.0 at masked positions, drivers fully available)
        ↓
Frozen Stage 1 Encoder
        ↓
H (288, 128)
        ↓
Stage 1 Reconstruction Head  Dense(128→64, ReLU) → Dense(64→1)   [kept from Stage 1]
        ↓
Predicted CGM at all positions including gap
```

The reconstruction head was trained jointly with the encoder in Stage 1. It is kept frozen (or lightly fine-tuned) and applied directly to the masked positions.

### Label Pipeline

No new labels needed. Test by artificially introducing synthetic gaps into real CGM sequences and measuring reconstruction quality at the hidden positions. Also test on real sensor dropouts present in the dataset (CGM = 0 runs lasting > 30 min that are not MTSM training masks).

### Evaluation

| Metric | Description |
|---|---|
| RMSE at gap positions | Primary metric |
| MAE at gap positions | Primary metric |
| Clarke Error Grid Zone A | Clinical safety metric |
| TIR error (%) | Downstream impact on time-in-range calculation |

**Baselines:** linear interpolation, cubic spline, last-value carry-forward, MICE (multiple imputation).

**Key result to show:** imputation quality degrades less than baselines during post-bolus gaps, because our model uses PI dynamics to track the falling glucose.

### Challenges

- The reconstruction head was trained with random mask positions; real gaps may have different statistical properties. If performance is poor, a light fine-tuning step on real gap examples would help.
- CGM = 0.0 is used both as the mask token and as a real sensor dropout value — the model cannot distinguish them. This is acceptable for now.
- **H heterogeneity:** gap imputation is the application least affected by this concern because the task uses the same H from the same window where the gap occurs. The encoder sees the full window context (drivers before and after the gap) and the reconstruction is entirely within the window. Patient-to-patient variation in H is handled naturally — the head reconstructs within the patient's own H space.

### Priority

**Highest.** Zero additional training cost. Directly demonstrates the MTSM objective translates to a real clinical task. Should be the first implemented Stage 2 result.

---

## Application 2 — Short-Horizon Glucose Forecasting (2–4h)

### Clinical Motivation

Predicting the next 2–4 hours of glucose is the core computational task for:
- **Closed-loop AID systems:** the insulin pump controller needs a predicted trajectory to compute the next basal rate
- **Meal bolus calculators:** "how much insulin for this meal, given the current metabolic state?"
- **Exercise safety:** "will glucose go low if I start running in 30 minutes?"

The standard evaluation horizons in the literature are **30 min, 60 min, and 120 min**, matching GluFormer and CGMformer. Matching these exactly is required for a fair benchmark.

### Why H Enables This

Standard CGM forecasting models (GluFormer, NovaSAR, standard LSTMs) use only past CGM. Our model additionally conditions on **future driver information** — planned insulin and expected carb absorption. This is the main architectural novelty: a clinician or AID system can tell the model "I plan to give 3U of insulin in 15 minutes" and the forecast reflects that plan.

Without drivers, forecasting regresses to predicting the mean trajectory — the infamous "regression-to-mean" problem where all forecasts look like gradual return to the patient's baseline. With drivers as explicit conditioning, the model can forecast the glucose drop following a bolus or the rise following a meal.

### Architecture: Cross-Attention Decoder

```
Input window (observed past 24h)
        ↓
Frozen Encoder
        ↓
H (288, 128)  ←── Keys and Values

Future driver sequence c_{t:t+k}   (k = 6, 12, 24 steps for 30/60/120 min)
        ↓
Linear projection W_q → Queries (k, 128)
        ↓
Cross-Attention:  Attention(Q, K=H W_k, V=H W_v)   → (k, 128)
        ↓
MLP Dense(128→64, ReLU) → Dense(64→1)
        ↓
Predicted CGM ŷ_{t+1:t+k}
```

**Why cross-attention, not concatenation?** Concatenating H and c_future and feeding to an MLP is simpler but loses the temporal alignment between future drivers and past metabolic states. The cross-attention operation explicitly computes: "given a future insulin event (Query), which past metabolic states (Keys from H) are most relevant?" This forces the model to ground future predictions in physiologically relevant past states.

### Future Driver Specification

Three modes with increasing realism:

| Mode | Future drivers available | Appropriate for |
|---|---|---|
| Zero-driver | No future events assumed | Baseline, benchmarking against CGM-only models |
| Oracle driver | True future PI and RA | Upper bound — shows maximum benefit of driver conditioning |
| Planned driver | User-specified future insulin/carbs → converted to PI/RA via Hovorka ODE | Real clinical deployment |

For benchmarking against GluFormer/CGMformer, use zero-driver mode to match their input. Report oracle-driver mode separately to show the gain from driver conditioning.

### Label Pipeline

Windows are 24h long with a 6h stride (18h overlap between consecutive windows). The forecast target is always **beyond the current window**, so labels are read from the **raw CGM time series**, not from another window's H.

For each patient, iterate over windows:
1. End of window = forecast origin at absolute timestamp `t_end`
2. Read raw CGM from the patient's time series at `t_end + 30min`, `t_end + 60min`, `t_end + 120min`
3. Filter: discard if any target timestamp has missing CGM (raw value = 0 or NaN)

The 6h stride means the forecast target falls inside the *overlapping* portion of the next window — but we never need that window's H. The labels are scalar CGM values from the raw series. The 18h overlap between consecutive windows is irrelevant here because the forecast is always past the end of the current window.

One edge case: the very last window of a patient's recording has no future data. Discard it.

### Loss Function

**Huber loss** (δ = 1.0 in z-score units):

```
L = δ² (sqrt(1 + (y - ŷ)²/δ²) − 1)
```

Huber loss is quadratic near zero (like MSE) but linear for large errors. This prevents the model from over-smoothing to avoid penalising large rare errors, which is a known failure mode of pure MSE for glucose forecasting.

**Do not use MSE alone:** MSE minimisation predicts the conditional mean, which for glucose forecasting means the model learns to output a flat line close to the patient's mean glucose. This is the regression-to-mean problem.

### Evaluation

| Metric | Horizons |
|---|---|
| RMSE (mg/dL, back-transformed from z-score) | 30 / 60 / 120 min |
| MAE (mg/dL) | 30 / 60 / 120 min |
| Clarke Error Grid Zone A+B (%) | 30 / 60 / 120 min |
| RMSE improvement vs zero-driver | 30 / 60 / 120 min |

**Baselines:**
- Persistence (last known value)
- Linear extrapolation
- GluFormer (CGM-only Transformer, if publicly available)
- CGMformer (CGM-only BERT-style, if available)

**Key result to show:** at 120 min, driver-conditioned forecasting substantially outperforms CGM-only forecasting, particularly in post-bolus and post-meal windows.

### Challenges

- **Regression-to-mean** even with Huber loss: the model may still under-forecast the magnitude of excursions. Adding a variance-penalising term to the loss (e.g. promoting diversity of predictions) may help.
- **Oracle vs planned drivers:** real clinical deployment requires the user to specify future insulin, which introduces uncertainty (patients do not always bolus as planned). This is a known challenge for AID systems and acceptable to leave as future work.
- **H heterogeneity:** this is the most pressing concern for forecasting. A basal-only overnight H produces a very different forecast (flat, slight drift) from a postprandial H (rise then fall). The cross-attention decoder partially handles this by design — the future driver query (Q) acts as a filter on H, so the head does not need to globally "understand" the window type; it just retrieves the relevant past metabolic states through the lens of the specific upcoming driver. However, if the head is tested on a window type that was rare in training (e.g. an MDI patient with long intervals between injections), performance may degrade. **Mitigation:** stratify evaluation by window type (driver-rich vs basal-only, overnight vs daytime) and report performance per stratum. If a large gap appears, consider separate heads or a window-type conditioning input.

### Priority

**High.** Directly comparable to published literature. Driver-conditioned forecasting is the clearest architectural novelty vs GluFormer/CGMformer.

---

## Application 3 — Overnight Hypoglycaemia Risk (Survival Analysis)

### Clinical Motivation

Nocturnal hypoglycaemia is the most dangerous CGM event in T1D: the patient is asleep, cannot self-correct, and may not be woken by the CGM alarm. Current clinical practice uses static risk scores. A model that takes the pre-sleep metabolic state H and outputs "how likely is a hypo tonight, and when?" directly improves bedtime insulin decisions.

The survival framing (time-to-hypo distribution rather than binary classification) is more clinically informative: "70% chance of hypo within 4 hours" is a stronger clinical signal than "high risk."

**Threshold:** CGM < 70 mg/dL (3.9 mmol/L).

### Why H Enables This

From the H analysis: at L5, H_t norm is anti-correlated with CGM (r = −0.60). The encoder already emphasises low-glucose states. H also encodes:
- Active insulin at bedtime (PI dynamics, via the bolus-triggered analysis)
- The dawn phenomenon pattern (from the circadian analysis)
- Whether the patient has already had hypoglycaemic episodes earlier in the window (bidirectional context)

These are precisely the physiological signals known to predict nocturnal hypo.

### Task Definition

**Overnight setting (primary):**
- Input window: the last 24h window before sleep (ending at 22:00–23:00)
- Look-ahead: from window end until 08:00 the following morning (9h window)
- Event: first CGM < 70 mg/dL within the look-ahead period
- Censored: if no CGM < 70 mg/dL occurs, the observation is right-censored at 08:00

**Daytime variant (secondary):** any window, look-ahead 4h. Higher event rate but confounded by bolus/meal activity. Lower priority.

### Architecture Option A — Binary Classifier (Simpler, Benchmarkable)

Predict three binary outcomes: "hypo within 2h", "hypo within 4h", "hypo within 9h".

```
H (288, 128)
        ↓
AttentionPooling → h_pool (128,)
        ↓
Concatenate bedtime driver features c_bed: [PI_current, RA_current, bolus_last_2h, CGM_mean_last_1h]
        ↓
MLP Dense(132→64, ReLU) → Dense(64→3, sigmoid)
        ↓
[P(hypo<2h), P(hypo<4h), P(hypo<9h)]
```

Loss: binary cross-entropy at each horizon. Class weights to compensate for hypo imbalance (hypos are rare, ~10–20% of overnight windows).

**Evaluation:** AUROC, sensitivity/specificity at clinical operating point (90% sensitivity for hypo ≥ 4h horizon). This directly benchmarks against the T1D literature.

### Architecture Option B — Weibull Survival Head (Richer, More Novel)

Predict the full time-to-hypo probability distribution as a Weibull(k, λ):

```
H (288, 128)
        ↓
AttentionPooling → h_pool (128,)
        ↓
MLP Dense(128→64, ReLU) → Dense(64→2, softplus)
        ↓
Weibull parameters: k (shape), λ (scale)
        ↓
Survival function: S(t) = exp(−(t/λ)^k)
Hazard function:   h(t) = (k/λ)(t/λ)^(k−1)
```

**Loss:** negative log-likelihood with right-censoring:

```
For event at time t_i:    log f(t_i) = log k − log λ + (k−1) log(t_i/λ) − (t_i/λ)^k
For censored at T_max:    log S(T_max) = −(T_max/λ)^k
L = −mean(observed NLL + censored NLL)
```

This is the survival analysis standard. The Weibull is the simplest parametric survival distribution with an interpretable shape parameter: k < 1 = decreasing hazard (risk concentrated early), k = 1 = constant hazard (exponential distribution), k > 1 = increasing hazard (risk increases over time — typical for nocturnal hypo as insulin-on-board peaks).

**Evaluation:** Concordance index (C-statistic), integrated Brier score, calibration curve.

**Recommendation:** Implement Option A first (binary classifier). It is directly comparable to literature (AUROC) and requires no survival machinery. Option B adds novelty and richer clinical outputs. Both can be reported — Option A as the primary result, Option B as the extended analysis.

### Label Extraction Pipeline

This is the most complex data engineering step in Stage 2.

```python
# Pseudocode
for patient in patients:
    windows = load_windows(patient)   # (N, 288, 11), each window = 24h
    for w_idx, window in enumerate(windows):
        window_end_time = get_window_end_time(window)
        if not is_bedtime_window(window_end_time):   # 21:00–23:00
            continue
        # Look ahead in subsequent windows
        lookahead_cgm = get_cgm_after(patient, window_end_time, horizon_hours=9)
        hypo_times = np.where(lookahead_cgm < 70)[0]  # CGM in mg/dL
        if len(hypo_times) > 0:
            event_time = hypo_times[0] * 5 / 60       # in hours
            censored = False
        else:
            event_time = 9.0                            # 9h look-ahead max
            censored = True
        labels.append((w_idx, event_time, censored))
```

Key engineering requirements:
- Convert CGM from z-score back to mg/dL using per-patient scaler parameters (`scaler_mean[0]`, `scaler_std[0]` stored in each patient's .npz)
- The 9h look-ahead spans across multiple overlapping windows, but **labels are read from the raw time series**, not from subsequent windows' H. No window-level look-ahead index is needed — just read the raw CGM values at the required timestamps from the patient's full recording
- Filter: exclude bedtime windows where >20% of the 9h look-ahead CGM is missing in the raw series
- Expected event rate: ~15–25% of bedtime windows contain a nocturnal hypo (varies by patient)

### Challenges

- **Class imbalance:** hypo events are rare. Weighted loss, oversampling, or threshold tuning at evaluation time are all valid approaches.
- **Look-ahead requires future windows:** the label pipeline reads ahead in time, creating a dependency on future data availability. At inference time (real deployment), no look-ahead is available — this is fine, as H is always computed on the observed window.
- **Bedtime window definition:** 21:00–23:00 is a heuristic. Patients have irregular sleep schedules. Using the CGM pattern to detect sleep onset (flat nocturnal CGM after dinner peak) would be more robust.
- **H heterogeneity:** this application benefits from a natural mitigation — by restricting input windows to **bedtime windows only**, the H distribution becomes far more homogeneous. All bedtime windows share the same approximate time of day (21:00–23:00), similar driver patterns (dinner bolus 2–3h earlier, RA decaying, PI residual), and a similar CGM trajectory shape (post-dinner decline toward overnight baseline). Patient-level variation remains (some patients have much higher residual insulin at bedtime than others), but this is precisely what the model needs to exploit for risk stratification. A patient with high bedtime PI → higher hypo risk, and H encodes PI dynamics — the heterogeneity is the signal, not the noise. The more serious concern is patients with unusual bedtime dynamics (e.g. no dinner, unusual exercise) who are rare in training. Stratified evaluation by dinner-bolus presence would expose this.

### Priority

**High.** Clinically impactful, directly addresses the most dangerous T1D event, and the binary classifier version is straightforward to implement. The survival head is the novelty differentiator.

---

## Application 4 — Dynamic ISF and CR Profiling

### Clinical Motivation

Insulin Sensitivity Factor (ISF, mg/dL per unit of insulin) and Carbohydrate Ratio (CR, grams of carbs per unit of insulin) are used to calculate correction and meal boluses. In standard clinical practice, ISF and CR are **static values** programmed into the pump — the same number is used at 7am and 10pm, in summer and winter, after exercise and during illness.

In reality, insulin sensitivity varies dramatically with time of day (dawn phenomenon increases resistance), previous meals, exercise, menstrual cycle, and stress. A model that estimates the patient's *current* ISF from the metabolic state H would enable more accurate dosing.

### Why H Enables This

H encodes the current metabolic state including circadian position, active insulin, and carbohydrate absorption history. The H norm analysis shows the encoder is aware of post-bolus dynamics and dawn phenomenon timing. If the sensitivity can be extracted from H as a learned function, this is directly clinically useful.

### Architecture: Siamese Perturbation Network

The core idea: run the encoder on two nearly-identical windows — the original and a synthetically perturbed version with a small extra insulin dose — and predict the difference in glucose response.

```
Original window x
        ↓
Frozen Encoder → H (288, 128)
                                            → h_diff = [H, H'] concat at t
Perturbed window x'  (+δU insulin at t)    →     ↓
        ↓                                   MLP Dense(256→64) → Dense(64→1)
Frozen Encoder → H' (288, 128)                   ↓
                                           Predicted ΔGlucose_max (mg/dL)
```

The perturbation `δU` (e.g. +1U of fast-acting insulin) is introduced by:
1. Adding δU to the bolus dose at timestep t
2. Re-running the Hovorka ODE to compute the perturbed PI curve
3. Replacing PI in the window with the perturbed PI

The MLP head predicts ΔGlucose_max — the maximum glucose drop expected from the perturbation, which is the ISF estimate.

**Alternative perturbation for CR:** add δg of carbohydrates at timestep t, re-run Hovorka RA, predict ΔGlucose_max (glucose rise) → CR estimate.

### Indirect Validation Strategies

There are no ground-truth ISF/CR labels in the dataset. Three validation approaches:

**A. Correlation with clinician-prescribed settings:** The T1DEXI dataset includes pump configuration data. Extract the programmed ISF/CR values for each patient. Correlate the model's predicted dynamic ISF/CR (averaged over bedtime windows) with the patient's programmed static ISF/CR. Expected result: moderate positive correlation — the static value captures the population-level sensitivity, but the dynamic model adds time-of-day variation.

**B. Retrospective bolus correction accuracy:** Take observed correction bolus events where both the dose (U) and the subsequent CGM drop (ΔGlucose) are known. If the patient gave X units and glucose dropped Y mg/dL, the true ISF at that moment was Y/X. Correlate this with the model's prediction.

**C. Dawn phenomenon pattern:** If the model correctly identifies lower insulin sensitivity during 04:00–08:00 (a well-established physiological fact), this validates the circadian component without needing per-patient ground truth.

### Why This Is Harder Than It Looks

- **Hovorka ODE perturbation is slow:** re-running the ODE for every (window, timestep) pair during training is computationally expensive. A faster approximation: just shift the PI curve at timestep t by a fixed amount, bypassing the full ODE re-computation.
- **No ground-truth ISF/CR:** all three validation strategies are indirect. This is the weakest link in the application.
- **Confounding:** the observed glucose response to a bolus depends on carb intake, exercise, and other factors that may not all be captured in H.
- **H heterogeneity:** this is the most critical concern for ISF/CR profiling. The ISF estimate from the Siamese head assumes that H and H' differ only because of the perturbation. But if the two windows (original and perturbed) come from different patients, the baseline H itself differs due to patient-specific dynamics. **The Siamese approach is always applied within a single patient's window** — the head computes [H_t, H'_t] from the same window, so patient-specific baseline differences cancel in the difference H' − H. Cross-patient variation is not a problem here; within-patient variation between window types is. The head must be tested on the same range of window types (time-of-day, driver density) it was trained on.

### Priority

**Medium.** Novel idea with clear clinical value, but validation is hard. Implement after Apps 1–3 are complete. If the indirect validation strategies yield clean results, this could be a strong secondary contribution.

---

## Application 5 — Personalised Digital Twin

### Clinical Motivation

A digital twin is a patient-specific model that can simulate counterfactual scenarios:
- "What would my glucose have done if I had bolused 15 minutes earlier?"
- "Is it safe to eat this 60g carb meal right now given my current state?"
- "What happens if I reduce my basal rate by 20% overnight?"

This is the most ambitious Stage 2 application and the one that most directly fulfils the foundation model vision: a general model personalised to a specific patient, producing actionable what-if simulations.

### Why H Enables This

H encodes population-level metabolic dynamics — the general relationship between insulin, carbs, and glucose across 988 patients. Personalisation means adapting this general knowledge to the specific pharmacokinetics and pharmacodynamics of one patient (their insulin absorption rate, carb sensitivity, dawn phenomenon pattern) using only a few weeks of their data.

This is exactly the problem PEFT (Parameter-Efficient Fine-Tuning) was designed for in NLP: adapting a large general model to a specific domain or style using minimal data.

### Architecture: CVAE Decoder + LoRA

The decoder has two jobs: generate plausible trajectories (probabilistic) and be adaptable to specific patients (LoRA). These are independent additions on top of the cross-attention backbone.

**Stage A — Probabilistic Population Decoder (CVAE)**

A plain cross-attention decoder is deterministic: same H + same c → same trajectory every time. To get a distribution of plausible futures we add a **Conditional VAE (CVAE) latent variable z**.

The architecture has three components:

*1. Inference network — used only at training time*
```
Observed window → H (frozen encoder)
Future CGM window x' → lightweight encoder → h_future

Concatenate [AttentionPool(H), h_future]
        ↓
MLP → μ (d_z), log σ (d_z)      d_z = 16–32
        ↓
z ~ N(μ, σ)   via reparameterisation: z = μ + σ·ε,  ε ~ N(0,I)
```

*2. Cross-attention decoder — conditions on H, c, and z*
```
For each future timestep t:
  q_t  = Linear(c_t)   ← project driver features to d_model
  q_t += Linear(z)     ← add the same z_proj to every query timestep

  attn_t = softmax(q_t · (H·W_k)^T / sqrt(d)) · (H·W_v)
  ŷ_t    = MLP(attn_t)
```

z is injected by **adding its projection to every query vector**. This shifts all queries uniformly in the attention key-space, causing the decoder to attend to a different part of H for each z sample — producing a different trajectory shape while remaining grounded in the same physiological context.

*3. Training loss*
```
L = Huber(ŷ, x') + β · KL( N(μ,σ) || N(0,I) )
     ↑ reconstruction      ↑ forces z toward the prior
```

The KL term ensures z ~ N(0,I) at the boundary, so sampling z from the prior at inference time is valid.

*Inference — generating n counterfactual trajectories*
```
1. Compute H from patient P's observed window (frozen encoder)
2. Define counterfactual driver sequence c'
3. For i = 1..n:
     z_i ~ N(0, I)            ← sample — no future window needed
     for each t: q_t = Linear(c'_t) + Linear(z_i)
     trajectory_i = decode(Q, K=H·W_k, V=H·W_v)
4. Return {trajectory_1, ..., trajectory_n}
```

Each z_i shifts the queries differently → each trajectory explores a different plausible response to the same intervention. The spread of trajectories represents the model's uncertainty about how glucose will respond to c'.

**Stage B — Patient-Specific LoRA Fine-Tuning**

The population decoder above captures average dynamics across 988 patients. For a specific patient P, the decoder may be miscalibrated — their insulin acts faster, their carb absorption is slower, their dawn phenomenon is stronger. LoRA corrects this without touching any frozen weights.

**What LoRA is:**

For any weight matrix W ∈ R^(d_out × d_in) in the decoder, LoRA adds a low-rank residual:

```
W_effective = W_frozen  +  B · A
              (never     (rank-r, only
               updated)   these trained)

B ∈ R^(d_out × r),   A ∈ R^(r × d_in),   r ≪ min(d_out, d_in)

Parameters: r·(d_out + d_in)   vs   d_out·d_in for full fine-tuning
At r=4, d=128:  4×256 = 1,024   vs   128×128 = 16,384  (16× fewer)
```

B is initialised to zero, A to small random values → at the start of fine-tuning ΔW = 0, so the model behaves identically to the population decoder. Updates only happen where the patient's data provides signal.

**What gets LoRA adapters:** the Q and V projection matrices of each cross-attention layer. Q controls how the decoder queries H; V controls what it reads out. Adapting these two is sufficient — K (which indexes into H) is less important because H is already fixed.

**Fine-tuning procedure for patient P:**
```
Input:   14–30 days of patient P's CGM + pump data (~120 windows at 12h stride)
Frozen:  Encoder weights, decoder W matrices, CVAE inference network
Trained: LoRA A and B matrices for Q and V projections only
Loss:    Huber(ŷ, x_P') on patient P's own held-out windows
Storage: ~4–8 KB of LoRA weights per patient
```

**What the LoRA weights encode:** the low-rank matrices B·A represent the *direction in weight-space* that best corrects the population decoder for patient P. Geometrically, LoRA shifts the query-space of the cross-attention so the decoder attends to the correct parts of H for this patient's specific physiological profile.

**Complete inference for patient P:**
```
H_P     = frozen_encoder(observed_window_P)
For i = 1..n:
    z_i ~ N(0, I)
    For each t:
        q_t = (W_Q + B_Q·A_Q) · c'_t  +  W_z · z_i    ← LoRA + z
    trajectory_i = cross_attention(Q, K=H_P·W_K, V=H_P·(W_V + B_V·A_V))
```

The output is n plausible personalised counterfactual glucose trajectories for patient P under intervention c'.

**Storage at scale:** one population decoder (~2M params, ~8 MB) + one LoRA file per patient (~4 KB). For 1000 patients: 8 MB + 4 MB = 12 MB total.

### Evaluation

| Scenario | Metric |
|---|---|
| Reconstruction: population decoder | RMSE on held-out windows (mean of n=50 samples) |
| Reconstruction: LoRA fine-tuned | RMSE on patient-held-out windows |
| LoRA vs population decoder | RMSE improvement per patient |
| Calibration of uncertainty | Does the spread of n samples cover the true trajectory? (coverage at 90% CI) |
| Counterfactual plausibility | Qualitative: earlier bolus → earlier glucose drop, proportional to dose |
| Counterfactual vs ground truth | When c' = observed c, does the sample mean match observed CGM? |

**Key results to show:**
1. The n-sample distribution covers the true trajectory well (calibrated uncertainty — the model doesn't over- or under-estimate variability)
2. LoRA fine-tuning consistently improves RMSE for held-out patient windows
3. Counterfactual trajectories are physiologically plausible (qualitative examples with varying intervention)
4. 14 days of patient data is sufficient for meaningful LoRA personalisation (data efficiency curve)

### H Heterogeneity — Why This Application Directly Addresses It

The digital twin is the application where the heterogeneity concern is most explicit, and also the one where the architecture most directly resolves it.

**The problem stated precisely:** H from patient A encodes patient A's metabolic dynamics. Running a counterfactual driver sequence c' on patient A's H and expecting the output to be valid for patient B is wrong — the H space is patient-dependent. You cannot take patient A's H and patient B's future insulin plan and get a meaningful prediction.

**Why the architecture handles this:** the digital twin is always applied *within a single patient*. The workflow is:
1. Record patient P's current window → compute H_P
2. Define a counterfactual driver scenario c' for patient P (e.g. 30-min earlier bolus)
3. Decode: H_P + c' → counterfactual trajectory for patient P

H_P encodes patient P's specific metabolic state. The counterfactual only modifies c', not H. This is valid: H represents the patient's physiological context (residual insulin, circadian state, sensitivity), and c' represents the hypothetical intervention. The decoder predicts how patient P's physiology would respond to the intervention given their current state.

**Where heterogeneity is still a concern:** the population decoder was trained on 988 patients. If patient P's H consistently sits in an underrepresented region of the H space (e.g. a very unusual metabolic profile), the population decoder will produce poor counterfactuals for that patient. This is the precise motivation for LoRA fine-tuning: the LoRA adapts the decoder to patient P's region of H space using 14–30 days of their own data, so the decoding is calibrated to their specific dynamics.

**LoRA as the answer to heterogeneity:** LoRA fine-tuning is not just a personalisation technique — it is the architectural response to the H heterogeneity problem. The frozen encoder produces H that varies across patients; LoRA teaches the decoder how to decode H correctly *for this specific patient*, regardless of where their H sits in the population H space.

### What Makes This Different from Just Retraining

A naive approach: for each patient, train a new LSTM from scratch on their data. This requires hundreds of patient-days of data and produces a model that only generalises within that patient.

The LoRA approach:
- Uses the full population-level knowledge from 988 patients (encoded in the frozen encoder and population decoder)
- Requires only 14–30 days of patient data
- The LoRA weights encode how this patient's H → glucose decoding differs from the population mean — this is a direct characterisation of their pharmacokinetic profile
- New patients (no data yet) default to the population model; personalisation improves with each day of data

### Architecture Decisions Still Open

**Should the decoder also be trained jointly with the encoder from Stage 1?** No — keeping Stage 1 and Stage 2 fully separate maintains the foundation model argument. If the decoder was trained alongside the encoder, the encoder might specialise toward the decoder's task.

**What drivers go into c for the population decoder?** The same features as the encoder input: PI (plasma insulin), RA (carb absorption), bolus_logged, carbs_logged, time features. For counterfactual scenarios, c' is constructed by modifying the PI/RA curves via the Hovorka ODE for the hypothetical intervention.

**Single patient or multi-patient LoRA?** Single patient per LoRA set. Multi-patient LoRA (shared adaptation across similar patients) is theoretically interesting but adds complexity.

### Priority

**Lower — thesis outlook.** The population decoder alone (without LoRA) can be implemented and used for basic counterfactual generation. LoRA personalisation adds the novelty but requires more data engineering. Present the decoder as a Stage 2 result and LoRA as the outlook/future work section of the thesis.

---

## Implementation Priority and Dependency Map

```
Stage 1 encoder (run12)
        │
        ├─ App 1: Gap Imputation ─────────── Frozen encoder + Stage 1 head
        │                                    No new training needed
        │                                    IMPLEMENT FIRST
        │
        ├─ App 2: Forecasting ────────────── Frozen encoder + cross-attention decoder
        │                                    Requires: label pipeline (next-window labels)
        │                                    IMPLEMENT SECOND
        │
        ├─ App 3: Hypo Risk ──────────────── Frozen encoder + attention pooling + MLP/Weibull
        │                                    Requires: bedtime window detection, look-ahead labels
        │                                    IMPLEMENT THIRD
        │
        ├─ App 4: ISF/CR Profiling ──────── Frozen encoder × 2 + Siamese MLP
        │                                    Requires: Hovorka perturbation, indirect validation
        │                                    IMPLEMENT IF TIME PERMITS
        │
        └─ App 5: Digital Twin ───────────── Frozen encoder + population decoder + LoRA
                                             Requires: App 2 decoder as starting point
                                             POPULATION DECODER: implement after App 2
                                             LORA: thesis outlook
```

---

## Evaluation Framework for the Foundation Model Claim

The central thesis argument is not "our model is the best forecaster." It is: **one pre-trained encoder supports multiple downstream tasks, each benefiting from the shared representation.**

To support this claim, for each application run two versions:

1. **Foundation model (FM):** frozen Stage 1 encoder + lightweight task head
2. **Trained from scratch (TS):** same task head architecture, but encoder weights initialised randomly and trained jointly with the head on the downstream task

**Expected result:** FM ≥ TS across all tasks, with the gap larger for tasks with less labelled training data (where the pre-trained encoder's population knowledge is most valuable).

**The key table for the thesis:**

| Application | FM RMSE / AUROC | TS RMSE / AUROC | FM vs TS | Data required |
|---|---|---|---|---|
| Gap imputation | — | — | — | 0 (zero-shot) |
| Forecasting 30min | — | — | — | all windows |
| Forecasting 120min | — | — | — | all windows |
| Hypo risk (AUROC) | — | — | — | bedtime windows only |
| ISF profiling | — | — | — | correction events only |

The FM vs TS comparison is the core empirical claim of the thesis.
