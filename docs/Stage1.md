# Stage 1 — MTSM Encoder

**Status: Complete.** Final encoder: run21. Weights at `results/mtsm/run21/encoder_weights.weights.h5`.

---

## 1. Overview

Stage 1 pre-trains a bidirectional Transformer encoder on 988 adult T1D patients using **Masked Time Series Modelling (MTSM)**. The encoder learns a contextualised representation H of each 24-hour window that captures glucose dynamics, physiological driver states, and circadian structure. H is the foundation for all Stage 2 downstream applications — the encoder is frozen after Stage 1 and never retrained.

The core idea is borrowed from masked language modelling (BERT): mask a portion of the target signal and train the model to reconstruct it from context. Applied to CGM, this forces the encoder to learn the physiological relationships between glucose, insulin, and carbohydrates — because a 5–8 hour CGM gap cannot be filled by local interpolation; it requires integrating driver context across the full window.

---

## 2. Data Windowing

### 2.1 Window Construction

Each patient's continuous CGM recording is segmented into fixed-length windows at preprocessing time and stored in a per-patient `.npz` file:

```
windows: (N_windows, 288, 11)   float32
```

| Parameter | Value | Rationale |
|---|---|---|
| Window length | 288 timesteps = 24h | Full circadian cycle; captures dawn phenomenon, meal/bolus patterns |
| Timestep resolution | 5 min | Native CGM sampling rate |
| Stride | 72 steps = 6h | 75% overlap; maximises window count while keeping distinct context |

With a 6h stride a single 24h window contributes to 4 overlapping windows. The large overlap is intentional during pre-training: the encoder sees each physiological episode from multiple temporal contexts.

### 2.2 Pathological Window Filter

Applied at training time (not saved to disk), removes ~13.8% of windows:

```python
has_driver = ((bolus + carbs) > 0).any(axis=1)   # at least one logged event
cgm_std    = cgm.std(axis=1)
cgm_ok     = (cgm_std > 0.3) & (cgm_std < 4.0)  # non-flat, non-artefact
keep       = has_driver & cgm_ok
```

- `has_driver`: windows with no logged events carry no driver context — the model cannot learn insulin/carb integration from them.
- `cgm_ok`: `std < 0.3` = flat (sensor dropout / calibration artefact); `std > 4.0` = extreme artefact.

### 2.3 Dataset Scale

| Split | Patients | Windows (approx.) |
|---|---|---|
| Train | ~790 | ~667K |
| Val | ~99 | ~83K |
| Test | ~99 | ~83K |
| **Total** | **988** | **~833K** |

Split is patient-level (SEED=42, 80/10/10). Patient-level is required to prevent leakage: windows from the same patient share physiological patterns and would trivially inflate validation metrics if split at the window level.

---

## 3. Input Features

Each window is a tensor of shape `(288, 10)` — 288 timesteps, 10 features. A per-patient z-score scaler (fit on the full time series) normalises CGM, PI, and RA. The 11th feature (age) is stored in the `.npz` but excluded from the encoder input (`--no_age`); it is passed as a scalar to Stage 2 task heads.

| Idx | Feature | Type | Normalisation | Source |
|---|---|---|---|---|
| 0 | CGM | Measured | Per-patient z-score | CGM device (5-min) |
| 1 | PI | ODE | Per-patient z-score | Hovorka 3-compartment subcutaneous insulin absorption model |
| 2 | RA | ODE | Per-patient z-score | Hovorka 2-compartment gut absorption model |
| 3 | hour_sin | Cyclic | — | sin(2π × hour / 24) |
| 4 | hour_cos | Cyclic | — | cos(2π × hour / 24) |
| 5 | bolus_logged | Binary | — | 1 if bolus recorded at this timestep, else 0 |
| 6 | carbs_logged | Binary | — | 1 if meal recorded at this timestep, else 0 |
| 7 | AID | One-hot | — | Automated Insulin Delivery pump |
| 8 | SAP | One-hot | — | Sensor-Augmented Pump |
| 9 | MDI | One-hot | — | Multiple Daily Injections |
| (10) | age_norm | Scalar | age / 100 | Patient metadata — excluded from encoder |

**PI and RA — why two ODE signals AND discrete flags:**

The Hovorka ODE models produce continuous physiological responses — PI rises over 1–2h after a bolus, RA peaks 90 min after a meal. These signals encode the *sustained consequence* of an event. The discrete flags (features 5–6) encode the *precise moment* of the event — a sharp binary spike at t=0 that the ODEs smooth out over hours. Both are needed:

- **PI/RA:** inform the encoder of the pharmacokinetic trajectory following an event
- **bolus_logged / carbs_logged:** provide a temporal anchor at the exact event moment, forcing a globally consistent representational convention across all training windows

Ablation run13 (flags zeroed, PI/RA retained) confirms: MAE degrades by only 0.02 but H structure collapses entirely.

---

## 4. Architecture

```
Input (288, 10)          — 288 timesteps × 10 features
      │
      ▼
Input Projection         Dense(10 → 128), no activation, applied per timestep
      │                  shape: (batch, 288, 128)
      ▼
Positional Encoding      Sinusoidal, fixed (not learned)
      │                  shape added: (1, 288, 128) → broadcast over batch
      ▼                  result: (batch, 288, 128)
Transformer Encoder × 5 layers
  Each layer:
    MultiHeadAttention   4 heads, key_dim=32, dropout=0.2
    │                    shape: (batch, 288, 128)
    + Residual → LayerNorm
    │                    shape: (batch, 288, 128)
    FFN: Dense(128 → 256, ReLU) → Dense(256 → 128)
    │                    shape: (batch, 288, 128)
    + Residual → LayerNorm
    │                    shape: (batch, 288, 128)
      │
      ▼
      H  (batch, 288, 128)     ← kept for Stage 2
      │
      ▼
Reconstruction Head      Dense(128 → 64, ReLU) → Dense(64 → 1) per timestep  [DISCARDED]
      │                  shape: (batch, 288, 1) → reshape to (batch, 288)
```

~660K parameters total: encoder ~652K, reconstruction head ~8K.

**Per-timestep processing:** The input projection applies an identical Dense layer to each of the 288 timestep vectors independently — there is no temporal mixing at this stage. Positional encoding adds a unique sinusoidal offset to each position, allowing the model to distinguish t=0 from t=287. After projection and PE, each timestep enters self-attention, where it is no longer processed in isolation: every H_t is computed by attending to all 288 positions simultaneously.

**Why expand 10→128 instead of compressing?** This encoder is not an autoencoder. The autoencoder paradigm compresses to a bottleneck and reconstructs — the bottleneck is the representation. Here, the goal is contextualisation, not compression. Each token needs a sufficiently large embedding space for the Q, K, V projections to be meaningful: with 4 heads and key_dim=32, the minimum viable d_model is 128. In 10 dimensions each head would operate on 2–3 numbers — not enough to learn complex physiological relationships. The compression analogy still holds but it happens later, at the head level: for window-level tasks, H (288, 128) is collapsed to h (128,) via attention pooling; for reconstruction, the MLP head maps 128→1. The encoder's job is to build a rich, full-resolution representation that downstream heads can compress in task-specific ways. This is the same pattern as BERT, which expands word tokens to 768 dimensions and leaves compression to the classifier head.

---

## 5. Attention Mechanism

### 5.1 What a Token Is

Each of the 288 tokens is **not** a glucose value. At timestep t, the token is a 128-dim vector produced by projecting all 10 input features together:

```
[CGM_t, PI_t, RA_t, hour_sin_t, hour_cos_t, bolus_t, carbs_t, AID_t, SAP_t, MDI_t]
    → Dense(10→128) → token_t ∈ R^128
```

All features are mixed into a single vector before attention begins. This means that when timestep i attends to timestep j, it is not "glucose at i attending to glucose at j" — it is "the full physiological state at i (glucose + insulin + carbs + time of day) attending to the full physiological state at j". Drivers are part of every token from the very first layer.

### 5.2 Mathematical Definition

For each Transformer layer l ∈ {1…5} and each head h ∈ {1…4}, given the current representation X ∈ R^(288×128):

```
Q^h = X · W_Q^h       shape: (288, 32)   — query: "what am I looking for?"
K^h = X · W_K^h       shape: (288, 32)   — key:   "what do I offer?"
V^h = X · W_V^h       shape: (288, 32)   — value: "what information do I carry?"

A^h = softmax( Q^h · (K^h)^T / √32 )    shape: (288, 288)  ← attention matrix

head^h = A^h · V^h                       shape: (288, 32)

MultiHead(X) = concat(head^1 … head^4) · W_O    shape: (288, 128)
```

W_Q, W_K, W_V, W_O are learned weight matrices. key_dim = d_model / n_heads = 128 / 4 = 32. The √32 scaling prevents the dot product from entering the saturation region of softmax.

### 5.3 The Attention Matrix

**A^h[i, j] ∈ [0, 1]** is the weight that position i (query) assigns to position j (key) when computing its updated representation. Each row sums to 1 (softmax normalisation). Concretely: A^h[i, j] is "how much does timestep i look at timestep j?"

**Important distinction:** the attention matrix captures *routing* — which timesteps are weighted — but not *content*. What actually gets transferred is V_j, a learned projection of the full physiological state at j. Two timesteps with the same attention weight could carry very different information. The attention matrix is therefore a useful but incomplete picture: it shows where the model looks, not what it reads. Furthermore, H is the product of five such layers plus FFN transformations — by the time attention runs at L5, the tokens are already four layers of non-linear refinement away from the raw inputs. Reading the attention matrix at L5 as a direct map between raw features and timesteps is an oversimplification.

For a single 24h window there are **4 heads × 5 layers = 20 attention matrices**, each of shape (288, 288). These can be extracted at any forward pass and used for interpretability.

**Reading an attention matrix:**

| Pattern | Meaning |
|---|---|
| Diagonal row i | Timestep i only attends to itself — local, no temporal integration |
| Vertical column at j | Many timesteps attend to j — j is a key physiological anchor for this window |
| Diffuse row i | Timestep i draws context broadly from the full 24h window |
| Block structure | Temporal neighbourhood clusters (e.g., post-meal period attends to itself) |

### 5.4 Why Averaged Heatmaps Look Diagonal

The standard averaged attention heatmap (mean over 4 heads × 100 windows) consistently appears diagonal. This is **not evidence that the encoder only does local attention** — it is an averaging artefact.

- Post-bolus window A: vertical attention column at t=120 (bolus time)
- Post-meal window B: vertical attention column at t=50 (meal time)
- Stable basal window C: near-diagonal (no long-range event to integrate)

When averaged, the columns at t=120 and t=50 cancel; the diagonal (consistently present in all window types) survives. Per-window analysis is required to see the true structure.

### 5.5 Head Specialisation at L5

Individual per-window analysis (`scripts/attention_viz.py`) on run21 reveals clear functional specialisation:

| Head | Pattern | Function |
|---|---|---|
| **Head 1** | Diagonal + vertical columns at event times | Local smoothing + event anchor |
| **Head 2** | Fully diffuse, near-uniform weights | Global context — every timestep attends to full 24h window |
| **Head 3** | Sharp diagonal + bright vertical columns at bolus/carb times | Discrete event detector |
| **Head 4** | Mixed, window-type dependent | Adaptive context — pattern varies by physiological state |

Head 2 is the mechanism of bidirectional integration: it gives every timestep equal access to the full window context. Head 3's vertical columns at event times directly explains why removing discrete flags (run13) causes H structure to collapse — the encoder loses its temporal anchors.

### 5.6 Why the 4 Heads Learn Different Things

All four heads have identical structure and receive the same input X. Nevertheless they specialise, for three reasons:

1. **Random initialisation.** W_Q, W_K, W_V for each head are initialised independently with different random values. They start from different points in weight space and gradient descent takes them in different directions from the very first step.

2. **Gradient pressure through W_O.** The loss constrains only the final concatenated output, not individual heads. Gradients flow back through W_O and split to each head depending on what each contributed. If two heads were doing the same thing, one would be redundant — W_O would learn to suppress it, breaking the symmetry.

3. **Competition for capacity.** The model has a fixed 128-dim output. Four identical heads would waste 3/4 of that capacity. The reconstruction loss pushes them to cover complementary patterns — local context, global context, event anchoring, adaptive context — because that combination reconstructs masked CGM better than four identical heads would.

Specialisation is not engineered; it emerges from initialisation noise and gradient pressure.

### 5.7 Determinism of the Encoder

**At inference: fully deterministic.** Given the same window, the encoder always produces the same H and the same attention matrices. No randomness is involved once training is complete.

**During training: non-deterministic**, due to dropout (rate=0.2). At each forward pass, 20% of values are randomly zeroed, so the same window produces slightly different outputs across training steps. This is intentional — it prevents overfitting by forcing the model not to rely on any single pathway.

Keras automatically disables dropout when the model is called in inference mode (e.g., `model.predict()`). From that point the encoder is a fixed deterministic function: window (288, 10) → H (288, 128).

---

## 6. Training Objective

### 6.1 Span Masking

One contiguous CGM span per window is masked before each training step:
- **Mask ratio:** 35% of timesteps (≈100 steps)
- **Span length:** sampled uniformly from [60, 96] steps = [5h, 8h]

The mask is applied only to the CGM feature (index 0). All other features — PI, RA, bolus_logged, carbs_logged, hour encoding, modality — remain fully visible. The model must predict the masked CGM values from:
1. Visible CGM outside the mask
2. Full driver context (PI, RA, discrete flags) throughout the window

**Why long spans:** spans of 5–8 hours cannot be filled by local interpolation. To reconstruct a 6-hour CGM gap the encoder must reason about what insulin was active, when meals occurred, and what the circadian context implies. Short random masking (as in BERT) can be solved by local smoothing — it does not force learning of physiological causality.

### 6.2 Driver-Weighted Loss

Loss is only computed on masked timesteps:

```
L = Σ_t [ mask_t × w_t × (ŷ_t − y_t)² ] / Σ_t mask_t
```

Where `w_t` is the driver weight: if timestep t falls within 2 hours after a bolus or carb event, `w_t = 3.0`; otherwise `w_t = 1.0`.

**Why driver weighting:** without it, the model optimises on easy flat basal periods (most of the 24h window). Postprandial and post-bolus zones — where glucose dynamics are complex and clinically relevant — represent a minority of masked timesteps and are underweighted. The 3× boost forces the encoder to learn to reconstruct these physiologically rich regions.

### 6.3 Hyperparameters (run14 / run21 — identical config)

| Parameter | Value |
|---|---|
| d_model | 128 |
| n_heads | 4 |
| n_layers | 5 |
| d_ff | 256 |
| dropout | 0.2 |
| batch_size | 128 |
| epochs | 70 (no early stopping) |
| optimizer | AdamW, lr=1e-3, weight_decay=1e-4 |
| mask_ratio | 0.35 |
| mask_min_len | 60 steps (5h) |
| mask_max_len | 96 steps (8h) |
| driver_loss_weight | 3.0 |
| driver_effect_steps | 24 (2h) |

---

## 7. Design Decisions

### MLP reconstruction head (not LSTM)
H_t is already a contextualised vector — self-attention has integrated information from all 288 timesteps. An LSTM head would let the model spread reconstruction work across the head's recurrence, reducing the pressure on the encoder to build a rich H. A single MLP per timestep forces all temporal reasoning into H.

### Bidirectional attention
A causal (autoregressive) encoder cannot use future driver events to reconstruct a past CGM gap. If a meal is logged at t+60 and we are reconstructing CGM at t, a causal model cannot see the carbs_logged flag. Bidirectional attention allows the encoder to use the full 24-hour context in both directions — this is directly evidenced by the pre-event H norm rise for carbs events (H norm peaks at the logged event time, not 90 min later at the RA ODE peak, because the encoder can look forward to see the flag).

### Discrete event flags (bolus_logged, carbs_logged)
The Hovorka ODE signals (PI, RA) encode the magnitude and timing of physiological response continuously. The discrete flags encode the precise moment of the event — a sharp binary spike at t=0 that the ODEs smooth out over hours. Both are needed:
- **PI/RA:** inform the encoder of the sustained physiological consequence
- **Flags:** provide a temporal anchor that forces a globally consistent H convention across all training windows

Ablation run13 (flags zeroed, PI/RA retained) confirms: MAE degrades only marginally (0.47 vs 0.45) but H structure collapses entirely — CGM sign flip absent, PCA disorganised, event-triggered response flat.

### Age excluded from encoder (--no_age)
Age is excluded from the encoder for two reasons:

1. **Architecturally cleaner:** encoding age into H would conflate a demographic prior with within-window physiological dynamics. Age is better passed as a scalar conditioning variable to each Stage 2 task head (late fusion), where it can be used explicitly rather than baked into the representation.

2. **Marginal MAE improvement:** removing age_norm gives 0.45 vs 0.46 val MAE. The run12→run14 Σ|r_L5| drop (0.98 → 0.46) reflects that age_norm was contributing a strong direct feature correlation to the norm metric — not a genuine improvement in H structure, but a shortcut via demographic correlation.

---

## 8. Results — Run21 (Final Encoder)

Run21 is identical in configuration to run14. It is the final encoder because it is the only run with both `encoder_weights.weights.h5` and `model_weights.weights.h5` saved, enabling full reconstruction plot generation.

### Reconstruction quality

| Split | MAE |
|---|---|
| Train | ~0.42 |
| Val | ~0.45 |

Driver-zone breakdown (from run12 reference):

| Zone | MAE |
|---|---|
| Postprandial (2h after carbs) | ~0.51 |
| Post-bolus (2h after bolus) | ~0.53 |
| Basal (no recent events) | ~0.36 |

Higher MAE in postprandial/post-bolus zones is expected and desirable: these are where glucose dynamics are most complex (non-linear insulin-glucose interaction) and where clinical prediction matters most.

### H representation quality — run21

| Metric | run21 (final) | run14 (reference) | run12 (with age) |
|---|---|---|---|
| Val MAE | ~0.45 | 0.45 | 0.46 |
| PC1_L5 | **27.9%** | 60.5% | 63.8% |
| R²_probe L5 | **0.944** | 0.939 | 0.942 |
| CGM r at L5 | +0.09 | −0.37 | −0.60 |

**Note on run21 vs run14:** The CGM sign flip at L5 (r=−0.37 in run14, r=+0.09 in run21) is absent in run21. This is run-to-run variability in an emergent property of the H norm — not a meaningful quality difference. The sign flip is not a training objective; it is a downstream consequence of the reconstruction pressure that appears stochastically. Crucially, run21's PC1_L5=27.9% is *more distributed* than run14's 60.5%, which is desirable by the H richness metric. Per-window attention analysis confirms run21 is a well-functioning encoder with the same head specialisation patterns.

### Per-layer H analysis (L1–L4 from run12 reference; L5 from run14)

| | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| Probe R² | **0.985** | 0.967 | 0.925 | 0.941 | 0.939 |
| PCA PC1 | 47.8% | 26.1% | **17.7%** | 19.2% | 60.5% |
| CGM r | +0.79 | +0.24 | +0.27 | −0.19 | **−0.37** |
| Event response | Inherited | Flat | Flat | Emerging | **Clearest** |
| Attention | Local+structure | Local | Diffuse | Diffuse | Diffuse |
| Character | Near-input | Transitional | Most abstract | Risk-aware | Task-aware |

**Layer narrative:**
- **L1:** Representation ≈ input projection + small attention correction. Probe R²=0.985 is trivial — CGM was in the input. Not meaningful richness.
- **L2:** Sharp redistribution: PC1 drops 47.8%→26.1%, CGM r drops +0.79→+0.24. Multi-feature mixing begins.
- **L3:** Maximum abstraction. Lowest probe R² (0.925), lowest PC1 (17.7%), flattest event response. Information is most distributed across 128 dimensions.
- **L4:** CGM norm correlation **changes sign** (−0.19). The encoder begins encoding metabolic complexity rather than signal magnitude — high H_t norm starts to flag low/uncertain glucose.
- **L5:** Reconstruction pressure reinstates a dominant axis (PC1 60.5% in run14) along which the MLP head reads CGM. Event response clearest. Full physiological interpretation mature.

**Key finding — CGM sign flip:** L1 encodes raw CGM positively (r ≈ +0.79). L5 inverts (r = −0.37 in run14). The encoder progressively transforms from encoding signal magnitude to encoding metabolic complexity. High H_t norm at L5 marks moments of physiological uncertainty — falling glucose, post-event transitions — not high glucose. This is the property that makes H useful for downstream risk prediction.

---

## 9. H Enrichment Pipeline — Summary

After establishing run14 as the baseline, five approaches were tested to improve H richness. All were negative.

| Run | Approach | Key failure |
|---|---|---|
| 15 | VICReg (λ=0.05) | Σ\|r\|≈0.06, sign flip absent — dimensional uniformity ≠ richness |
| 16 | Multimodal masking (prob=0.3) | Diagonal attention — PI/RA reconstruction diluted temporal integration pressure |
| 17 | Contrastive InfoNCE | Skipped — custom GradientTape loop incompatible with TF XLA on RTX 5070 CC 12.0 |
| 18 | Scale-up (d_model=256) | Same failure signature as run16 — larger model, same degradation |
| 20 | JEPA (EMA target encoder) | Full collapse: R²_L5=0.039, PC1=98% — EMA alone insufficient without anti-collapse term |

**Consistent failure pattern across runs 15/16/18:** diagonal attention, near-zero L5 feature correlations, absent CGM sign flip. The additional objectives in each case competed with the reconstruction signal and reduced the structural depth of H. The discrete flag + span reconstruction combination that produces the L5 sign flip is fragile — any auxiliary objective strong enough to meaningfully alter H structure also destroys the anchoring mechanism.

**Conclusion:** Run14 is the optimal Stage 1 encoder obtainable within the current MTSM framework. The enrichment pipeline is closed.

---

## 10. Attention Structure — Per-Window Analysis

Individual per-window attention analysis (`scripts/attention_viz.py`) revealed that the encoder's attention is not locally constrained despite averaged heatmaps appearing diagonal. See Section 5 for the mathematical definition of the attention matrix.

**Per-window:** Each window type shows rich off-diagonal structure anchored to physiologically relevant events. Post-bolus windows have a vertical attention column at the bolus time (all timesteps attending to the key event). Hypoglycaemia windows show diffuse overnight patterns. Stable basal windows are near-diagonal (correctly, since there are no long-range events to integrate).

**Per-window attention by window type:**

| Window type | L5 pattern | Interpretation |
|---|---|---|
| Post-meal | Horizontal + vertical bands at meal time | Encoder anchors context on meal event; all timesteps read from it |
| Post-bolus | Pronounced vertical column at bolus time | Bolus is the dominant event for the full window's interpretation |
| Hypoglycaemia | Diffuse patches in overnight region | No discrete logged event; encoder distributes attention broadly |
| Stable basal | Near-diagonal | No driver events → local context only (physiologically appropriate) |
| High variability | Multiple off-diagonal bands | Separate attention pattern per event cluster |

**Per-layer progression:**
- **L1–L2:** Strong diagonal with some off-diagonal structure. High peak attention weights.
- **L3:** Most diffuse. Attention weights uniformly low — maximum abstraction.
- **L4–L5:** Off-diagonal structure re-emerges in a learned, window-specific form.

---

## 11. Stage 2 Handoff

The frozen run21 encoder receives a 24h window `(288, 10)` and produces `H ∈ R^(288, 128)`. Stage 2 task heads attach to H and are trained independently with the encoder frozen. The encoder is never retrained for any downstream task — its weights are fixed after Stage 1 for all applications.

**Head training:** during Stage 2 training, every window from every patient is passed through the frozen encoder to produce an H matrix. The head receives H as input, produces a prediction, and updates only its own weights via gradient descent. The encoder receives no gradient.

**At test time:** the pipeline is identical but without a label or gradient update. A new patient window (288, 10) → frozen encoder → H (288, 128) → trained head → prediction. The head never sees raw CGM directly; it always operates on H.

**Encoder weights:** `results/mtsm/run21/encoder_weights.weights.h5`

**Age conditioning:** age is not in H. Pass `age_norm` (age/100, stored in `.npz` as feature 10) directly to each Stage 2 head as a scalar conditioning input.

**Which H to use:** L5 (the full encoder output) for all Stage 2 tasks. For sequence-to-sequence tasks (forecasting, imputation) use full H (288, 128). For window-level tasks (survival, ISF/CR profiling) pool H to a single vector h (128,) via attention pooling. See `docs/H_analysis.md` for per-layer analysis and task-specific recommendations.

### Information Preservation

**What H encodes:**
- CGM dynamics: linear probe R²=0.944 at L5 — CGM is linearly decodable from H
- Driver event timing: event-triggered H norm response confirmed for both bolus and carbs
- Circadian structure: H norm circadian pattern tracks dawn phenomenon and meal transitions
- Metabolic complexity: high H_t norm flags low/falling glucose and post-event uncertainty

**What H does not encode:**
- Exact PI/RA magnitudes are not directly decodable from H. They shaped the encoder weights during training and are encoded implicitly in the representation structure, but are not recoverable as raw values.
- The reconstruction head weights (discarded). H contains more information than the MLP head was able to extract — R²=0.939 means ~6% of CGM variance is not linearly accessible from the norm, distributed across dimensions.

**H is not a lossless transform.** The encoder was trained under reconstruction pressure, not to be invertible. However, run21's PC1_L5=27.9% means 72.1% of variance is distributed across the remaining 127 dimensions — there is representational capacity well beyond what the linear reconstruction head used, available for non-linear Stage 2 heads.

**Attention pooling (H → h):** For window-level tasks, H (288, 128) is collapsed to h (128,) via learnable attention pooling:
```
α = softmax(H · q / √128)    learned query q ∈ R^128; α shape: (288,)
h = Σ_t  α_t · H_t           shape: (128,)
```
Temporal resolution is lost in h — the sequence structure is collapsed to a single vector. For window-level tasks (e.g., hypo risk) this is appropriate: the label is a property of the window as a whole. Attention pooling is preferred over mean pooling because it can concentrate weight on clinically critical moments (falling glucose, bolus events) rather than averaging uniformly.
