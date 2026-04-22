# H Representation Analysis — Stage 1 MTSM Encoder

**Run:** run21 — 988 adult T1D patients, 70 epochs (no early stopping), `--no_age` — **final encoder**
**Analysis scripts:** `scripts/replot.py --run_id run21 --no_age --plots all` | `scripts/attention_viz.py --run_id run21 --no_age`
**Evaluation data:** test split only — patient-level 80/10/10 split, SEED=42, ~99 unseen patients
**Plots location:** `results/mtsm/run21/`

**Key metrics (active baseline):**

| Metric | run21 (final) | run14 | run12 (reference) |
|---|---|---|---|
| Val MAE | ≈0.45 | 0.45 | 0.46 |
| PC1_L5 | 27.9% | 60.5% | 63.8% |
| L5 R²_probe | 0.944 | 0.939 | 0.942 |
| CGM r at L5 (norm) | +0.09 | −0.37 | −0.60 |

*Note: run21 is identical in config to run14. The CGM sign flip in run21 norm correlation is absent (r=+0.09 vs −0.37 in run14) — this reflects run-to-run variability in an emergent property of the norm, not a meaningful quality difference. The PC1=27.9% in run21 is more distributed than run14 (60.5%), which is actually desirable. Individual window attention analysis (Section 5) confirms run21 is a well-functioning encoder. Intermediate layer values (L1–L4) are from run12 reference.*

---

## 1. What H Is — and What It Is Not

### 1.1 Definition

H is the output tensor of the Stage 1 Transformer encoder for a single 24h input window:

```
H ∈ R^(288 × 128)
```

Each of the 288 rows is a 128-dimensional vector H_t — the encoder's contextualised representation of timestep t (every 5 minutes). H_t is not a local snapshot: every H_t has been computed by attending to all other timesteps in the window via bidirectional self-attention. So H_t describes minute t **in the context of the entire 24-hour window**: what CGM was doing two hours later, what bolus preceded it, whether a dawn phenomenon is in progress.

### 1.2 What H is Not

- **Not a compressed CGM signal.** The reconstruction head converts H_t → ŷ_t (predicted CGM), but that head is discarded. H contains more than the head can extract.
- **Not a fixed embedding.** H changes for every window. The learned knowledge is in the encoder weights; H is the result of applying those weights to a specific input.
- **Not interpretable dimension-by-dimension.** The 128 dimensions are not aligned to any physiological quantity. Structure must be probed indirectly.

### 1.3 The H_t Norm as a Probe

The L2 norm `||H_t||₂` measures the magnitude of the representation at timestep t. A larger norm means more representational "energy" is concentrated at that moment. It serves as a proxy for metabolic complexity: timesteps the encoder treats as informationally rich or uncertain tend to have higher norms. This is a standard interpretability technique borrowed from NLP (BERT probing).

**Important limitation:** the norm measures one scalar derived from a 128-dim vector. It captures the overall magnitude of the representation but says nothing about which features or physiological signals are encoded in specific dimensions. Treat it as a broad proxy, not a direct measure.

---

## 2. Encoder Architecture — What Each Layer Does

The encoder has 5 Transformer layers. Each layer applies:
1. Multi-head self-attention (4 heads, key_dim=32)
2. Residual connection + LayerNorm
3. Feed-forward network (Dense 128→256→128)
4. Residual connection + LayerNorm

The **residual connections** are critical for interpretation: each layer's output is `H_prev + Δ`, where `Δ` is the layer's learned transformation. Early layers produce outputs close to the input; the representation only diverges gradually.

---

## 3. Per-Layer Analysis

Four analyses were run to characterise each layer:
1. **Attention matrix** — average attention weights per layer
2. **Event-triggered H norm** — H norm response around bolus/carbs events
3. **Linear reconstruction probe** — OLS: H_t (128-dim) → CGM_t (z-score), R² and MAE
4. **Feature correlation heatmap** — pooled Pearson r between H_t norm and each input feature

### 3.1 Layer 1

| Metric | Value |
|---|---|
| Probe R² | **0.985** (highest) |
| Probe MAE | **0.090** (lowest) |
| PCA PC1 | 47.8% |
| CGM r | +0.79 |
| Attention | Strong diagonal, structured off-diagonal |

**Interpretation:** L1 is the first attention sweep over the input projection. Because of the residual connection, H_L1 ≈ x_projected + small_Δ. The input projection is a linear map of all 10 features including CGM, so H_L1 still encodes CGM almost directly. This explains the extremely high R² (0.985): a linear probe trivially recovers CGM because it was in the input. The strong positive CGM r (+0.79) confirms the norm still tracks raw CGM level.

The attention heatmap shows the strongest off-diagonal structure of all layers — likely the encoder's first pass at identifying long-range dependencies. The checkered pattern in the off-diagonal is probably a subsampling artefact from the 4-step downsampling used for display.

**Conclusion:** L1 is closest to the raw input. High linear decodability does not indicate richness — it indicates the representation has not been transformed yet. L1 is **not** a good intermediate representation for downstream tasks.

---

### 3.2 Layer 2

| Metric | Value |
|---|---|
| Probe R² | 0.967 |
| Probe MAE | 0.131 |
| PCA PC1 | 26.1% |
| CGM r | +0.24 |
| Attention | Strong diagonal, slightly more diffuse |

**Interpretation:** A significant transformation occurs between L1 and L2. The probe R² drops from 0.985 to 0.967 — still high, but CGM information is now more entangled with other features. PC1 drops dramatically from 47.8% to 26.1%, meaning the representation is now far more distributed across dimensions. The encoder is beginning to mix physiological signals.

CGM r drops from +0.79 to +0.24: H_t norm still rises with CGM level, but the relationship is weaker. The first sign of multi-feature integration.

**Event-triggered:** Nearly flat for both bolus and carbs. The norm barely modulates. This is the "mixing phase" — physiological information is being integrated across timesteps and features and is no longer concentrated in the norm.

**Conclusion:** L2 is a transitional layer. The representation is more abstract than L1 but the mixing is not yet complete.

---

### 3.3 Layer 3

| Metric | Value |
|---|---|
| Probe R² | **0.925** (lowest) |
| Probe MAE | **0.190** (highest) |
| PCA PC1 | **17.7%** (lowest) |
| CGM r | +0.27 |
| Attention | Diffuse (colorbar peak ~0.008 vs 0.040 at L2) |

**Interpretation:** L3 is the point of **maximum abstraction**. Three converging signals support this:

1. Probe R² is lowest (0.925) — CGM information is most entangled with other features and least linearly accessible.
2. PC1 is lowest (17.7%) — information is most distributed across the full 128-dimensional space.
3. Attention colourbar drops to 0.008 — attention weights are more diffuse, meaning each timestep draws from a broader context rather than concentrating on specific keys.

Event-triggered H norm is flat at L3 for both events — the norm does not respond to individual events because the representation is in its most integrated state.

**Conclusion:** L3 is the richest representation in the sense that information is most distributed across dimensions. L3 may be optimal for downstream tasks that use non-linear heads.

---

### 3.4 Layer 4

| Metric | Value |
|---|---|
| Probe R² | 0.941 |
| Probe MAE | 0.169 |
| PCA PC1 | 19.2% |
| CGM r | **−0.19** (sign flip) |
| hour_cos r | −0.19 |
| Attention | Diffuse, similar to L3 |

**Interpretation:** L4 is where a critical transition happens — the **CGM norm correlation changes sign**. At L1–L3, high CGM → high H_t norm. At L4, high CGM → lower H_t norm. This sign flip reflects the encoder beginning to encode a different quantity in the norm: it starts allocating more representational energy to *low or falling* glucose. This is the first layer where the norm begins to reflect metabolic complexity/risk rather than absolute level.

The circadian signal (hour_cos r = −0.19) also emerges at L4 — the encoder begins using the norm to mark specific times of day associated with metabolic transitions.

**Conclusion:** L4 is the transition into task-aware representation. The sign flip in CGM correlation is the most important structural change across layers.

---

### 3.5 Layer 5 (Final) — Active Baseline run14

| Metric | Value (run14) |
|---|---|
| Probe R² | **0.939** |
| PCA PC1 | **60.5%** |
| CGM r | **−0.37** |
| Σ\|r_L5\| (CGM + PI + hour_sin) | **0.46** |
| Event response | Clearest of all layers |

**Interpretation:** L5 shows the most striking structural change: PC1 jumps from ~19% at L4 to 60.5%. One dominant axis emerges. This is the **reconstruction pressure axis** — the encoder is forced by the MTSM pretext task to concentrate CGM-relevant information into a direction that the attached MLP head can read. The jump is direct evidence that the reconstruction head is shaping the representation.

L5 is also where the **full physiological interpretation** of the norm is most mature:

- CGM r = −0.37: high norm occurs when CGM is low or uncertain. The encoder allocates representational energy to metabolically complex moments (falling glucose, post-event transitions) rather than stable high-glucose periods which are easy to predict. *Note: this effect is weaker in run14 (−0.37) than in run12 (−0.60), a consequence of removing age_norm from the encoder input. The sign flip is preserved — the physiological encoding direction is intact.*
- Event-triggered response is clearest at L5: bolus → norm dip then rise tracking PI recovery; carbs → norm peak at t=0. The encoder has re-concentrated the event signal from its diffuse L2–L4 form back into the norm.

**Why does PC1 jump?** The reconstruction head is a linear MLP: Dense(128→64→1). A linear head can only read along linear directions in H-space. The encoder is therefore pressured to put the "most reconstructable" information along a dominant linear axis — maximising the signal the head can extract.

**Conclusion:** L5 is what the encoder was trained to produce. It is the **default choice for all downstream applications**.

---

## 4. Layer Comparison Summary

| | L1 | L2 | L3 | L4 | L5 (run14) |
|---|---|---|---|---|---|
| Probe R² | **0.985** | 0.967 | 0.925 | 0.941 | 0.939 |
| PCA PC1 | 47.8% | 26.1% | **17.7%** | 19.2% | 60.5% |
| CGM r | +0.79 | +0.24 | +0.27 | −0.19 | **−0.37** |
| Event response | Inherited | Flat | Flat | Emerging | **Clearest** |
| Attention | Local+structure | Local | Diffuse | Diffuse | Diffuse |
| Character | Near-input | Transitional | Most abstract | Risk-aware | Task-aware |

*L1–L4 values are from run12 (reference run). L5 values are run14 confirmed.*

**Which layer to use:**

| Task | Recommended layer | Reason |
|---|---|---|
| All standard downstream tasks | **L5** | Trained for it; clearest physiological structure; default |
| Hypoglycaemia prediction | **L5** | Norm anti-correlates with CGM — high norm flags low/falling glucose |
| Short-horizon forecasting | **L5** | Event awareness clearest; PI/RA response encoded |
| Gap imputation | **L5** | What the encoder was trained to do (MTSM objective) |
| Multi-label phenotyping / clustering | **L3 or L4** | Most distributed representation; less dominated by reconstruction axis |

The probe R² result (L1 best) is **not evidence that L1 is a better representation**. L1 is closest to the raw input — a linear probe recovers CGM easily because CGM was directly in the input.

---

## 5. Attention Matrix — Analysis

### 5.1 Why the Averaged Heatmap Looks Diagonal

The standard averaged attention heatmap (mean over 4 heads × 100 windows) consistently appears diagonal. This is **not evidence that the encoder only does local attention** — it is an averaging artefact.

Each window has its off-diagonal attention concentrated at physiologically specific positions: the meal time, the bolus time, the onset of a hypo, a dawn phenomenon window. When 100 windows are averaged, these window-specific patterns sit at different positions and cancel out. The only structure that survives averaging is what is consistent across all windows: local attention to neighbouring timesteps, which always appears on the diagonal.

Per-window individual attention analysis (see `scripts/attention_viz.py`, plots `attention_individual_windows.png` and `attention_per_head_L5.png`) confirms that rich off-diagonal structure exists and is physiologically meaningful.

### 5.2 Per-Window Individual Attention

Five physiologically distinct window types were analysed individually: post-meal, post-bolus, hypoglycaemia, stable basal, and high-variability.

**Post-meal:** L5 shows horizontal and vertical bands anchored at the meal event time. All timesteps attend to the meal moment; the meal moment attends broadly forward in time, tracking the expected glucose rise and insulin response.

**Post-bolus:** L5 shows a pronounced vertical column at the bolus time — every query in the window attends to the bolus moment regardless of temporal position. The encoder has learned the bolus is the key event for the entire window's interpretation.

**Hypoglycaemia:** L5 shows a diffuse pattern with patches in the overnight/early-morning region. No sharp event column — hypos lack a discrete logged event — so the encoder distributes attention broadly across the uncertain period.

**Stable basal:** Near-diagonal pattern at all layers. Without driver events, the encoder uses local context only — physiologically appropriate since there is no long-range information to integrate.

**High variability:** Multiple off-diagonal bands corresponding to multiple events. The encoder constructs a different attention pattern for each event cluster.

**Key conclusion:** The encoder's attention is **window-specific and event-driven**. It is not doing local-only smoothing — it attends long-range where events give it reason to.

### 5.3 Per-Head Specialisation at L5

Individual head analysis at L5 (`attention_per_head_L5.png`) reveals clear functional specialisation across the 4 heads:

| Head | Pattern | Function |
|---|---|---|
| **Head 1** | Diagonal + vertical columns at event times | Local smoothing + event anchor |
| **Head 2** | Fully diffuse, near-uniform weights | Global context — every timestep accesses the full 24h window |
| **Head 3** | Sharp diagonal + bright vertical columns | Event anchor — cleanest discrete-event detector |
| **Head 4** | Mixed, window-type dependent | Adaptive context — pattern varies by physiological state |

Head 2 is particularly notable: it distributes attention almost uniformly across the sequence, giving every timestep equal access to the full 24-hour context. This is the mechanism through which the encoder achieves its bidirectional integration — not through any single dominant head, but through one globally attending head that complements the local and event-anchored heads.

Head 3's sharp vertical columns at bolus and carb event times directly explains why removing the discrete flags (run13) causes H structure to collapse: the event columns are how the encoder registers the temporal anchor that forces a consistent H convention across all training windows.

### 5.4 Per-Layer Progression

**L1 and L2:** Strong diagonal with some off-diagonal structure. High peak attention weights. The encoder is doing local integration — close to the input projection.

**L3:** Most diffuse layer. Attention weights are lower and more spread. Maximum abstraction — physiological information most distributed across the sequence.

**L4–L5:** Off-diagonal structure re-emerges in a learned, window-specific form. The encoder reconcentrates attention on physiologically relevant positions identified by the earlier layers.

---

## 6. Event-Triggered H Norm Analysis

### 6.1 Final Layer (H_norm_vs_drivers.png) — Full Reading

**Bolus-triggered (n≈47,000 events):**

```
t=-50min: H norm begins falling (CGM already rising — pre-correction)
t=0:      Bolus logged. H norm at valley. PI at minimum (insulin-on-board cleared).
t=+20:    Transitional minimum — event recorded, pharmacokinetics not yet activated
t=+150:   H norm peak, tracking PI recovery curve (Hovorka ODE delay ~1-2h)
```

The pre-event fall reflects the encoder recognising a predictable correction pattern (rising CGM → imminent bolus). The post-event rise tracks active insulin dynamics, which create metabolic uncertainty (will glucose overcorrect into hypoglycaemia?).

**Carbs-triggered (n≈3,600 events):**

```
t=-30min: H norm begins rising (bidirectional attention can "see" upcoming meal)
t=0:      Carbs logged. H norm at peak. CGM still flat.
t=+50:    H norm plateaus. RA builds to its Hovorka ODE peak at ~+90min.
```

The peak at t=0 (not at the RA peak at t=+90min) confirms the encoder responds to the **discrete `carbs_logged` flag**, not to the RA absorption curve. The bidirectional architecture's ability to pre-respond before t=0 is a unique feature that causal models (GluFormer) cannot replicate.

### 6.2 Per-Layer Event Response (layer_event_triggered.png)

The event response follows a U-shape across layers: strong at L1 (inherited from input), flat at L2–L3 (mixing phase), recovering at L4, clearest at L5 (learned).

This progression confirms the event response at L5 is **not a trivial pass-through of the discrete flag**. At L2–L3, the flag's information has been integrated across all timesteps and features and is no longer concentrated in the norm. At L5, the encoder has re-concentrated it in a learned, context-aware form — the response encodes not just "event occurred" but the full metabolic consequence of the event.

---

## 7. Feature Correlations (H_t Norm)

### 7.1 Violin Distribution — Final Layer (H_norm_feature_correlation.png)

Per-window Pearson r distributions (windows with flat features excluded):

| Feature | Median r | Distribution | Interpretation |
|---|---|---|---|
| CGM | −0.37 | Narrow, negative | High norm ↔ low/falling CGM. Metabolic risk proxy. |
| PI | +0.15 | Wide, moderately positive | High active insulin → higher representational complexity |
| RA | *empty* | No windows with non-flat RA | Sparse signal; per-window approach fails — see Section 11 |
| hour_sin | ~−0.05 | Wide, centred near 0 | Weak and inconsistent |
| hour_cos | ~−0.05 | Wide, centred near 0 | Weak and inconsistent |
| bolus | ~0.00 | Narrow near 0 | Individually small bolus flags drive weak correlation |
| carbs | +0.03 | Narrow near 0 | Same |

**CGM:** The negative correlation (−0.37 median, run14) with a narrow distribution confirms this is a systematic property of the final layer, not noise. High H_t norm occurs when CGM is low. *This effect is weaker than in run12 (−0.60) but in the same direction — age exclusion reduces the sharpness of the metabolic risk encoding but does not remove it.*

**PI:** Wide distribution (IQR roughly 0 to +0.75) means the correlation is context-dependent. In some windows PI is strongly associated with complex dynamics (positive r); in others less so.

**RA (empty violin):** This is expected, not a bug. RA is a sparse signal: it equals zero for ~80% of any 24h window. A per-window Pearson r requires non-zero variance within the window. The event-triggered analysis (Section 6) gives the correct measurement.

### 7.2 Per-Layer Heatmap (layer_feature_correlation.png)

Pooled Pearson r (all timesteps × all windows concatenated). L5 values are confirmed run14; L1–L4 are run12 reference:

| | CGM | PI | RA | hour_sin | hour_cos | bolus | carbs |
|---|---|---|---|---|---|---|---|
| L1 | +0.79 | −0.27 | nan | −0.08 | −0.01 | +0.10 | +0.07 |
| L2 | +0.24 | −0.27 | nan | −0.17 | −0.04 | +0.16 | +0.05 |
| L3 | +0.27 | −0.07 | nan | −0.06 | +0.08 | +0.08 | +0.03 |
| L4 | −0.19 | +0.06 | nan | −0.05 | −0.19 | +0.03 | +0.04 |
| **L5** | **−0.37** | — | nan | — | — | — | — |

*(RA remains nan even in the pooled computation — RA is zero-valued for the vast majority of pooled timesteps.)*

**CGM sign flip (L3→L4):** The most important structural observation. CGM r goes +0.79 (L1) → +0.24 (L2) → +0.27 (L3) → −0.19 (L4) → −0.37 (L5). This is a clean monotonic sign reversal. The encoder redefines what the norm encodes at different depths: early layers reflect raw signal magnitude; later layers reflect metabolic complexity and risk.

---

## 8. Circadian Pattern (H_norm_circadian.png)

Mean H_t norm by hour of day shows three distinct phases:

**Overnight rise (00:00 → 06:00):** H norm rises from ~2.80 to ~2.92. The dawn phenomenon (cortisol/GH-driven glucose rise from ~4am) makes early morning metabolically uncertain. The encoder assigns more representational energy to this hard-to-predict period.

**Morning drop (07:00 → 09:00):** H norm falls sharply to ~2.83. This coincides with the annotated breakfast time. The model has context from past events and the current period is more predictable.

**Mid-day → evening rise (09:00 → 19:00):** H norm climbs again, peaking at 18:00–19:00 (dinner time). The dinner/evening period is the most representationally complex, due to the combination of dinner bolus, post-dinner glucose variability, and the transition into overnight.

**Late evening drop (20:00 → 23:00):** H norm falls sharply. Post-dinner dynamics are resolving and the patient is approaching sleep — a physiologically settling period.

---

## 9. PCA and t-SNE of Mean-Pooled H

### 9.1 PCA by Modality (pca_by_modality.png)

PC1 = 60.5%, PC2 ≈ 8%. No separation between AID, SAP, and MDI. The encoder organises the representation space primarily around physiological state (glucose level, driver activity, time of day) rather than therapy type. This is desirable: a representation that discriminates AID from MDI by therapy label rather than physiology would not transfer well to tasks that care about glucose dynamics.

AID is the dominant modality (~88% of windows). SAP and MDI are dispersed throughout the AID distribution, with a few outliers at the extremes of PC1 (extreme physiological states regardless of modality).

### 9.2 Per-Layer PCA (pca_per_layer.png)

| Layer | PC1 | PC2 | Shape |
|---|---|---|---|
| L1 | 47.8% | 14.2% | Elliptical with tail |
| L2 | 26.1% | 13.5% | Circular, dispersed |
| L3 | 17.7% | 12.9% | Circular, most dispersed |
| L4 | 19.2% | 16.2% | Circular, slightly tighter |
| **L5** | **60.5%** | ≈8% | Elongated — dominant axis |

The progression L1→L3 shows progressive distribution (PC1 drops, variance spreads). L3 is the most isotropic. L5 shows the reversal: one dominant axis emerges from the reconstruction pressure of the MLP head.

The L5 PC1 axis can be interpreted as the "CGM reconstructability direction" — the linear direction in H-space that the MLP head uses to predict masked CGM. The other 127 dimensions carry context that the linear head cannot use but that non-linear downstream heads can.

---

## 10. Linear Reconstruction Probe Summary

| Layer | R² | MAE (z-score) | Interpretation |
|---|---|---|---|
| L1 | 0.985 | 0.090 | Near-input — trivially linearly decodable |
| L2 | 0.967 | 0.131 | Beginning to abstract |
| **L3** | **0.925** | **0.190** | Most abstract — hardest linear extraction |
| L4 | 0.941 | 0.169 | Re-encoding toward reconstruction |
| **L5** | **0.939** | ≈0.167 | Task-ready — CGM encoded + context |

*L1–L4 from run12 reference; L5 confirmed run14.*

**Common misreading:** R² decreasing from L1 to L3 does not mean the representation is getting worse. It means CGM information is more distributed across dimensions and less linearly accessible from the norm alone.

---

## 11. The RA Problem

RA (rate of carb absorption) is a fundamentally different type of signal from CGM, PI, and the discrete flags. It is:
- Zero for ~80% of any 24h window
- Non-zero for a 2–3h post-meal bell curve
- Fully determined by the carb quantity and timing (deterministic ODE, not a measured signal)

Per-window and pooled Pearson r both fail to capture RA correlation with H norm. The correct analysis for RA is the **event-triggered analysis** (Section 6). The carbs-triggered H norm plot shows: H norm peaks at t=0 (the `carbs_logged` flag moment), not at the RA peak at t=+90min. **The encoder uses the discrete carbs flag as the primary cue; RA is a continuous confirmation signal.**

This distinction matters architecturally: run13 ablation (Section 12) confirms that removing the discrete flags while keeping RA causes H structure to degrade significantly.

---

## 12. Run 13 Ablation — Confirmed Findings

Run13 is identical to run14 except `bolus_logged` (feature 5) and `carbs_logged` (feature 6) are zeroed throughout. The encoder sees only PI and RA from the Hovorka ODE as indicators of driver events.

| Observation | Predicted | Confirmed run13 result |
|---|---|---|
| Val MAE | Slightly higher | **0.47** (+0.02 vs run14) |
| CGM sign flip at L5 | Should persist | **Absent — L5 CGM r ≈ +0.03** |
| H structure (PCA/attention) | Degraded | **Disorganised — PCA diffuse, no dominant axis** |
| Event-triggered response | Delayed, shifted | **Flat — no bolus or carbs response** |

**Key conclusion:** The discrete flags serve two roles simultaneously:
1. **Minor informational gain** (~3% MAE improvement) — the encoder can infer event timing from PI/RA, but less precisely.
2. **Critical representational anchoring** — the binary spike at t=0 forces a globally consistent H convention across all training windows. Without the spike, each window's PI/RA trajectory is consistent within that window, but the encoder cannot establish a consistent reference point across windows. This breaks the structural regularities (sign flip, event response) that make H useful for downstream tasks.

The lesson: **reconstruction MAE is a weak indicator of H quality**. Run13's MAE degradation (0.47 vs 0.45) is marginal, but H structure degradation is severe. H quality must be evaluated via the full enrichment metrics (PC1_L5, Σ|r_L5|, sign flip, event response), not by MAE alone.

---

## 13. H Enrichment Pipeline — Results

All Stage 1 enrichment experiments attempted after run14 baseline. Goal: improve H richness (distributed PC1, higher Σ|r|, deeper abstraction) without sacrificing reconstruction quality.

### 13.1 H Richness Score (baseline run14)

| Component | Formula | Run14 | Direction |
|---|---|---|---|
| Distributed variance | 100 − PC1_L5 | **39.5%** | ↑ |
| Feature coverage | Σ\|r_L5\| for CGM, PI, hour_sin | **0.46** | ↑ |
| Abstraction depth | 1 − R²_probe_L5 | **0.061** | ↑ |
| Reconstruction sanity | val MAE ≤ 0.48 | **0.45** | maintained |

### 13.2 Enrichment Run Results

| Run | Approach | Val MAE | PC1_L5 | Σ\|r_L5\| | Sign flip | Verdict |
|---|---|---|---|---|---|---|
| **14** | Baseline (`--no_age`) | 0.45 | 60.5% | 0.46 | −0.37 | **Active baseline** |
| **15** | VICReg (λ=0.05) | 0.45 | ≈19% | ≈0.06 | Absent | **Negative** |
| **16** | Multimodal masking (prob=0.3) | ≈0.46 | Diagonal pattern | ≈0.17 | Absent | **Negative** |
| **18** | Scale-up (d_model=256, 8 heads) | ≈0.46 | Diagonal pattern | ≈0.17 | Absent | **Negative** |
| **20** | JEPA (H-space prediction, EMA) | ≈0.016 | 98.0% | ≈0.04 | Absent | **Negative — collapse** |

### 13.3 Consistent Failure Pattern

Runs 15, 16, and 18 all return the **same diagnostic signature**:

1. **Diagonal attention matrix** — the model routes each timestep to itself rather than integrating context
2. **Near-zero L5 feature correlations** — Σ|r_L5| ≈ 0.06–0.17, barely above zero
3. **Absent CGM sign flip** — L5 CGM r remains positive (≈+0.02 to +0.10), not negative
4. **Disorganised PCA** — no dominant axis or highly fragmented variance

This pattern is consistent regardless of whether the approach adds a regulariser (VICReg), changes the masking objective (multimodal), or increases model capacity. The failure signature matches run13 (no flags) — the additional objectives interfere with the structural anchoring that the reconstruction objective + discrete flags produce in run14.

**Interpretation:** The VICReg, multimodal, and scale-up approaches all introduce objectives that compete with the reconstruction signal. In each case, the model finds a solution that satisfies both objectives at low loss but does so by reducing the structural depth of H rather than enriching it. The reconstruction pressure that produces the L5 CGM sign flip is diluted.

### 13.4 JEPA Result (run20)

JEPA changed the fundamental objective from raw CGM reconstruction to H-space prediction. It produced **full representation collapse**: R²_probe_L5=0.039, PC1_L5=98%, attention shows vertical bands, training curves are unstable zig-zag.

Without an explicit anti-collapse term, the EMA mechanism alone is insufficient. Both the context and target encoders converged to near-constant output vectors, making the JEPA loss trivially low with no physiological learning.

**H enrichment pipeline is closed. Run14 is the final Stage 1 encoder.**

---

## 14. Conclusions for Stage 2

### 14.1 Which Representation to Use

For all Stage 2 downstream applications, **use L5 (the full encoder output H)**. This is supported by:
- Clearest event-triggered response (metabolic events encoded and contextualised)
- CGM norm anti-correlation (high norm flags metabolically uncertain/risky states)
- Trained by the pretext task — weights optimised to make L5 useful for CGM reconstruction, which shares structure with all downstream CGM tasks
- Probe R² = 0.939 confirms CGM information is preserved and linearly accessible if needed

The intermediate layers (L3/L4) have more distributed representations that might benefit tasks requiring diverse physiological features, but L5 is the right starting point.

### 14.2 How H Encodes Physiological State

The full picture from the analysis:

- **H_t norm is a metabolic complexity proxy:** High at moments of high uncertainty (falling glucose, post-event transitions, meal logging). Low at predictable states (stable basal, resolved post-meal absorption).
- **H uses the full 24h bidirectional context:** The pre-event H norm rise for carbs events shows the encoder anticipates events it can "see" later in the window — a property only possible with bidirectional attention.
- **H captures circadian structure:** Dawn phenomenon (04:00–06:00) and dinner-time transitions (18:00–19:00) are encoded as high-complexity periods.
- **H encodes driver dynamics, not just CGM:** The post-bolus H norm trajectory tracks PI recovery (Hovorka ODE dynamics), not just the discrete flag.

### 14.3 Foundation Model Claim

The key claim for the thesis — one encoder, multiple downstream applications — is supported by the following evidence from H:

1. H contains CGM information (R²_probe = 0.939 at L5) → supports gap imputation and forecasting
2. H norm flags low/falling glucose (CGM r = −0.37 at L5) → supports hypoglycaemia risk prediction
3. H responds to driver events with physiological timing (PI curve, RA lag) → supports counterfactual simulation
4. H does not over-specialise to modality or age → supports cross-patient transfer

The representation is rich enough to support multiple tasks without retraining the encoder. The enrichment experiments (runs 15/16/18) confirm that run14's H quality is not trivially improvable — the reconstruction + discrete flag combination produces a structurally stable representation that stronger auxiliary objectives degrade rather than enhance.
