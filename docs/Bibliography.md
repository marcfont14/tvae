
# Bibliography & Literature Notes

## Glucose Foundation Model — T1D Multi-Task Foundation Model Thesis

This document collects all references used or discussed during the project, organised by topic, with a note on relevance for each entry. Intended as a handoff document for the thesis writing phase.

---

## 1. Foundational Architecture

### 1.1 Transformer

**Vaswani, A. et al. (2017). _Attention Is All You Need_. NeurIPS.**

- The original Transformer paper. Defines the encoder-decoder architecture, multi-head self-attention, scaled dot-product attention, and sinusoidal positional encoding.
- **Relevance:** Direct basis for our encoder architecture. Cite when introducing the Transformer encoder, MHSA, and positional encoding.

**Alammar, J. (2018). _The Illustrated Transformer_. [blog]**

- Visual walkthrough of the Transformer architecture. Not a paper but widely cited.
- **Relevance:** Useful reference if the thesis includes a visual explanation of attention for a non-specialist reader. Do not cite in the main text — use as background.

---

## 2. Transformers for Time Series

**Lim, B. et al. (2021). _Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting_. International Journal of Forecasting.**

- TFT: the most influential Transformer for multivariate time series forecasting. Introduces variable selection networks, gated residual networks, and interpretable multi-head attention (shared values across heads for direct interpretability of attention weights).
- **Relevance:** Main reference architecture. Our model is simpler but conceptually related. Cite when justifying the Transformer encoder choice for multivariate time series with heterogeneous inputs (CGM + drivers + modality).

**Zhou, H. et al. (2021). _Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting_. AAAI.**

- Proposes ProbSparse attention to reduce O(L²) complexity for very long sequences.
- **Relevance:** Cite to acknowledge the quadratic complexity problem of standard attention and justify why it is not a concern for L=288. Also useful context for why sparse attention exists.

**Nie, Y. et al. (2023). _A Time Series is Worth 64 Words: Long-term Forecasting with Transformers_ (PatchTST). ICLR.**

- Divides the time series into patches (sub-sequences) treated as tokens, analogous to Vision Transformers. Reduces sequence length from L to L/P. Strong results on forecasting benchmarks.
- **Relevance:** Alternative tokenisation strategy. Mention as a design choice we considered but did not adopt — we use per-timestep tokens to preserve the full 5-min resolution needed for driver response modelling.

**Garza, A. & Mergenthaler-Canseco, M. (2023). _TimeGPT-1_. arXiv.**

- A foundation model for general time series, pre-trained on a large corpus of diverse time series. Demonstrates the foundation model paradigm in time series.
- **Relevance:** Motivates the foundation model approach (pre-train on large diverse data, fine-tune on specific task). Cite in the introduction when establishing that foundation models work for time series.

**Goswami, M. et al. (2024). _MOMENT: A Family of Open Time-series Foundation Models_. ICML.**

- Open foundation models for time series, pre-trained on large heterogeneous corpora using masked prediction (MTSM-style objective).
- **Relevance:** Direct precedent for our MTSM pre-training strategy. Cite when introducing the masked reconstruction objective. Note the difference: MOMENT is domain-agnostic, our model is domain-specific (T1D glucose + physiological drivers).

---

## 3. Glucose-Specific Foundation Models

**Lutsker, G. et al. (GluFormer). (2025). _A foundation model for continuous glucose monitoring data_. Nature. (arXiv v1: Aug 2024, Nature pub: Jan 2025.)**

#### Architecture & Training

- Discretises CGM into 460 glucose bins (tokens). Causal Transformer decoder trained with **next-token prediction** (cross-entropy loss). Context window of 1200 tokens (~300h at 15-min resolution). Trained on >10M glucose measurements from 10,812 non-diabetic adults (Human Phenotype Project, Israel). Not a T1D dataset — mostly normoglycemic/prediabetic adults.
- Patient embedding: **max pooling** over the final hidden states of all tokens → 1024-d vector. Deterministic (no VAE, no probabilistic encoder).
- Multimodal extension (GluFormer+Diet): discrete dietary tokens (meals described as macronutrient bins) concatenated into the autoregressive sequence before the corresponding postprandial window. The model autoregressively predicts future CGM tokens conditioned on those dietary tokens — no separate decoder, no latent perturbation, just next-token prediction with dietary context.

#### Results

- Generalises to 19 external cohorts (n=6,044): 5 countries, 8 CGM devices, diverse conditions (T1D, T2D, gestational diabetes, obesity). Outperforms traditional CGM metrics (GMI, TIR) for predicting HbA1c, liver markers, lipids, sleep indices. In a 580-person 12-year longitudinal study, top quartile of GluFormer risk score captured 66% of new-onset diabetes cases and 69% of cardiovascular deaths.

#### How GluFormer generates counterfactual glucose trajectories

GluFormer+Diet feeds dietary tokens + recent CGM history as context into the causal decoder, which then **autoregressively samples** future CGM tokens. Different meal compositions produce different trajectories. Crucially, this is generative by nature (causal LM sampling), not by a VAE. Stochasticity comes from the sampling temperature at inference, not from a learned prior over patient state.

#### Differences from our model — for thesis positioning

|Dimension|GluFormer|Our model|
|---|---|---|
|Pre-training objective|Next-token prediction (autoregressive, causal)|Masked reconstruction (bidirectional, BERT-style)|
|Attention direction|Unidirectional (past → future only)|Bidirectional (full context)|
|Input modalities|CGM only|CGM + PI + RA + bolus + carbs + modality|
|Population|Non-diabetic adults (HPP cohort)|T1D patients (METABONET + T1DEXI)|
|Transfer capability|Single-task forecasting only|Multi-task: forecasting + hypo risk + imputation + ISF/CR|
|Physiological grounding|None|PI and RA (Hovorka model) as explicit input drivers|
|Clinical target|Risk stratification, long-term outcomes (non-T1D)|Multi-task therapy support (T1D)|

**Key argument for thesis:** GluFormer is the dominant CGM foundation model but it is a single-task forecasting model — causal attention means it cannot use future context to fill a gap, it has no multi-task transfer capability, and it has no principled way to simulate counterfactual insulin scenarios. Our model addresses this: the bidirectional encoder learns a rich representation H grounded in physiological drivers (PI, RA), and a single frozen encoder supports multiple downstream tasks simultaneously — forecasting, hypo risk prediction, imputation, ISF/CR profiling. This is architecturally impossible in GluFormer's single-task autoregressive design.

- **Cite:** When introducing related work on CGM foundation models. Cite extensively when positioning our contribution in the Discussion. Also cite in the Introduction to motivate the need for T1D-specific, driver-aware, counterfactual models.

**Lu, Y. et al. (CGMformer). (2025). _A pretrained transformer model for decoding individual glucose dynamics from continuous glucose monitoring data_. National Science Review.**

#### Architecture & Training

- Bidirectional Transformer encoder (BERT-style). Pre-training objective: **masked CGM reconstruction** with 45–60% masking ratio. Key innovation: **TF-IDF weighted masking** — glucose tokens representing hyperglycaemia (>180 mg/dL) and hypoglycaemia (<70 mg/dL) are assigned higher masking probability, so the model is forced to reconstruct clinically abnormal moments more often. This is the CGM-domain equivalent of weighting rare/informative tokens more in NLP.
- Training corpus: 964 participants from a Nationwide Multicenter Chinese hospital study (Normoglycemic + IGR + T2D). Pre-training extended to 58,847 users for 1,310,548 days (National Real-World CGM). **T2D/prediabetes population, not T1D.**
- Patient embedding: mean of all time-point hidden states H → fixed-size individual embedding vector. Deterministic (no VAE).
- CGMformer_Diet: takes the individual embedding + pre-meal CGM values + meal macronutrient description → **MLP** that outputs a predicted postprandial glucose curve. Deterministic regression, not generative.

#### Latent space — where does it come from?

CGMformer does **not** use a VAE. The "latent space" referenced in the paper is simply the individual embedding — the mean hidden state vector produced by the bidirectional encoder for a given 24h window. It is a deterministic function of the input; there is no learned prior, no reparameterisation trick, and no ability to sample diverse trajectories for the same patient. For CGMformer_Diet, the meal perturbation is fed as a new input to the MLP head — not as a modification of the latent code. This means CGMformer cannot perform counterfactual simulation in the generative sense: it cannot ask "what would this patient's glucose trajectory look like if I changed the insulin dose?", only "given this fixed patient embedding and this meal description, what is the expected postprandial curve?".

#### Results

- MAE = 3.7 mg/dL on glucose imputation across 5 external datasets. AUROC = 0.914 for T2D screening, 0.741 for complication screening. Postprandial prediction Pearson r = 0.763. Identifies 6 non-diabetic sub-clusters including a lean-but-dysglycemic phenotype missed by traditional glucose measurements.

#### Differences from our model — for thesis positioning

|Dimension|CGMformer|Our model|
|---|---|---|
|Pre-training objective|Masked reconstruction (bidirectional ✅)|Masked reconstruction (bidirectional ✅)|
|Masking strategy|TF-IDF weighting by glucose level (hyper/hypo)|Driver-weighted loss (×3 on postprandial/post-insulin spans)|
|Input modalities|CGM only|CGM + PI + RA + bolus + carbs + modality|
|Population|T2D / prediabetes (Chinese multicenter)|T1D (METABONET + T1DEXI)|
|Patient representation|Deterministic mean-pool embedding|Full H (288, 128) — temporal resolution preserved|
|Transfer capability|Single task demonstrated per paper|Multi-task: one frozen encoder → forecasting + hypo risk + imputation|
|Downstream tasks|Classification, imputation, dietary recommendation|Forecasting, survival/hypo, imputation, ISF/CR profiling|
|Uncertainty|None|Stage 2 task-specific (survival: Weibull; twin: CVAE)|

**Key argument for thesis on masking:** The TF-IDF weighting biases _which timesteps get masked_ — the model must reconstruct peaks and valleys more often. Our driver-weighted loss instead biases _which masked timesteps receive higher gradient weight_ — the model is penalised more for failing to reconstruct the response to a clinical event. These are complementary strategies. An extension worth noting: we could combine both — use TF-IDF-style placement (bias spans toward postprandial windows) AND driver-weighted loss for gradient emphasis within those spans.

**Key argument for thesis on representation:** CGMformer's "latent space" is a deterministic mean-pool over H — a single fixed vector per window. Our H is the full `(288, 128)` sequence, preserving temporal resolution and enabling cross-attention decoding, attention pooling for classification, and timestep-level probing. More importantly, our encoder is explicitly designed for multi-task transfer: the same frozen H feeds forecasting heads, survival heads, and imputation — a capability CGMformer never demonstrated.

- **Cite:** When describing our MTSM masking strategy and justifying driver-weighted loss as a domain-specific alternative to TF-IDF weighting. Also cite in Related Work as the closest bidirectional pre-training precedent for CGM.

---

**Rosenthal, I. et al. (2023). _Interpretable Mechanistic Representations for Meal-level Glycemic Control in the Wild_. arXiv:2312.03344.**

#### Architecture

- Hybrid VAE: neural network encoder + **ODE decoder** (mechanistic, physiologically grounded). The latent variables z act as inputs to the ODE system — they define physiological parameters (insulin sensitivity, glucose appearance rate) rather than abstract embedding dimensions. The ODE is the Hovorka/minimal model structure. Amortised variational inference: a single encoder network infers z for any new (CGM, meal) pair without re-fitting.
- Dataset: 964 postprandial responses from 33 individuals. Sequence length T=60 (5h window: 1h pre-meal context + 4h postprandial).

#### Key ideas

1. **Mechanistic decoder as inductive bias:** instead of a free neural decoder that could learn any function, the decoder is constrained to physiologically plausible trajectories defined by the ODE. This prunes the hypothesis space and improves generalisation on small datasets.
2. **Interpretable latent axes:** because z parameterises the ODE, each dimension has a clinical meaning (e.g., insulin sensitivity, gastric emptying rate). No rotation ambiguity — unlike vanilla VAE where latent dimensions can mix freely.
3. **Robust to erroneous meal data:** the encoder simultaneously infers an effective glucose appearance rate from CGM, correcting for unreported or incorrectly logged meals.

#### Differences from our model

- Ours uses a neural decoder (not ODE), which is more flexible but less interpretable. The ODE approach requires fixing a physiological model structure, which is appropriate for a single-meal window (T=60, 5h) but harder to apply to our full 24h window (L=288) with multiple meals and exercise. Our PI and RA features (Hovorka-derived) inject physiological structure as input features rather than as decoder constraints — a middle ground between full mechanistic decoding and a purely data-driven decoder.
    
- Theirs is meal-level (33 subjects, 964 meals). Ours is patient-level across the full day (1046 patients, 1.4M windows).
    
- **Relevance:** Strong precedent for the principle that latent variables in glucose models should have physiological interpretability. Supports our design choice of conditioning the decoder on PI and RA rather than raw bolus/carbs, and motivates the broader goal of disentangled latent representations. Cite in Methods (VAE section) and Discussion (interpretability, comparison to fully mechanistic approaches).
    
- **Note for thesis:** This paper is a direct bridge between mechanistic ODE modelling (Hovorka 2004) and latent variable generative models — exactly the intellectual lineage our model occupies. Worth citing together with Hovorka 2004 to frame our approach as a data-driven relaxation of classical mechanistic modelling.
    

**CGM-LSM. (2025). _A large sensor foundation model pretrained on continuous glucose monitor data for diabetes management_. npj Health Systems.**

- Autoregressive Transformer decoder pre-trained on 1.6M CGM records. 48.51% RMSE reduction vs prior methods on 1h forecasting. Strong zero-shot generalisation across held-out patient groups.
- **Relevance:** Second autoregressive CGM foundation model. Reinforces the trend toward large-scale pre-training for CGM. Cite alongside GluFormer as evidence that foundation models outperform task-specific models for glucose. Note that both are forecasting-focused (causal), not representation-focused (bidirectional) as in our work.

---

## 4. Self-Supervised Pre-training Strategies

**Devlin, J. et al. (2019). _BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding_. NAACL.**

- The masked language model (MLM) objective: mask 15% of tokens, train the model to reconstruct them using bidirectional context. The direct conceptual basis for our MTSM.
- **Relevance:** Foundational reference for the masked reconstruction pre-training paradigm. Cite when introducing the MTSM objective and explaining why bidirectional attention is appropriate for pre-training (vs causal for forecasting).

**Zhang, X. et al. (2022). _Self-Supervised Contrastive Pre-Training For Time Series via Time-Frequency Consistency_ (TF-C). NeurIPS.**

- Contrastive pre-training based on the principle that time-domain and frequency-domain representations of the same time series should be close in a joint latent space. Evaluated on EEG, activity recognition, fault detection. 15.4% improvement over baselines in one-to-one transfer.
- **Relevance:** The frequency-domain consistency idea is relevant for CGM — the signal has interpretable spectral components (circadian ~24h, postprandial ~2-4h). Forcing H to be consistent with the FFT of the CGM signal could improve circadian and postprandial pattern capture without explicit supervision. Currently not implemented — annotate as **future work / possible extension**.

**Krishnan, R., Rajpurkar, P. & Topol, E.J. (2022). _Self-supervised learning in medicine and healthcare_. Nature Biomedical Engineering.**

- Systematic overview of SSL approaches in biomedical domains. Covers contrastive, generative (masked), and predictive pre-training strategies.
- **Relevance:** Background reference for the SSL motivation (no labels needed, pre-training on large unlabelled datasets). Cite in the introduction when arguing that SSL is the right pre-training paradigm for clinical time series.

---

## 4b. Representation Learning Objectives (H Enrichment)

**Bardes, A., Ponce, J. & LeCun, Y. (2022). _VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning_. ICLR.**

- Adds three terms to the SSL loss: variance (prevent dimension collapse), invariance (two views of same sample should be close), covariance (decorrelate dimensions). No negative pairs needed.
- **Relevance:** Used in run15 (`--vicreg_lambda 0.05`) to prevent H dimension collapse. Result: PC1_L5 dropped from 60.5% → 21.0% (good) but MLP probe R²_L5 dropped to 0.66 (information loss — too aggressive at λ=0.05). Negative result — documented. Lower λ may work; not carried forward.

**He, K. et al. (2022). _Masked Autoencoders Are Scalable Vision Learners_ (MAE). CVPR.**

- Reconstruction-based SSL at patch level. Very high masking ratios (75%) force the encoder to develop abstract representations. Conceptual basis for our MTSM with long span masking.
- **Relevance:** Justifies our high masking ratio (35%) and long spans. Cite when introducing MTSM.

**Assran, M. et al. (2023). _Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture_ (I-JEPA). CVPR.**

- Predicts abstract H-space representations of masked regions rather than raw pixels, using an EMA target encoder. Produces more semantic representations than pixel-level reconstruction.
- **Relevance:** Direct basis for run20 (JEPA objective) in the H enrichment pipeline. Key idea: predicting in H-space (abstract) vs predicting raw CGM (low-level). If run20 works, cite as the primary motivation for the JEPA objective.

**Chen, T. et al. (2020). _A Simple Framework for Contrastive Learning of Visual Representations_ (SimCLR). ICML.**

- Contrastive SSL: pull together representations of augmented views of the same sample, push apart representations of different samples. InfoNCE loss.
- **Relevance:** Conceptual basis for run17 (`--contrastive_lambda 0.1`), which uses consecutive same-patient windows as positive pairs. Cite when introducing the contrastive term.

---

## 5. Variational Autoencoders & Generative Models

**Kingma, D.P. & Welling, M. (2013). _Auto-Encoding Variational Bayes_. ICLR.**

- The original VAE. Defines the ELBO objective (reconstruction loss + KL divergence), the reparameterisation trick, and the probabilistic encoder-decoder framework.
- **Relevance:** Foundational reference for our Stage 2. Cite when introducing the VAE encoder, the reparameterisation trick, and the KL term.

**Higgins, I. et al. (2017). _β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework_. ICLR.**

- β-VAE: multiplies the KL term by β > 1 to encourage disentanglement of latent factors. Higher β → more structured, disentangled latent space but worse reconstruction.
- **Relevance:** Directly used in our loss function. Cite when introducing β and the annealing schedule. The disentanglement argument is relevant for our digital twin: we want z to capture patient-specific metabolic state independently of the driver variables c.

**Tashiro, Y. et al. (2021). _CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation_. NeurIPS.**

- Diffusion model for time series imputation and forecasting. Conditional on observed values. Probabilistic — generates a distribution of plausible completions.
- **Relevance:** Conceptually related to our MTSM (both do conditional reconstruction of missing time series segments). More powerful generative model but much more complex. Cite as an alternative generative approach and note that our VAE-based approach is more interpretable and directly supports counterfactual generation.

---

## 6. Physiological Models

**Hovorka, R. et al. (2004). _Nonlinear model predictive control of glucose concentration in subjects with type 1 diabetes_. Physiological Measurement.**

- The Hovorka model: compartmental ODE system for insulin pharmacokinetics (plasma insulin PI) and carbohydrate absorption (rate of appearance RA). Standard in closed-loop insulin delivery research.
- **Relevance:** Direct basis for our PI and RA feature computation in the preprocessing pipeline. Cite when describing the Hovorka model integration and justifying the choice of physiological features over raw bolus/carbs events.

**Isaac, S. et al. (2025). _SSM-CGM: Interpretable State-Space Forecasting Model of Continuous Glucose Monitoring_. arXiv:2510.04386.**

- State-space model for CGM forecasting. Interpretable by design.
- **Relevance:** Alternative modelling approach using state-space models. Useful for the related work section to contrast with Transformer-based approaches. Note that SSMs have linear complexity in sequence length (vs O(L²) for attention).

**Gu, A. & Dao, T. (2023). _Mamba: Linear-Time Sequence Modeling with Selective State Spaces_. arXiv:2312.00752.**

- Selective state-space model with linear complexity. Competitive with Transformers on long sequences.
- **Relevance:** Architecture alternative to the Transformer for long sequences. Relevant if the model is eventually extended to longer context windows. Currently not used — mention in future work if sequence length becomes a bottleneck.

---

## 7. Datasets

**T1DEXI — Type 1 Diabetes Exercise Initiative.**

- Multi-site study with CGM, insulin pump data, and exercise annotations for T1D patients.
- **Relevance:** One of the two datasets in our training corpus. Cite when describing the dataset.

**METABONET.**

- Multi-modal metabolic dataset including CGM, insulin, and dietary data.
- **Relevance:** Second dataset in our training corpus. Cite when describing the dataset.

---

## 8. Design Decisions & Open Questions

_(Notes for the thesis — not citations)_

### 8.1 MTSM masking strategy — resolved

**Adopted approach:** contiguous spans of 5–8h (MASK_MIN_LEN=60, MASK_MAX_LEN=96 steps), driver-weighted loss (×3 on timesteps within 2h after a bolus/carbs event). Mask token = 0.0 (z-score mean).

**Why 5–8h minimum:** shorter spans are reconstructible by local interpolation from span boundaries. 5h minimum forces the encoder to use physiological driver context (PI curve, RA timing) — it cannot simply interpolate.

**Alternative (CGMformer-inspired):** bias span *placement* toward postprandial windows (TF-IDF-style). Remains a possible extension but not currently implemented — driver-weighted loss achieves similar gradient emphasis without changing the data distribution.

**Status: resolved.** Current masking config confirmed as best across runs 1–12.

### 8.2 TF-C extension (future work)

Add a frequency-domain consistency loss alongside the MTSM objective: for each window, compute the FFT of the CGM signal and train a lightweight frequency encoder in parallel. Enforce that the time-domain representation h (post attention pooling) and the frequency-domain representation are close in the latent space. Expected benefit: H captures circadian and postprandial frequency components explicitly.

**Status:** not implemented. Mention as future work in thesis.

### 8.3 Population-level H analysis (complete)

Extensive H analysis implemented in `scripts/analyse_H.py` (16 diagnostic plots). Includes: PCA by modality/age, t-SNE, per-layer feature correlation heatmap, event-triggered H norm, linear and non-linear (MLP) reconstruction probes, attention matrices per layer.

Key finding: H develops progressive abstraction across layers. L1 encodes raw CGM directly (r≈+0.9). L5 inverts (r≈−0.4) — H is a transformed representation, not smoothed CGM. The CGM sign flip at L5 and the dominant PC1 axis serve as a globally consistent H convention that downstream heads can rely on.

**Status: complete.** Run on runs 12–15. Will be re-run on each enrichment run.

### 8.4 Driver blindness (acknowledged limitation)

MDI patients with incomplete bolus logging produce PI=0, RA=0 throughout — encoder generates H without access to actual insulin state. Mitigation: modality one-hot (AID/SAP/MDI) in input, wider VAE posterior for high-uncertainty windows.

---

## 9. Pre-training Strategy Comparison

_(Summary table for supervisor presentation)_

|Strategy|Bidirectional|Drivers|Complexity|Relevant papers|Status|
|---|---|---|---|---|---|
|**MTSM (ours)**|✅|✅ (loss weight)|Medium|BERT, MOMENT, CGMformer|Active|
|Autoregressive|❌|❌|Low|GluFormer, CGM-LSM|Discarded — causal, no counterfactuals|
|Masked + clinical weighting|✅|Partial|Low|CGMformer (TF-IDF)|Possible replacement for loss weighting|
|Contrastive TF-C|✅|❌|High|TF-C (Zhang 2022)|Future work|

---

## 9b. Three-Way Model Comparison: GluFormer vs CGMformer vs Ours

_(Ready-to-use table for thesis Related Work section)_

|Dimension|GluFormer (Nature 2025)|CGMformer (Nat Sci Rev 2025)|**Our model**|
|---|---|---|---|
|Pre-training objective|Next-token prediction (causal)|Masked reconstruction (bidirectional)|Masked reconstruction (bidirectional)|
|Attention|Unidirectional|Bidirectional ✅|Bidirectional ✅|
|Input|CGM only|CGM only|CGM + PI + RA + bolus + carbs + modality ✅|
|Physiological grounding|None|None|Hovorka-derived PI and RA ✅|
|Population|Non-diabetic adults|T2D / prediabetes (Chinese)|**T1D** (METABONET + T1DEXI) ✅|
|Dataset size|10,812 subjects, >10M measurements|964 subjects (pre-train); 58,847 (extended)|988 adults, 951K windows|
|Transfer capability|Single task|Single task per paper|**Multi-task: one encoder → 4+ applications** ✅|
|Patient representation|Deterministic (max pool)|Deterministic (mean-pool)|Full H (288, 128) — temporal resolution preserved ✅|
|Downstream tasks|Risk stratification (single)|Classification, imputation (single)|Forecasting + hypo risk + imputation + ISF/CR ✅|
|Uncertainty|Sampling temperature (heuristic)|None|Task-specific (Weibull survival, CVAE twin) ✅|

**Summary sentence for thesis:** GluFormer and CGMformer establish that pre-trained Transformer encoders learn clinically meaningful representations from large CGM corpora; however, both are single-task models trained on non-T1D populations with CGM-only inputs. Our model extends this paradigm to T1D by incorporating physiological driver variables (insulin pharmacokinetics and carbohydrate absorption via the Hovorka model), and — critically — demonstrates that a single frozen encoder can serve as a foundation for multiple clinically relevant downstream tasks simultaneously. This multi-task transfer capability, grounded in explicit physiological drivers, is the primary differentiating contribution.

---

## 10. Citation Checklist for Thesis Sections

_(Fill in as writing progresses)_

|Thesis section|Key references|
|---|---|
|Introduction — motivation|Krishnan 2022, GluFormer Nature 2025, CGMformer, MOMENT|
|Related work — Transformers|Vaswani 2017, Lim 2021, Nie 2023, Garza 2023|
|Related work — CGM models|GluFormer, CGMformer, CGM-LSM, Rosenthal 2023|
|Related work — SSL / foundation models|BERT, MOMENT, TF-C|
|Methods — preprocessing|Hovorka 2004, docs/Preprocessing.md|
|Methods — encoder|Vaswani 2017|
|Methods — MTSM pre-training|BERT, MOMENT, CGMformer (masking comparison)|
|Methods — Stage 2 forecasting head|Lim 2021 (cross-attention), GluFormer (benchmark)|
|Methods — Stage 2 survival head|DeepHit / Weibull references (to be added)|
|Methods — Stage 2 digital twin (outlook)|Kingma 2013, Higgins 2017, Rosenthal 2023, LoRA|
|Discussion — limitations|SSM-CGM, Mamba (alternative architectures)|