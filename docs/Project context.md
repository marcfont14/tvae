# Glucose Foundation Model — Project Context

**Undergraduate Thesis in Biomedical Engineering** — _Last updated: 2026-04-10_

---

## 1. Project Goal

Build a **glucose foundation model for Type 1 Diabetes (T1D)** and demonstrate that a single pre-trained encoder can support **multiple clinically relevant downstream applications** — the core thesis of foundation models applied to continuous glucose monitoring.

**Two-stage strategy:**

**Stage 1 (complete):** Bidirectional Transformer encoder pre-trained with Masked Time Series Modelling (MTSM) on 988 adult T1D patients. The encoder learns a rich contextualised representation H of each 24h window integrating CGM, plasma insulin (PI), carb absorption (RA), and logged events.

**Stage 2 (fully designed, implementation pending):** Multiple task-specific heads that hang from the frozen Stage 1 encoder, each addressing a different clinical application. The central claim: **one representation, many applications** — the same H supports tasks that previously required separate bespoke models. Full specifications in `docs/stage_2_proposals.md`.

**Key differentiator vs prior work:** GluFormer and CGMformer are single-task CGM-only models. This work explicitly demonstrates multi-task transfer from one pre-trained encoder grounded in physiological drivers (PI, RA).

---

## 2. Dataset

**Sources:** METABONET + T1DEXI adults (adults-only scope — fixed)

| Dataset | Patients | Age range | Modality |
|---|---|---|---|
| METABONET | 831 | 1–80 | AID/SAP/MDI |
| T1DEXI adults | 497 | 18–40 | AID/MDI |
| T1DEXI pediatric | 247 | 12–17 | AID |
| **After quality filter + age ≥ 18** | **988** | **18–80** | AID/SAP/MDI |

Active training set: `data/processed/adults/` — 988 patients, ~951K windows.

**Window shape:** `(N, 288, 11)` — 288 timesteps × 11 features. Stride = 72 steps (6h). Encoder receives 10 features (age excluded — see below).

**Feature tensor:**

| Index | Feature | Description |
|---|---|---|
| 0 | CGM | z-score per patient |
| 1 | PI | Plasma insulin (Hovorka 3-compartment ODE) |
| 2 | RA | Carb absorption rate (Hovorka 2-compartment ODE) |
| 3 | hour_sin | sin(2π × hour / 24) |
| 4 | hour_cos | cos(2π × hour / 24) |
| 5 | bolus_logged | Binary bolus event flag |
| 6 | carbs_logged | Binary carbs event flag |
| 7 | AID | One-hot modality |
| 8 | SAP | One-hot modality |
| 9 | MDI | One-hot modality |
| 10 | age_norm | age/100 — stored in .npz but **excluded from encoder** (late fusion) |

---

## 3. Architecture

### 3.1 Stage 1 — MTSM Encoder

```
Input (288, 10)
      │
      ▼
Input Projection       Dense(→128), no activation
      │
      ▼
Positional Encoding    Sinusoidal, fixed
      │
      ▼
Transformer Encoder × 5 layers
  Each layer:
    MultiHeadAttention (4 heads, key_dim=32, dropout=0.2)
    + Residual → LayerNorm
    FFN: Dense(128→256, ReLU) → Dense(256→128)
    + Residual → LayerNorm
      │
      ▼
      H  (288, 128)     ← the representation we keep
      │
      ▼
Reconstruction Head    Dense(128→64, ReLU) → Dense(64→1)  [DISCARDED]
```

~660K parameters total (encoder ~640K, head ~8K).

**Baseline hyperparameters (run14):**
```
d_model=128, n_heads=4, n_layers=5, d_ff=256
dropout=0.2, batch_size=128, epochs=70
mask_ratio=0.35, mask_min_len=60 (5h), mask_max_len=96 (8h)
driver_loss_weight=3.0, driver_effect_steps=24 (2h)
```

### 3.2 Stage 2 — Downstream Applications

The Stage 1 encoder is **frozen**. Each application adds a lightweight task-specific head on top of H. Priority order:

1. **Short-Horizon Forecasting (2–4h)** — cross-attention decoder, H as K/V, future drivers as Q
2. **Time-to-Hypo Survival Analysis** — attention-pooled H → Weibull head
3. **Gap Imputation** — zero-shot reuse of MTSM reconstruction head
4. **Dynamic ISF/CR Profiling** — Siamese perturbation encoder
5. **Personalised Digital Twin (LoRA)** — CVAE decoder + per-patient LoRA fine-tuning (thesis outlook)

---

## 4. Stage 1 Results

### Completed experiments

| Run | Change | Result |
|---|---|---|
| 1–11 | Exploratory (span tuning, driver weight, population, head type) | Run 12 won |
| **12** | 2-layer MLP head, adults only, 70 epochs | **Val MAE=0.46. Best Stage 1. CGM sign flip −0.60 at L5.** |
| 13 | `--no_logged_events` ablation | Val MAE=0.47. Sign flip absent. Discrete flags are critical. |
| **14** | `--no_age` | Val MAE=0.45. PC1_L5=60.5%, Σ\|r\|≈0.46, R²=0.939. encoder_weights only. |
| **21** | Identical to run14, full weights saved | **Val MAE≈0.45. R²=0.944. Individual attention confirmed. FINAL ENCODER.** |
| 15 | `--vicreg_lambda 0.05` | Σ\|r\|≈0.06, sign flip absent. **Negative.** |
| 16 | `--multimodal_prob 0.3` | Diagonal attention, Σ\|r\|≈0.17, sign flip gone. **Negative.** |
| 17 | `--contrastive_lambda 0.1` | Custom training loop incompatible with TF XLA on CC 12.0. **Skipped.** |
| 18 | `--d_model 256 --n_heads 8 --d_ff 512` | Same diagonal pattern as run16. **Negative.** |
| **20** | JEPA — predict H-space, EMA target encoder | Collapse: R²_L5=0.039, PC1=98%. **Negative.** |

### Key H representation findings (run14 baseline)

- **H_t norm encodes metabolic complexity.** High norm = high physiological activity, not raw glucose level.
- **CGM sign flip at L5:** L1 encodes raw CGM positively (r≈+0.9). L5 inverts (r≈−0.37). H is an abstract transformed representation.
- **Discrete flags are critical:** ablation run13 shows that removing bolus_logged/carbs_logged causes H structure to collapse (sign flip absent, disorganised PCA) even though reconstruction MAE degrades only marginally.
- **Age excluded from encoder:** passing age_norm causes demographic shortcuts (diagonal attention). Age passed to Stage 2 heads directly as conditioning variable.

### H enrichment findings

All enrichment approaches returned one of two failure modes:

1. **Consistent failure pattern** (runs 15/16/18): near-zero feature correlations at L5, absent CGM sign flip, diagonal attention. Any auxiliary objective strong enough to alter H destroys the structural anchoring that the reconstruction + discrete flags combination produces.
2. **Representation collapse** (run20 JEPA): encoder outputs near-constant vectors (R²_L5=0.039, PC1=98%). EMA alone is insufficient to prevent collapse without an explicit anti-collapse term.

**H enrichment pipeline is closed. Run14 is the final Stage 1 encoder.**

---

## 5. File Paths

| Content | Location |
|---|---|
| MTSM training | `scripts/experiment_mtsm.py` |
| Replot / H analysis | `scripts/replot.py` |
| Individual attention visualisation | `scripts/attention_viz.py` |
| Preprocessing | `src/preprocessing.py` |
| Processed adults (active) | `data/processed/adults/*.npz` |
| **Final Stage 1 encoder** | `results/mtsm/run21/encoder_weights.weights.h5` |

### Container

```bash
docker start -ai tvae       # start if stopped
docker exec -it tvae bash   # enter running container
cd /mnt/workspace/tvae
python -u scripts/experiment_mtsm.py ... | tee results/mtsm/runXX_log.txt
```

---

## 6. Literature — Key References

| Paper | Relevance |
|---|---|
| **GluFormer** — Lutsker et al., Nature 2025 | Closest prior. Autoregressive, CGM-only, non-T1D, deterministic. |
| **CGMformer** — Lu et al., Nat Sci Rev 2025 | Bidirectional masked CGM. TF-IDF masking. T2D. |
| **BERT** — Devlin et al., NAACL 2019 | Conceptual basis for MTSM. |
| **MOMENT** — Goswami et al., ICML 2024 | Time series foundation model, masked reconstruction. |
| **PatchTST** — Nie et al., ICLR 2023 | Patch-level masking. Closest to our span masking design. |
| **I-JEPA** — Assran et al., CVPR 2023 | Basis for JEPA (run20): predict in representation space, not pixel space. |
| **Hovorka 2004** | Compartmental ODE for PI and RA. |
| **Rosenthal et al., 2023** | VAE-ODE hybrid. Physiological latent axes. Stage 2 precedent. |

### Key differentiators vs prior work

| | GluFormer | CGMformer | **Ours** |
|---|---|---|---|
| Attention | Causal | Bidirectional | Bidirectional |
| Inputs | CGM only | CGM only | CGM + PI + RA + events + modality |
| Population | Non-diabetic | T2D | **T1D adults** |
| Applications | Single-task | Single-task | **Multi-task from one encoder** |

---

## 7. Current Status

### Done ✅

- Full preprocessing pipeline (Hovorka PI/RA, per-patient .npz, adults only)
- MTSM training with driver-weighted loss, span masking, diagnostic plots
- Stage 1 complete — run14 final encoder (val MAE=0.45, R²=0.939)
- Ablations: flags critical (run13), age late fusion (run14)
- H enrichment pipeline complete — runs 15/16/17/18/20, all negative, pipeline closed
- Stage 2 fully designed — `docs/stage_2_proposals.md`
- Documentation: `docs/Preprocessing.md`, `docs/Stage1.md`, `docs/H_analysis.md` up to date

### In progress 🔄

- Stage 2 implementation

### Next 📋

- Stage 2: App 2 (Forecasting) + App 1 (Hypo Survival) first
- App 4 (Gap Imputation) — zero-shot, essentially free
- App 3 (ISF/CR Profiling) after

### Thesis writing 📝

- [ ] Related Work — literature table ready in `docs/Bibliography.md`
- [ ] Methods (preprocessing) — `docs/Preprocessing.md` complete ✅
- [ ] Methods (Stage 1 architecture + training) — `docs/Stage1.md` complete ✅
- [ ] Results (Stage 1) — H analysis plots ready; `docs/H_analysis.md` complete ✅
- [ ] Results (Stage 2) — pending implementation
