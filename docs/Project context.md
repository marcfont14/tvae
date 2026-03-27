# Glucose Foundation Model — Project Context

**Undergraduate Thesis in Biomedical Engineering** _Last updated: 2026-03-27_

---

## 1. Project Goal

Build a **glucose foundation model for Type 1 Diabetes (T1D)** and demonstrate that a single pre-trained encoder can support **multiple clinically relevant downstream applications** — the core thesis of foundation models applied to continuous glucose monitoring.

**Two-stage strategy:**

**Stage 1 (complete):** Bidirectional Transformer encoder pre-trained with Masked Time Series Modelling (MTSM) on 988 adult T1D patients. The encoder learns a rich contextualised representation H of each 24h window integrating CGM, plasma insulin (PI), carb absorption (RA), and logged events.

**Stage 2 (in design):** Multiple task-specific heads that hang from the frozen Stage 1 encoder, each addressing a different clinical application. The central claim: **one representation, many applications** — the same H supports tasks that previously required separate bespoke models.

**Key differentiator vs prior work:** GluFormer and CGMformer are single-task models. This work explicitly demonstrates multi-task transfer from one pre-trained encoder, grounded in physiological drivers (PI, RA) that prior CGM foundation models ignore.

---

## 2. Dataset

**Sources:** METABONET + T1DEXI adults + T1DEXI pediatric (merged)

| Dataset | Patients | Age range | Modality |
|---|---|---|---|
| METABONET | 831 | 1–80 | AID/SAP/MDI |
| T1DEXI adults | 497 | 18–40 | AID/MDI |
| T1DEXI pediatric | 247 | 12–17 | AID |
| **Total** | **1,575** | **1–80** | AID 81%, SAP 11%, MDI 8% |

**After quality filter:**
- `data/processed/all/`: 1,441 patients, 1.46M windows
- `data/processed/adults/`: 988 patients, 951K windows (active training set)

**Per-patient .npz keys:** `windows`, `scaler_mean`, `scaler_std`, `patient_id`, `modality`, `age`

**Window shape:** `(N, 288, 11)` — 288 timesteps × 11 features. Stride = 72 steps (6h). Windows with >20% null CGM discarded.

**Feature tensor (axis=2):**

| Index | Feature | Description |
|---|---|---|
| 0 | CGM | Normalised glucose (z-score per patient) |
| 1 | PI | Plasma insulin (Hovorka ODE) |
| 2 | RA | Rate of carb absorption (Hovorka ODE) |
| 3 | hour_sin | sin(2π × hour / 24) |
| 4 | hour_cos | cos(2π × hour / 24) |
| 5 | bolus_logged | Binary bolus event flag |
| 6 | carbs_logged | Binary carbs event flag |
| 7 | AID | One-hot modality |
| 8 | SAP | One-hot modality |
| 9 | MDI | One-hot modality |
| 10 | age_norm | age/100, static per patient |

**Note:** Feature 10 (age_norm) is dropped from encoder input in the current baseline (run14+) via `--no_age`. Data is still stored as 11 features; sliced to 10 before the model. Age will be passed directly to Stage 2 heads (late fusion).

---

## 3. Architecture

### 3.1 Stage 1 — MTSM Encoder (complete)

```
Input (288, 10 or 11)
      │
      ▼
Input Projection       Dense(→128), no activation, per timestep
      │
      ▼
Positional Encoding    Sinusoidal, fixed → (288, 128)
      │
      ▼
Transformer Encoder × 5 layers
  Each layer:
    MultiHeadAttention (4 heads, key_dim=32, dropout=0.2)
    + Residual → LayerNorm
    FFN: Dense(128→256, ReLU) → Dense(256→128, dropout=0.2)
    + Residual → LayerNorm
      │
      ▼
      H  (288, 128)     ← the representation we keep
      │
      ▼
Reconstruction Head    Dense(128→64, ReLU) → Dense(64→1)  [DISCARDED after Stage 1]
```

**Parameters:** ~660K total (encoder ~640K, head ~8K)

**Current best hyperparameters (run12/run14 baseline):**
```
d_model=128, n_heads=4, n_layers=5, d_ff=256
dropout=0.2, batch_size=128, epochs=70 (early stopping patience=10)
mask_ratio=0.35, mask_min_len=60 (5h), mask_max_len=96 (8h)
driver_loss_weight=3.0, driver_effect_steps=24 (2h)
```

**MTSM objective:** mask contiguous spans of CGM (feature 0), predict masked values using full visible context including PI, RA, bolus, carbs.

**Why 2-layer MLP head (not LSTM, not single Dense):** H_t is already fully contextualised via self-attention. An MLP forces all temporal reasoning into H. A single Dense(1) is insufficient; an LSTM would reduce pressure on the encoder.

**Why long spans (5–8h):** shorter spans are reconstructible by local interpolation. 5h minimum guarantees the model must use physiological driver context.

**Why driver weighting:** without it, the model optimises on easy flat basal periods and ignores the physiologically important postprandial/post-bolus zones.

### 3.2 Stage 2 — Downstream Applications (in design)

The Stage 1 encoder is **frozen**. Each application adds a lightweight task-specific head on top of H.

**Planned applications (priority order):**

1. **Short-Horizon Forecasting (2–4h)** — H as Keys/Values; future driver sequence as Queries → cross-attention → MLP → ŷ. Benchmarkable against GluFormer/CGMformer.
2. **Time-to-Hypo Survival Analysis** — AttentionPooling(H) → MLP → Weibull(k, λ) parameters. Predict time distribution until CGM < 70 mg/dL.
3. **Gap Imputation** — zero-shot from MTSM masked reconstruction. Free result.
4. **Dynamic ISF/CR Profiling** — Siamese perturbation: encoder on original + synthetic +1U insulin perturbation → MLP([H, H']) → ΔGlucose.
5. **Personalised Digital Twin (LoRA)** — CVAE decoder + per-patient LoRA fine-tuning. Thesis outlook.

---

## 4. MTSM Pre-training — Key Decisions and Results

### Masking design

Contiguous spans of 5–8h (not random tokens). Masked positions replaced with 0.0 (z-score mean). Longer spans prevent local interpolation — the encoder must use physiological drivers.

### Driver-weighted loss

Timesteps within 2h after a bolus/carbs event → weight 3.0. Others → weight 1.0. MAE tracked unweighted.

### Experiment history (abbreviated)

| Run | Key change | Result |
|---|---|---|
| Forecasting | Transformer vs MLP, 1h forecast | Regression-to-mean. Closed. |
| Runs 1–7 | Span tuning, driver weight, shape loss, multimodal masking | Progressive improvement. Shape loss and multimodal masking (v1) not recommended. |
| **Run 8** | batch_size=128, MASK_MIN_LEN=60 | **Best small-scale run (150 patients). Val MAE=0.485.** |
| Run 9–10 | All patients, 35–50 epochs | Not converged. H more diagonal with mixed population. |
| Run 11 | Adults only, 70 epochs, MLP head | Superseded by run12. |
| **Run 12** | 2-layer MLP head, adults only, 70 epochs | **Best run. Val MAE=0.46. Early stop ~56ep. Strong attention + PI/RA reactivity.** |
| Run 13 | Ablation: `--no_logged_events` | Negative. CGM sign flip absent. Discrete flags are critical for H organisation. |
| **Run 14** | `--no_age` (clean baseline) | **New baseline. Val MAE=0.45. PC1_L5=60.5%. CGM sign flip preserved.** |
| Run 15 | `--vicreg_lambda 0.05` | Negative. PC1_L5=21% (good) but MLP probe L5 R²=0.66 vs run14 linear 0.94 — real information loss. |

### Best run metrics (run14 — active baseline)

- Val MAE ~0.45, train ~0.41
- PC1_L5=60.5%, PC2=14.2%
- L5 pooled Pearson r: CGM=−0.37, PI=+0.08 (CGM sign flip preserved)
- Linear probe R²_L5=0.939
- No early stopping (ran 70 epochs)

### Key H representation findings

- **H_t norm encodes metabolic complexity**, not raw glucose level. High norm = high physiological activity.
- **CGM sign flip at L5:** early layers encode raw CGM positively (r≈+0.9); L5 inverts (r≈−0.4) — H is a transformed, abstract representation, not just smoothed CGM.
- **Bolus/carbs flags are critical:** they serve as temporal anchors forcing a globally consistent H convention. Without them (run13), reconstruction MAE degrades only marginally but H loses its organised structure — harmful for Stage 2.
- **age_norm dropped from encoder (run14+):** Late fusion design. Including age caused demographic shortcuts in attention. Age will be passed directly to Stage 2 heads.

---

## 5. H Enrichment Pipeline (current work)

**Goal:** maximise H representation richness before Stage 2. Baseline = run14.

### H Richness Score

| Component | Formula | Direction |
|---|---|---|
| Distributed variance | `100 − PC1_L5 (%)` | ↑ higher |
| Feature coverage | `Σ\|r_L5\|` for CGM, PI, hour_sin | ↑ higher |
| Abstraction depth | `1 − R²_probe_L5` (linear) | ↑ higher |
| Reconstruction sanity | val MAE ≤ run14 + 0.02 | no regression |

**Run14 reference:** PC1_L5=60.5%, Σ|r|≈0.46, R²_probe=0.939, val MAE=0.45.

### Run table

| Run | Change | Status | Outcome |
|---|---|---|---|
| 14 | `--no_age` clean baseline | **COMPLETE** | New baseline |
| 15 | `--vicreg_lambda 0.05` | **COMPLETE** | Negative — λ too aggressive, info loss |
| 16 | `--multimodal_prob 0.3` (PI/RA masking, 2:1 ratio) | **RUNNING** | — |
| 17 | `--contrastive_lambda 0.1` (InfoNCE patient pairs) | Queued | — |
| 18 | `--d_model 256 --n_heads 8 --d_ff 512` (scale-up) | Queued | — |
| 19 | Patch encoding (48 patches × 30 min) | Queued — new script | — |
| 20 | JEPA (predict H-space, EMA target encoder) | Queued — new script | — |
| 21 | JEPA + Patch | Queued | — |
| 22 | Best Phase-1 + JEPA | Queued | — |

---

## 6. File Paths

### Scripts

| File | Location |
|---|---|
| MTSM training | `scripts/experiment_mtsm.py` |
| H representation analysis | `scripts/analyse_H.py` |
| Preprocessing | `src/preprocessing.py` |
| Merge datasets | `scripts/merge_datasets.py` |
| Parse T1DEXI | `scripts/parse_t1dexi.py` |

### Data

| Content | Location |
|---|---|
| Processed adults (active) | `data/processed/adults/*.npz` |
| Processed all ages | `data/processed/all/*.npz` |
| Merged interim | `data/interim/combined_filtered.parquet` |

### Results

| Content | Location |
|---|---|
| Best Stage 1 run | `results/mtsm/run12/` |
| Active baseline (no_age) | `results/mtsm/run14/` |
| All MTSM runs | `results/mtsm/{run_id}/` |

### Run commands

```bash
# H enrichment baseline (run14 pattern)
python scripts/experiment_mtsm.py \
    --data data/processed/adults --epochs 70 --run_id run1X --no_age

# Deep H analysis
python scripts/analyse_H.py \
    --data data/processed/adults --run_id run1X --no_age
```

### Container

All Python/TF code runs inside Docker container `tvae`. No virtualenv — dependencies installed system-wide.

```bash
docker start -ai tvae       # start if stopped
docker exec -it tvae bash   # enter running container
cd /mnt/workspace/tvae
```

---

## 7. Literature — Key References

| Paper | Relevance |
|---|---|
| **GluFormer** — Lutsker et al., Nature 2025 | Closest prior. Autoregressive, CGM-only, non-T1D, deterministic. |
| **CGMformer** — Lu et al., Nat Sci Rev 2025 | Bidirectional masked CGM. TF-IDF masking. T2D. Deterministic. |
| **Rosenthal et al., 2023** (VAE-ODE) | Hybrid VAE + ODE decoder. Physiological latent axes. Stage 2 precedent. |
| **BERT** — Devlin et al., NAACL 2019 | Conceptual basis for MTSM. |
| **MOMENT** — Goswami et al., ICML 2024 | Time series foundation model, masked reconstruction. |
| **PatchTST** — Nie et al., ICLR 2023 | Patch-level masking. Closest to our span masking design. |
| **Hovorka 2004** | Compartmental ODE for PI and RA computation. |
| **β-VAE** — Higgins et al., ICLR 2017 | Stage 2 loss and β annealing. |

### Key differentiators vs prior work

| | GluFormer | CGMformer | **Ours** |
|---|---|---|---|
| Attention | Causal | Bidirectional | Bidirectional |
| Inputs | CGM only | CGM only | CGM+PI+RA+bolus+carbs+modality |
| Population | Non-diabetic | T2D | **T1D** |
| Applications | Single-task | Single-task | **Multi-task from one encoder** |
| Counterfactual | Partial | None | Any driver variable (Stage 2) |

---

## 8. Current Status & Next Steps

### Done ✅

- Full preprocessing pipeline (Hovorka PI/RA, per-patient .npz, 11 features)
- Dataset: METABONET + T1DEXI adults + T1DEXI pediatric (1,575 → 988 adults after filter)
- MTSM training script with driver-weighted loss, span masking, diagnostic plots, encoder saving
- Stage 1 complete — run12 best overall, run14 clean baseline (no_age)
- Deep H analysis script (`analyse_H.py`) with 16 diagnostic plots including non-linear MLP probe
- Ablations complete: discrete flags critical (run13), age_norm → late fusion (run14), VICReg negative (run15)
- Stage 2 fully designed in `docs/stage_2_proposals.md`

### In progress 🔄

- H enrichment pipeline (runs 16–22): multimodal masking, contrastive, scale-up, patch encoding, JEPA
- Run 16 running: `--multimodal_prob 0.3`

### Next 📋

- Evaluate run16, queue run17
- Implement `experiment_patch.py` (run19) and `experiment_jepa.py` (run20)
- After enrichment: begin Stage 2 — App 2 (Forecasting) and App 1 (Survival/Hypo) first

### Thesis writing 📝

- [ ] Related Work — literature table ready
- [ ] Methods (preprocessing) — `src/Preprocessing.md` complete
- [ ] Methods (Stage 1) — needs writing based on run12/run14 results
- [ ] Results (Stage 1) — H analysis plots ready for all runs
