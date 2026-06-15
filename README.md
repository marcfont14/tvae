# Glucose Foundation Model — T1D CGM

Transformer-based foundation model pre-trained on continuous glucose monitor (CGM) data from 934 Type 1 Diabetes patients. The model learns glucose dynamics from short-term physiological signals (plasma insulin and carbohydrate absorption derived from the Hovorka ODE) and transfers those representations to downstream clinical tasks.

Pre-training follows a BERT/GPT dual-track: a bidirectional masked encoder and a causal next-token-prediction decoder, both operating on 24-hour windows at 5-minute resolution. Downstream tasks are gap imputation, short-horizon forecasting, and nocturnal hypoglycaemia risk.

This repository accompanies the undergraduate thesis *"A Glucose Foundation Model for Type 1 Diabetes"* (Biomedical Engineering, Universitat de Barcelona / UdG, 2026).

---

## Environment

The project runs inside an NVIDIA TensorFlow Docker container (CUDA 12, RTX 5070). Start it with:

```bash
docker start -ai tvae
docker exec -it tvae bash
cd /mnt/workspace/tvae
```

Install Python dependencies (most are already present in the base image):

```bash
pip install -r requirements.txt
```

Always prefix long-running scripts with `python -u` to get unbuffered output.

---

## Repository structure

```
tvae/
├── main.py                      # Stage 2 CLI entry point
├── requirements.txt
├── src/
│   ├── encoder.py               # Load frozen encoder / decoder weights
│   ├── preprocessing.py         # Windowing, normalisation, Hovorka ODE
│   ├── settings.py              # Global constants
│   └── stage2/
│       ├── data.py              # tf.data pipelines and patient splits
│       ├── models.py            # All Stage 2 model architectures
│       ├── train.py             # Training loop (Huber loss, early stopping)
│       ├── evaluate.py          # Metrics and plots
│       └── apps/
│           ├── imputation.py    # App 1 — zero-shot gap imputation
│           ├── forecasting.py   # App 2 — 2-hour ahead forecasting
│           └── hypo_risk.py     # App 3 — nocturnal hypoglycaemia risk
├── scripts/                     # Standalone pipeline scripts (see below)
├── data/
│   ├── raw/                     # Source parquet files (not tracked — see data/README.md)
│   ├── interim/                 # Merged + filtered parquet
│   └── processed/
│       └── adults_global_norm/  # Per-patient NPZ windows (main track)
└── results/
    ├── mtsm/
    │   ├── encoder_global_norm/ # Pre-trained encoder weights (main track)
    │   └── decoder_global_norm/ # Pre-trained decoder weights (main track)
    ├── outlier_analysis/        # Patient splits + global scalers
    ├── embedding_study_global_norm/
    ├── patient_level_global_norm/
    └── stage2/                  # Per-app run outputs
```

---

## Pipeline

### Stage 0 — Data ingestion and preprocessing

See `data/README.md` for raw data access. Once raw files are in place:

```bash
# 1. Parse T1DEXI raw CSVs → parquet
python -u scripts/parse_t1dexi.py

# 2. Merge METABONET + T1DEXI → combined_filtered.parquet
python -u scripts/merge_datasets.py
python -u scripts/filter_dataset.py

# 3. Window into per-patient NPZ + compute global PI/RA scalers
python -u scripts/outlier_analysis.py
python -u scripts/compute_global_scalers.py
# preprocessing.py is called by the scripts above; it handles windowing,
# Hovorka ODE integration, step-change artifact filtering, and normalisation.
```

Output: `data/processed/adults_global_norm/` — 1,037 `.npz` files, windows of shape `(N, 288, 11)`.  
Patient split files and global scalers are saved to `results/outlier_analysis/`.

### Stage 1 — Pre-training

```bash
# Encoder (BERT-style masked span reconstruction)
python -u scripts/experiment_mtsm.py \
  --run_id encoder_global_norm --data data/processed/adults_global_norm \
  2>&1 | tee results/mtsm/encoder_global_norm_log.txt

# Decoder (GPT-style next-token prediction)
python -u scripts/train_decoder.py \
  --run_id decoder_global_norm --data data/processed/adults_global_norm \
  2>&1 | tee results/mtsm/decoder_global_norm_log.txt
```

Weights are saved to `results/mtsm/{encoder,decoder}_global_norm/`.  
`src/encoder.py` points to these directories — update the paths there if you change run IDs.

### Stage 2 — Downstream tasks

```bash
# App 1 — Gap imputation (zero-shot, no training)
python -u main.py --app imputation --run_id run01 --mode thesis \
  --data data/processed/adults_global_norm \
  2>&1 | tee results/stage2/imputation/run01_log.txt

# App 2 — 2-hour forecasting
python -u main.py --app forecasting --run_id run01 --epochs 50 --mode thesis \
  --data data/processed/adults_global_norm \
  2>&1 | tee results/stage2/forecasting/run01_log.txt

# App 3 — Nocturnal hypoglycaemia risk (bedtime filter, 8h horizon)
python -u main.py --app hypo_risk --run_id run01 --epochs 50 --mode thesis \
  --data data/processed/adults_global_norm \
  2>&1 | tee results/stage2/hypo_risk/run01_log.txt
```

`--mode thesis` runs all model variants: raw LSTM, FM frozen, FM fine-tuned (encoder and decoder).  
`--eval_only` skips training and loads saved weights for inference only.

### Stage 3 — Representation analysis

```bash
# Forward-pass all patients → cache embeddings
python -u scripts/embedding_study.py

# ISF / CR / HbA1c ridge probes on cached embeddings
python -u scripts/patient_level_analysis.py
```

### Stage 4 — Feature importance

```bash
python -u scripts/feature_ablation.py          # zero-out ablation per feature group
python -u scripts/gradient_feature_importance.py
python -u scripts/time_feature_importance.py   # time-of-day gradient saliency
python -u scripts/variable_justification.py    # time-stratified ablation
```

### Stage 5 — Bootstrap CIs and figures

```bash
# Confidence intervals for all Stage 2 tables
python -u scripts/bootstrap_ci.py \
  2>&1 | tee results/stage2/bootstrap_ci.log

# EDA figures
python -u scripts/plot_eda_demographics.py
python -u scripts/plot_eda_glycaemic_profile.py
python -u scripts/plot_eda_physiological_patterns.py
python -u scripts/plot_eda_driver_blindness.py
python -u scripts/plot_eda_preprocessing.py
python -u scripts/plot_eda_tir_by_modality.py

# Result figures
python -u scripts/plot_training_curves.py
python -u scripts/plot_metrics_clean.py
python -u scripts/plot_roc_clean.py
python -u scripts/plot_forecast_examples.py
python -u scripts/plot_imputation_examples_v2.py
python -u scripts/plot_feature_importance.py

# Embedding and patient-level figures
python -u scripts/plot_embeddings.py          # 2D UMAP (umap_2d.png) + 3D interactive
python -u scripts/plot_umap_clean.py          # 2-panel 2D UMAP, GRI quintile (umap_clean.png)
python -u scripts/plot_umap_3d_final.py       # 2×3 grid, 3 viewing angles (umap_3d_2x3.png)
python -u scripts/plot_patient_scatter_2x3.py
```

---

## Key results

| Task | Model | Metric |
|---|---|---|
| Gap imputation (4 h) | FM encoder (zero-shot) | R² = 0.78 vs Raw R² = 0.12 |
| Forecasting t+5 (short horizon) | Raw LSTM | RMSE = 7.94 mg/dL (best short) |
| Forecasting t+120 (long horizon) | Decoder fine-tuned | RMSE = 45.65 mg/dL (best long) |
| Nocturnal hypo risk | Decoder fine-tuned | AUROC = 0.737 (p = 0.004 vs Raw) |
| ISF recovery | Decoder H (ridge) | R² = 0.499 vs CGM stats R² = −0.034 |
| CR recovery | Encoder h_cls (ridge) | R² = 0.406 vs CGM stats R² = −0.008 |

---

## Notes for future work

- **Active feature set:** 7 features (CGM, PI, RA, hour_sin/cos, bolus flag, carbs flag). Therapy modality slots (cols 7–9) are zero-filled — ablation showed Δ ≤ 0.04 and representation probe = chance.
- **Global vs per-patient normalisation:** Global PI/RA z-scoring is the main track. Per-patient norm archives are in `results/mtsm/encoder_clean` and `decoder_clean` for reference.
- **All LSTMs must use `unroll=True`** — cuDNN 9 requires explicit sequence lengths that TF 2.17 doesn't provide; `unroll=False` crashes at runtime regardless of sequence length.
- **GradientTape + XLA incompatibility** — always use `model.fit()`, not a custom GradientTape loop.
