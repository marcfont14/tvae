"""
settings.py
===========
Centralised configuration for the TVAE project.
All hyperparameters and paths are defined here.

Usage:
    from src.settings import Settings
    cfg = Settings()
    print(cfg.model.d_model)          # 256
    print(cfg.preprocessing.window_size)  # 288

    # Override for experiments
    cfg = Settings(model=ModelSettings(d_model=512))
"""

from dataclasses import dataclass, field
from pathlib import Path


# ── Paths ─────────────────────────────────────────────────────────────────────

@dataclass
class PathSettings:
    data_raw:       Path = Path('data/raw')
    data_interim:   Path = Path('data/interim')
    data_processed: Path = Path('data/processed')
    plots:          Path = Path('plots')

    # Key files
    combined_parquet: Path = Path('data/interim/combined_filtered.parquet')
    metabonet_parquet: Path = Path('data/raw/metabonet_public_train.parquet')


# ── Preprocessing ─────────────────────────────────────────────────────────────

@dataclass
class HovorkaPI:
    """Plasma Insulin — 3-compartment ODE (Hovorka model).
    S1 -> S2 -> Ifa (active insulin)
    Same parameters as research group colleague.
    """
    VI:     float = 0.12    # Insulin volume of distribution (L/kg)
    Ke:     float = 0.138   # Insulin elimination rate (1/min)
    tmaxI:  float = 55.0    # Time of max insulin absorption (min)
    dt:     float = 5.0     # Time step (min) — matches CGM resolution
    pi_sign: int  = -1      # Sign convention (negative = insulin lowers glucose)


@dataclass
class HovorkaRA:
    """Rate of Absorption — 2-compartment ODE (Hovorka model).
    D1 -> D2 -> RA (gut absorption)
    Same parameters as research group colleague.
    """
    tau_D: float = 40.0     # Time constant for carb absorption (min)
    A_G:   float = 0.8      # Carb bioavailability fraction (unitless)
    dt:    float = 5.0      # Time step (min)


@dataclass
class PreprocessingSettings:
    # CGM cleaning
    cgm_lower:          float = 39.0    # Physiological lower bound (mg/dL)
    cgm_upper:          float = 400.0   # Physiological upper bound (mg/dL)
    max_gap_points:     int   = 12      # Max consecutive nulls to interpolate (12 x 5min = 1h)

    # Quality filters
    cgm_std_min:        float = 15.0    # Min CGM std to include patient
    cgm_missing_max:    float = 50.0    # Max % missing CGM to include patient

    # Windowing
    window_size:        int   = 288     # Steps per window (288 x 5min = 24h)
    stride:             int   = 72      # Stride between windows (72 x 5min = 6h)
    max_missing_window: float = 0.20    # Max fraction of missing CGM per window

    # Normalisation — z-score per patient (individual, not group)
    scaler: str = 'standard'            # 'standard' (z-score) or 'minmax'

    # Hovorka model parameters
    params_pi: HovorkaPI = field(default_factory=HovorkaPI)
    params_ra: HovorkaRA = field(default_factory=HovorkaRA)

    # Features to compute
    compute_pi: bool = True     # Plasma Insulin via Hovorka
    compute_ra: bool = True     # Rate of Absorption via Hovorka


# ── Model ─────────────────────────────────────────────────────────────────────

@dataclass
class ModelSettings:
    # Transformer encoder
    d_model:    int   = 256     # Embedding dimension
    n_heads:    int   = 8       # Number of attention heads (d_model must be divisible by n_heads)
    n_layers:   int   = 6       # Number of encoder layers
    d_ff:       int   = 1024    # Feed-forward hidden dimension (4 * d_model)
    dropout:    float = 0.1     # Dropout rate

    # Input sequence
    window_size: int  = 288     # Must match PreprocessingSettings.window_size
    n_features: int = 10  # CGM, PI, RA, hour_sin, hour_cos, bolus_logged, carbs_logged, AID, SAP, MDI

    # VAE latent space
    latent_dim:  int  = 64      # Dimension of z

    # Forecasting head (pretext task during pre-training)
    forecast_horizon: int = 48  # Steps to forecast (48 x 5min = 4h)


# ── Training ──────────────────────────────────────────────────────────────────

@dataclass
class TrainingSettings:
    # Optimiser
    batch_size:     int   = 32
    learning_rate:  float = 1e-4
    weight_decay:   float = 1e-2     # AdamW weight decay
    epochs:         int   = 100

    # Learning rate schedule — warmup + cosine decay
    warmup_steps:   int   = 4000
    
    # Checkpointing
    checkpoint_dir: Path  = Path('checkpoints')
    save_every:     int   = 5        # Save checkpoint every N epochs

    # Hardware
    mixed_precision: bool = True     # Use float16 on GPU


# ── Root settings ─────────────────────────────────────────────────────────────

@dataclass
class Settings:
    """
    Root settings object. Import and instantiate this in any script.

    Examples
    --------
    Default config:
        cfg = Settings()

    Override single parameter:
        cfg = Settings(model=ModelSettings(d_model=512))

    Override nested parameter:
        cfg = Settings(preprocessing=PreprocessingSettings(window_size=144))
    """
    paths:          PathSettings          = field(default_factory=PathSettings)
    preprocessing:  PreprocessingSettings = field(default_factory=PreprocessingSettings)
    model:          ModelSettings         = field(default_factory=ModelSettings)
    training:       TrainingSettings      = field(default_factory=TrainingSettings)


if __name__ == '__main__':
    cfg = Settings()
    print(cfg)