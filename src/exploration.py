"""
metabonet_exploration.py  [v3 — memory efficient]
==================================================
Carga eficiente en dos pasos:
  1. Carga ultra-ligera para identificar pacientes con datos de drivers
  2. Carga selectiva solo de esos pacientes y columnas relevantes

Uso:
    python metabonet_exploration.py \
        --train raw_data/metabonet_public_train.parquet \
        --test  raw_data/metabonet_public_test.parquet \
        --output plots/
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
CGM_MIN, CGM_MAX    = 40, 400
HYPO_THRESHOLD      = 70
HYPER_THRESHOLD     = 180
TARGET_LOW          = 70
TARGET_HIGH         = 140

# Columnas finales a cargar tras el filtrado
LOAD_COLS = [
    'id', 'date', 'CGM',
    'bolus', 'basal', 'insulin',  # drivers insulina
    'carbs',                       # driver carbohidratos
    'heartrate', 'steps',          # contexto fisiológico
    'age', 'gender',               # demografía básica
    'age_of_diagnosis', 'treatment_group',
    'insulin_delivery_modality',   # MDI vs bomba — relevante para IOB
]

# Umbral mínimo de eventos para considerar un paciente válido
MIN_BOLUS_EVENTS = 10
MIN_CARB_EVENTS  = 10


# ─────────────────────────────────────────────
# PASO 1: FILTRADO DE PACIENTES
# ─────────────────────────────────────────────
def get_valid_patient_ids(path: Path, label: str) -> list:
    """
    Carga solo id + drivers para identificar pacientes con datos suficientes.
    Muy ligero en memoria.
    """
    print(f"\n  [{label}] Paso 1: identificando pacientes validos...")

    import pyarrow.parquet as pq
    all_cols = pq.read_schema(path).names
    print(f"  Columnas en archivo: {all_cols}")

    filter_cols = [c for c in ['id', 'bolus', 'basal', 'carbs', 'insulin']
                   if c in all_cols]
    df_filter = pd.read_parquet(path, columns=filter_cols)

    mem = df_filter.memory_usage(deep=True).sum() / 1024**2
    print(f"  Carga ligera: {df_filter.shape[0]:,} filas, {mem:.1f} MB")

    # Verificar relacion entre bolus, basal e insulin
    if all(c in df_filter.columns for c in ['bolus', 'basal', 'insulin']):
        print(f"\n  Inspeccionando relacion bolus + basal vs insulin...")
        sample = df_filter[df_filter['insulin'] > 0].head(10000)
        if len(sample) > 0:
            derived = sample['bolus'].fillna(0) + sample['basal'].fillna(0)
            corr = derived.corr(sample['insulin'])
            match_pct = 100 * (np.abs(derived - sample['insulin']) < 0.01).mean()
            print(f"    Correlacion (bolus+basal) vs insulin : {corr:.4f}")
            print(f"    Filas donde bolus+basal == insulin   : {match_pct:.1f}%")
            print(f"    -> {'insulin = bolus + basal (columna derivada)' if match_pct > 80 else 'insulin es independiente de bolus+basal'}")

    # Contar eventos por paciente
    agg = df_filter.groupby('id').agg(
        bolus_events  = ('bolus',  lambda x: (x > 0).sum()),
        carb_events   = ('carbs',  lambda x: (x > 0).sum()),
        total_records = ('bolus',  'count'),
    ).reset_index()

    # Pacientes con datos suficientes
    valid = agg[
        (agg['bolus_events'] >= MIN_BOLUS_EVENTS) &
        (agg['carb_events']  >= MIN_CARB_EVENTS)
    ]

    print(f"\n  Pacientes totales              : {len(agg):,}")
    print(f"  Pacientes con >={MIN_BOLUS_EVENTS} eventos bolus  : {(agg['bolus_events'] >= MIN_BOLUS_EVENTS).sum():,}")
    print(f"  Pacientes con >={MIN_CARB_EVENTS} eventos carbs   : {(agg['carb_events']  >= MIN_CARB_EVENTS).sum():,}")
    print(f"  Pacientes validos (ambos)      : {len(valid):,}")
    print(f"\n  Distribucion eventos bolus (todos los pacientes):")
    print(f"    Mediana: {agg['bolus_events'].median():.0f}  Max: {agg['bolus_events'].max()}")
    print(f"  Distribucion eventos carbs:")
    print(f"    Mediana: {agg['carb_events'].median():.0f}  Max: {agg['carb_events'].max()}")

    del df_filter
    return valid['id'].tolist(), agg


# ─────────────────────────────────────────────
# PASO 2: CARGA SELECTIVA
# ─────────────────────────────────────────────
def load_filtered(path: Path, valid_ids: list, label: str) -> pd.DataFrame:
    """Carga solo columnas necesarias y solo pacientes válidos."""
    print(f"\n  [{label}] Paso 2: carga selectiva ({len(valid_ids)} pacientes)...")

    import pyarrow.parquet as pq
    all_cols = pq.read_schema(path).names
    cols = [c for c in LOAD_COLS if c in all_cols]
    print(f"  Columnas a cargar: {cols}")

    df = pd.read_parquet(path, columns=cols)
    df = df[df['id'].isin(valid_ids)].copy()

    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    if 'id' in df.columns and 'date' in df.columns:
        df = df.sort_values(['id', 'date']).reset_index(drop=True)

    mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"  Shape final: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"  Memoria    : {mem:.1f} MB")
    return df


# ─────────────────────────────────────────────
# ANÁLISIS
# ─────────────────────────────────────────────
def section_cgm(df: pd.DataFrame, label: str):
    print(f"\n{'─'*60}")
    print(f"  [{label}] CGM")
    print(f"{'─'*60}")

    cgm   = df['CGM']
    total = len(cgm)
    missing      = cgm.isna().sum()
    out_of_range = (~cgm.between(CGM_MIN, CGM_MAX) & cgm.notna()).sum()

    print(f"  Total registros           : {total:,}")
    print(f"  Missing (NaN)             : {missing:,}  ({100*missing/total:.1f}%)")
    print(f"  Fuera de rango clinico    : {out_of_range:,}  ({100*out_of_range/total:.1f}%)")

    cgm_clean = cgm.dropna()
    cgm_clean = cgm_clean[cgm_clean.between(CGM_MIN, CGM_MAX)]
    if len(cgm_clean) > 0:
        tir = 100 * cgm_clean.between(TARGET_LOW, TARGET_HIGH).mean()
        tbr = 100 * (cgm_clean < HYPO_THRESHOLD).mean()
        tar = 100 * (cgm_clean > HYPER_THRESHOLD).mean()
        print(f"\n  Estadisticas:")
        print(f"    Media   : {cgm_clean.mean():.1f}  Std: {cgm_clean.std():.1f}  "
              f"Rango: [{cgm_clean.min():.0f}, {cgm_clean.max():.0f}]")
        print(f"    TIR ({TARGET_LOW}-{TARGET_HIGH}): {tir:.1f}%  "
              f"TBR: {tbr:.1f}%  TAR: {tar:.1f}%")

    pm = df.groupby('id')['CGM'].agg(
        total='count',
        missing=lambda x: x.isna().sum()
    )
    pm['missing_pct'] = 100 * pm['missing'] / pm['total']
    print(f"\n  Missing CGM por paciente:")
    print(f"    Mediana : {pm['missing_pct'].median():.1f}%")
    print(f"    >20% NaN: {(pm['missing_pct'] > 20).sum()} pacientes")
    print(f"    >50% NaN: {(pm['missing_pct'] > 50).sum()} pacientes")


def section_drivers(df: pd.DataFrame, label: str):
    print(f"\n{'─'*60}")
    print(f"  [{label}] DRIVERS")
    print(f"{'─'*60}")

    total = len(df)
    for col in ['bolus', 'basal', 'insulin', 'carbs']:
        if col not in df.columns:
            continue
        s       = df[col]
        missing = s.isna().sum()
        nonzero = (s > 0).sum()
        print(f"\n  [{col.upper()}]")
        print(f"    Missing          : {missing:,} ({100*missing/total:.1f}%)")
        print(f"    Eventos reales>0 : {nonzero:,} ({100*nonzero/total:.1f}%)")
        if nonzero > 0:
            vals = s[s > 0]
            print(f"    Media / Mediana  : {vals.mean():.3f} / {vals.median():.3f}")
            print(f"    Maximo           : {vals.max():.3f}")


def section_per_patient(df: pd.DataFrame, label: str):
    print(f"\n{'─'*60}")
    print(f"  [{label}] POR PACIENTE")
    print(f"{'─'*60}")

    records = df.groupby('id').size()
    print(f"  Registros (5-min) por paciente:")
    print(f"    Media / Mediana : {records.mean():.0f} / {records.median():.0f}")
    print(f"    Min / Max       : {records.min()} / {records.max()}")

    for window, lbl_w in [(288, '24h'), (1008, '3.5d'), (2016, '1sem')]:
        n = (records >= window).sum()
        pct = 100 * n / len(records)
        print(f"    >={window} registros ({lbl_w}): {n} pacientes ({pct:.0f}%)")

    if 'date' in df.columns:
        durations = df.groupby('id')['date'].agg(lambda x: (x.max()-x.min()).days)
        print(f"\n  Duracion dias por paciente:")
        print(f"    Media / Mediana : {durations.mean():.0f} / {durations.median():.0f}")
        print(f"    >30 dias: {(durations > 30).sum()}  >90 dias: {(durations > 90).sum()}")


def section_usable_windows(df: pd.DataFrame, label: str,
                            window_size: int = 288,
                            min_cgm_coverage: float = 0.8):
    print(f"\n{'─'*60}")
    print(f"  [{label}] VENTANAS UTILIZABLES")
    print(f"     window={window_size} ({window_size*5/60:.0f}h), CGM>={min_cgm_coverage*100:.0f}%")
    print(f"{'─'*60}")

    stride = window_size // 4
    counts = []
    for pid, group in df.groupby('id'):
        cgm_vals = group['CGM'].values
        n = len(cgm_vals)
        c = sum(
            1 for start in range(0, n - window_size + 1, stride)
            if np.mean(~np.isnan(cgm_vals[start:start+window_size])) >= min_cgm_coverage
        )
        counts.append(c)

    total = sum(counts)
    print(f"  Total ventanas usables    : {total:,}")
    print(f"  Pacientes sin ventanas    : {sum(1 for c in counts if c == 0)}")
    print(f"  Mediana ventanas/paciente : {np.median(counts):.0f}")
    print(f"  Max en un paciente        : {max(counts)}")
    for bs in [32, 64]:
        print(f"  Batches posibles (bs={bs})  : {total // bs:,}")


def section_compare(train: pd.DataFrame, test: pd.DataFrame):
    print(f"\n{'─'*60}")
    print(f"  COMPARACION TRAIN vs TEST")
    print(f"{'─'*60}")
    for col in ['CGM', 'bolus', 'carbs']:
        if col not in train.columns:
            continue
        tr = train[col].dropna()
        te = test[col].dropna() if col in test.columns else pd.Series(dtype=float)
        print(f"  {col}: train media={tr.mean():.2f} std={tr.std():.2f} | "
              f"test media={te.mean():.2f} std={te.std():.2f}")

    if 'id' in train.columns and 'id' in test.columns:
        overlap = set(train['id'].unique()) & set(test['id'].unique())
        print(f"\n  Pacientes train : {train['id'].nunique()}")
        print(f"  Pacientes test  : {test['id'].nunique()}")
        if overlap:
            print(f"  SOLAPAMIENTO    : {len(overlap)} pacientes — REVISAR")
        else:
            print(f"  Sin solapamiento train/test — OK")


# ─────────────────────────────────────────────
# VISUALIZACIONES
# ─────────────────────────────────────────────
def section_plots(train: pd.DataFrame, test: pd.DataFrame,
                  agg_all: pd.DataFrame, output_dir: Path):
    print(f"\n{'─'*60}")
    print("  GENERANDO VISUALIZACIONES")
    print(f"{'─'*60}")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("METABONET — Exploracion (pacientes con bolus+carbs)",
                 fontsize=15, fontweight='bold')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1: Distribucion CGM
    ax = fig.add_subplot(gs[0, 0])
    cgm = train['CGM'].dropna()
    cgm = cgm[cgm.between(CGM_MIN, CGM_MAX)]
    ax.hist(cgm, bins=80, color='steelblue', alpha=0.85, edgecolor='none')
    ax.axvline(HYPO_THRESHOLD,  color='red',    linestyle='--', lw=1.5, label=f'Hipo {HYPO_THRESHOLD}')
    ax.axvline(HYPER_THRESHOLD, color='orange', linestyle='--', lw=1.5, label=f'Hiper {HYPER_THRESHOLD}')
    ax.set_xlabel('CGM (mg/dL)')
    ax.set_title('Distribucion CGM — Train')
    ax.legend(fontsize=7)

    # 2: Eventos bolus y carbs por paciente
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(agg_all['bolus_events'], agg_all['carb_events'],
               alpha=0.4, s=10, color='steelblue')
    ax.axvline(MIN_BOLUS_EVENTS, color='red', linestyle='--', lw=1)
    ax.axhline(MIN_CARB_EVENTS,  color='red', linestyle='--', lw=1)
    ax.set_xlabel('Eventos bolus por paciente')
    ax.set_ylabel('Eventos carbs por paciente')
    ax.set_title('Eventos driver por paciente\n(rojo=umbral minimo)')

    # 3: Registros por paciente
    ax = fig.add_subplot(gs[0, 2])
    records = train.groupby('id').size()
    ax.hist(records, bins=40, color='mediumseagreen', alpha=0.85, edgecolor='none')
    ax.axvline(288,  color='blue',   linestyle='--', lw=1.5, label='24h')
    ax.axvline(2016, color='purple', linestyle='--', lw=1.5, label='1sem')
    ax.set_xlabel('Registros/paciente')
    ax.set_title('Duracion por paciente')
    ax.legend(fontsize=7)

    # 4: Distribucion bolus
    ax = fig.add_subplot(gs[1, 0])
    if 'bolus' in train.columns:
        ev = train[train['bolus'] > 0]['bolus'].dropna()
        if len(ev) > 0:
            ax.hist(ev.clip(upper=ev.quantile(0.99)), bins=60,
                    color='coral', alpha=0.85, edgecolor='none')
            ax.set_xlabel('Bolus (U)')
            ax.set_title(f'Distribucion bolus (n={len(ev):,})')

    # 5: Distribucion carbs
    ax = fig.add_subplot(gs[1, 1])
    if 'carbs' in train.columns:
        ev = train[train['carbs'] > 0]['carbs'].dropna()
        if len(ev) > 0:
            ax.hist(ev.clip(upper=ev.quantile(0.99)), bins=60,
                    color='gold', alpha=0.85, edgecolor='none')
            ax.set_xlabel('Carbohidratos (g)')
            ax.set_title(f'Distribucion carbs (n={len(ev):,})')

    # 6: Perfil CGM ejemplo
    ax = fig.add_subplot(gs[1, 2])
    if 'date' in train.columns:
        best_pid = train.groupby('id')['CGM'].count().idxmax()
        pdata = train[train['id'] == best_pid].sort_values('date').head(2016)
        ax.plot(pdata['date'], pdata['CGM'], color='steelblue', lw=0.8)
        ax.axhline(HYPO_THRESHOLD,  color='red',    linestyle='--', lw=1, alpha=0.7)
        ax.axhline(HYPER_THRESHOLD, color='orange', linestyle='--', lw=1, alpha=0.7)
        ax.fill_between(pdata['date'], TARGET_LOW, TARGET_HIGH,
                        alpha=0.1, color='green', label='Target')
        ax.set_ylabel('CGM (mg/dL)')
        ax.set_title('Perfil CGM — paciente ejemplo')
        ax.tick_params(axis='x', rotation=30, labelsize=7)
        ax.legend(fontsize=7)

    # 7: Train vs Test CGM
    ax = fig.add_subplot(gs[2, 0])
    for df_part, lbl, color in [(train, 'Train', 'steelblue'), (test, 'Test', 'coral')]:
        if 'CGM' not in df_part.columns:
            continue
        vals = df_part['CGM'].dropna()
        vals = vals[vals.between(CGM_MIN, CGM_MAX)]
        ax.hist(vals, bins=60, alpha=0.5, color=color, label=lbl, density=True)
    ax.set_xlabel('CGM (mg/dL)')
    ax.set_title('CGM: Train vs Test')
    ax.legend()

    # 8: Missing data en columnas clave
    ax = fig.add_subplot(gs[2, 1])
    cols_check = [c for c in ['CGM', 'bolus', 'basal', 'insulin', 'carbs',
                               'heartrate', 'steps'] if c in train.columns]
    mp = [100 * train[c].isna().mean() for c in cols_check]
    colors_bar = ['#e74c3c' if p > 50 else '#f39c12' if p > 20 else '#2ecc71' for p in mp]
    ax.barh(cols_check, mp, color=colors_bar)
    ax.set_xlabel('% Missing')
    ax.set_title('Missing por columna (Train)')

    # 9: Basal vs Bolus scatter
    ax = fig.add_subplot(gs[2, 2])
    if 'bolus' in train.columns and 'basal' in train.columns:
        mask = (train['bolus'] > 0) | (train['basal'] > 0)
        sample = train[mask].sample(min(5000, mask.sum()), random_state=42)
        ax.scatter(sample['basal'].fillna(0), sample['bolus'].fillna(0),
                   alpha=0.3, s=5, color='steelblue')
        ax.set_xlabel('Basal (U/5min)')
        ax.set_ylabel('Bolus (U)')
        ax.set_title('Basal vs Bolus')

    out_path = output_dir / 'metabonet_exploration.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Guardado: {out_path}")
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',  required=True)
    parser.add_argument('--test',   required=True)
    parser.add_argument('--output', default='plots')
    parser.add_argument('--window', type=int, default=288)
    args = parser.parse_args()

    train_path = Path(args.train)
    test_path  = Path(args.test)
    output_dir = Path(args.output)

    print("\n" + "="*60)
    print("  METABONET EXPLORATION v3 — memory efficient")
    print("="*60)

    # TRAIN
    valid_ids_train, agg_train = get_valid_patient_ids(train_path, 'TRAIN')
    train = load_filtered(train_path, valid_ids_train, 'TRAIN')

    section_cgm(train, 'TRAIN')
    section_drivers(train, 'TRAIN')
    section_per_patient(train, 'TRAIN')
    section_usable_windows(train, 'TRAIN', window_size=args.window)

    # TEST — solo resumen, sin analizar en profundidad
    valid_ids_test, _ = get_valid_patient_ids(test_path, 'TEST')
    test = load_filtered(test_path, valid_ids_test, 'TEST')
    section_cgm(test, 'TEST')

    # Comparacion
    section_compare(train, test)

    # Plots
    section_plots(train, test, agg_train, output_dir)

    print(f"\n{'='*60}")
    print("  EXPLORACION COMPLETA")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
