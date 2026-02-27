"""
T1DEXI Parser
=============
Reads T1DEXI CSV files and produces a parquet file compatible with
the METABONET schema:
    id, date, CGM, bolus, basal, carbs, insulin, heartrate,
    age, gender, age_of_diagnosis, insulin_delivery_modality

Usage:
    python scripts/parse_t1dexi.py \
        --input  data/raw/T1DEXI/ \
        --output data/raw/t1dexi_parsed.parquet
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ── Modality mapping (replicates categorize_insulin_delivery_modality) ────────

def categorize_modality(device: str) -> str:
    if pd.isna(device) or device == '':
        return None
    d = str(device).upper()
    if 'MULTIPLE DAILY INJECTIONS' in d:
        return 'MDI'
    if 'OMNIPOD' in d and 'OMNIPOD 5' not in d:
        return 'SAP'
    if ('770G' in d and 'MANUAL' in d) or ('670G' in d and 'MANUAL' in d):
        return 'SAP'
    for model in ['630G', '640G', '551', '530G', '751', '522', '523', '723']:
        if model in d:
            return 'SAP'
    if 'PARADIGM' in d:
        return 'SAP'
    return 'AID'  # everything else: Omnipod 5, Tandem Control-IQ, etc.


# ── Per-table loaders ─────────────────────────────────────────────────────────

def load_cgm(path: Path) -> pd.DataFrame:
    """LB.csv → id, date, CGM"""
    df = pd.read_csv(path, encoding='latin-1',
                     usecols=['USUBJID', 'LBTESTCD', 'LBSTRESN', 'LBDTC'])
    df = df[df['LBTESTCD'] == 'GLUC'].copy()
    df['date'] = pd.to_datetime(df['LBDTC'], format='mixed')
    df['CGM'] = pd.to_numeric(df['LBSTRESN'], errors='coerce')
    df = df.rename(columns={'USUBJID': 'id'})[['id', 'date', 'CGM']]
    return df


def load_insulin(path: Path) -> pd.DataFrame:
    """FACM.csv → id, date, bolus, basal"""
    df = pd.read_csv(path, encoding='latin-1',
                     usecols=['USUBJID', 'FATESTCD', 'FASTRESN', 'FADTC',
                               'FADUR', 'INSSTYPE', 'INSNMBOL', 'INSEXBOL'])
    df['date'] = pd.to_datetime(df['FADTC'], format='mixed')
    df['dose'] = pd.to_numeric(df['FASTRESN'], errors='coerce')
    df['duration'] = pd.to_timedelta(df['FADUR'], errors='coerce')
    df = df.rename(columns={'USUBJID': 'id'})

    # ── Bolus ──────────────────────────────────────────────────────────────────
    df_bolus = df[df['FATESTCD'] != 'INSBASAL'].copy()

    # Override dose with delivered bolus for extended boluses
    mask = df_bolus['INSNMBOL'].notna()
    df_bolus.loc[mask, 'dose'] = pd.to_numeric(df_bolus.loc[mask, 'INSNMBOL'], errors='coerce')

    # Square boluses: immediate dose = 0, all extended
    df_bolus.loc[df_bolus['INSSTYPE'] == 'square', 'dose'] = 0.0

    # Spread extended bolus across 5-min intervals
    ext_mask = df_bolus['INSEXBOL'].notna() & (df_bolus['INSEXBOL'] != 0)
    ext = df_bolus[ext_mask].copy()
    ext['dose'] = pd.to_numeric(ext['INSEXBOL'], errors='coerce')

    bolus_rows = []

    # Immediate bolus rows (non-extended part)
    imm = df_bolus[~ext_mask][['id', 'date', 'dose']].copy()
    imm = imm[imm['dose'].notna() & (imm['dose'] > 0)]
    imm = imm.rename(columns={'dose': 'bolus'})
    bolus_rows.append(imm)

    # Extended bolus rows — split across duration
    for _, row in ext.iterrows():
        dur = row['duration']
        if pd.isna(dur) or dur.total_seconds() <= 0:
            n = 1
        else:
            n = max(1, round(dur.total_seconds() / 300))
        dose_per_step = row['dose'] / n
        for i in range(n):
            bolus_rows.append(pd.DataFrame([{
                'id':    row['id'],
                'date':  row['date'] + pd.Timedelta(minutes=5 * i),
                'bolus': dose_per_step,
            }]))

    df_bolus_final = pd.concat(bolus_rows, ignore_index=True)

    # ── Basal ──────────────────────────────────────────────────────────────────
    df_basal = df[df['FATESTCD'] == 'INSBASAL'][['id', 'date', 'dose', 'duration']].copy()
    df_basal = df_basal.sort_values(['id', 'date']).reset_index(drop=True)

    # Fill duration from gap to next record for same patient
    df_basal['next_date'] = df_basal['date'].shift(-1)
    df_basal['next_id']   = df_basal['id'].shift(-1)
    df_basal['duration']  = df_basal.apply(
        lambda r: (r['next_date'] - r['date'])
        if r['id'] == r['next_id'] else r['duration'],
        axis=1
    )

    basal_rows = []
    for _, row in df_basal.iterrows():
        dur = row['duration']
        if pd.isna(dur) or (hasattr(dur, 'total_seconds') and dur.total_seconds() <= 0):
            n = 1
        else:
            n = max(1, round(dur.total_seconds() / 300))
        dose_per_step = row['dose'] / n if pd.notna(row['dose']) else np.nan
        for i in range(n):
            basal_rows.append({
                'id':    row['id'],
                'date':  row['date'] + pd.Timedelta(minutes=5 * i),
                'basal': dose_per_step,
            })

    df_basal_final = pd.DataFrame(basal_rows)

    return df_bolus_final, df_basal_final


def load_carbs(path: Path) -> pd.DataFrame:
    """ML.csv → id, date, carbs"""
    df = pd.read_csv(path, encoding='latin-1',
                     usecols=['USUBJID', 'MLDTC', 'MLDOSE', 'MLDOSU'])
    df = df[df['MLDOSU'] == 'g'].copy()   # keep only gram entries
    df['date'] = pd.to_datetime(df['MLDTC'], format='mixed')
    df['carbs'] = pd.to_numeric(df['MLDOSE'], errors='coerce')
    df = df.rename(columns={'USUBJID': 'id'})[['id', 'date', 'carbs']]
    # Aggregate multiple food items at same timestamp
    df = df.groupby(['id', 'date'], as_index=False)['carbs'].sum()
    return df


def load_demographics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='latin-1',
                     usecols=['USUBJID', 'AGE', 'SEX'])
    df = df.rename(columns={'USUBJID': 'id', 'AGE': 'age', 'SEX': 'gender'})
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['gender'] = df['gender'].map({'M': 'Male', 'F': 'Female'})
    df['id'] = df['id'].astype(str)   # sin T_
    return df[['id', 'age', 'gender']].drop_duplicates('id')


def load_modality(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='latin-1',
                     usecols=['USUBJID', 'DXTRT'])
    df = df.rename(columns={'USUBJID': 'id'})
    df['insulin_delivery_modality'] = df['DXTRT'].apply(categorize_modality)
    df['id'] = df['id'].astype(str)   # sin T_
    df = (df.groupby('id')['insulin_delivery_modality']
            .agg(lambda x: x.mode()[0] if len(x) > 0 else None)
            .reset_index())
    return df


# ── Resampler ─────────────────────────────────────────────────────────────────

def resample_to_5min(df_cgm, df_bolus, df_basal, df_carbs) -> pd.DataFrame:
    """
    For each patient: build a 5-min grid from CGM timestamps,
    merge bolus/basal/carbs onto it.
    """
    processed = []

    subject_ids = df_cgm['id'].unique()
    print(f"Resampling {len(subject_ids)} patients...")

    for i, pid in enumerate(subject_ids):
        cgm = df_cgm[df_cgm['id'] == pid].set_index('date')[['CGM']]
        cgm = cgm.resample('5min').mean()

        # Check CGM present
        if cgm['CGM'].notna().sum() == 0:
            print(f"  Skip {pid} — no CGM data")
            continue

        # Bolus
        bol = df_bolus[df_bolus['id'] == pid].set_index('date')[['bolus']]
        if len(bol) > 0:
            bol = bol.resample('5min').sum()
            bol['bolus'] = bol['bolus'].replace(0, np.nan)
        
        # Basal
        bas = df_basal[df_basal['id'] == pid].set_index('date')[['basal']]
        if len(bas) > 0:
            bas = bas.resample('5min').sum()

        # Carbs
        car = df_carbs[df_carbs['id'] == pid].set_index('date')[['carbs']]
        if len(car) > 0:
            car = car.resample('5min').sum()
            car['carbs'] = car['carbs'].replace(0, np.nan)

        # Merge everything onto CGM grid
        df_pat = cgm.copy()
        for sub, col in [(bol, 'bolus'), (bas, 'basal'), (car, 'carbs')]:
            if len(sub) > 0:
                df_pat = df_pat.join(sub, how='left')
            else:
                df_pat[col] = np.nan

        # Ensure 5-min grid is uniform
        df_pat = df_pat.resample('5min').asfreq()
        df_pat['id'] = str(pid)
        df_pat = df_pat.reset_index().rename(columns={'index': 'date'})

        processed.append(df_pat)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(subject_ids)} done")

    df_out = pd.concat(processed, ignore_index=True)
    return df_out


# ── Main ──────────────────────────────────────────────────────────────────────

def main(input_dir: str, output_path: str):
    p = Path(input_dir)
    print("Loading CSVs...")

    df_cgm        = load_cgm(p / 'LB.csv')
    df_bolus, df_basal = load_insulin(p / 'FACM.csv')
    df_carbs      = load_carbs(p / 'ML.csv')
    df_demo       = load_demographics(p / 'DM.csv')
    df_modality   = load_modality(p / 'DX.csv')

    print(f"  CGM rows:    {len(df_cgm)}")
    print(f"  Bolus rows:  {len(df_bolus)}")
    print(f"  Basal rows:  {len(df_basal)}")
    print(f"  Carbs rows:  {len(df_carbs)}")
    print(f"  Patients:    {df_cgm['id'].nunique()}")

    # Resample to 5-min grid
    df = resample_to_5min(df_cgm, df_bolus, df_basal, df_carbs)

    # Add insulin = bolus + basal
    df['insulin'] = df['bolus'].fillna(0) + df['basal'].fillna(0)
    df['insulin'] = df['insulin'].replace(0, np.nan)

    # Add demographics
    df = df.merge(df_demo, on='id', how='left')
    df = df.merge(df_modality, on='id', how='left')

    # Add missing columns to match METABONET schema
    df['heartrate']        = np.nan
    df['age_of_diagnosis'] = np.nan

    # Enforce column order and types
    cols = ['id', 'date', 'CGM', 'bolus', 'basal', 'carbs', 'insulin',
            'heartrate', 'age', 'gender', 'age_of_diagnosis',
            'insulin_delivery_modality']
    df = df[cols]
    df['id']     = df['id'].astype(str)
    df['date']   = pd.to_datetime(df['date'])
    df['gender'] = df['gender'].astype(str)
    df['insulin_delivery_modality'] = df['insulin_delivery_modality'].astype(str)

    # Offset patient IDs to avoid collision with METABONET (which goes up to ~1210)
    # T1DEXI IDs start at 100 — prefix with 'T_' to make them unambiguous
    df['id'] = 'T_' + df['id']

    print(f"\nFinal dataset:")
    print(f"  Rows:     {len(df)}")
    print(f"  Patients: {df['id'].nunique()}")
    print(f"  Columns:  {df.columns.tolist()}")
    print(f"  Modality: {df.drop_duplicates('id')['insulin_delivery_modality'].value_counts().to_dict()}")
    print(df.head(3))

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='Path to T1DEXI CSV folder')
    parser.add_argument('--output', required=True, help='Output parquet path')
    args = parser.parse_args()
    main(args.input, args.output)