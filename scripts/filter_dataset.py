"""
filter_dataset.py
=================
Generates data/raw/metabonet_train_filtered.parquet from the original
data/raw/metabonet_public_train.parquet file.

Filtering criteria:
    - Patients with >= MIN_BOLUS bolus events (bolus > 0)
    - Patients with >= MIN_CARBS carb events  (carbs > 0)

Columns kept:
    id, date, CGM, bolus, basal, carbs, insulin,
    age, gender, age_of_diagnosis, insulin_delivery_modality

Usage:
    python scripts/filter_dataset.py
    python scripts/filter_dataset.py --input path/to/input.parquet
                                     --output path/to/output.parquet
                                     --min-bolus 10 --min-carbs 10
"""

import argparse
import os
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_INPUT  = 'data/raw/metabonet_public_train.parquet'
DEFAULT_OUTPUT = 'data/raw/metabonet_train_filtered.parquet'
MIN_BOLUS      = 10
MIN_CARBS      = 10
BATCH_SIZE     = 1_000_000

KEEP_COLS = [
    'id', 'date', 'CGM', 'bolus', 'basal', 'carbs', 'insulin',
    'age', 'gender', 'age_of_diagnosis', 'insulin_delivery_modality',
]


def parse_args():
    p = argparse.ArgumentParser(description='Filter METABONET dataset')
    p.add_argument('--input',      default=DEFAULT_INPUT)
    p.add_argument('--output',     default=DEFAULT_OUTPUT)
    p.add_argument('--min-bolus',  type=int, default=MIN_BOLUS)
    p.add_argument('--min-carbs',  type=int, default=MIN_CARBS)
    return p.parse_args()


def find_valid_patients(input_path, min_bolus, min_carbs):
    """Pass 1 — count bolus and carb events per patient."""
    print(f"Pass 1: counting driver events per patient...")
    f = pq.ParquetFile(input_path)

    counts = defaultdict(lambda: {'bolus': 0, 'carbs': 0})
    total_rows = 0

    for batch in f.iter_batches(columns=['id', 'bolus', 'carbs'],
                                batch_size=BATCH_SIZE):
        df = batch.to_pandas()
        total_rows += len(df)

        for pid, grp in df.groupby('id'):
            counts[pid]['bolus'] += (grp['bolus'] > 0).sum()
            counts[pid]['carbs'] += (grp['carbs'] > 0).sum()

    total_patients = len(counts)
    valid = {
        pid for pid, c in counts.items()
        if c['bolus'] >= min_bolus and c['carbs'] >= min_carbs
    }

    print(f"  Total rows      : {total_rows:,}")
    print(f"  Total patients  : {total_patients:,}")
    print(f"  Valid patients  : {len(valid):,}  "
          f"(>={min_bolus} bolus AND >={min_carbs} carbs)")
    print(f"  Excluded        : {total_patients - len(valid):,}")

    return valid


def write_filtered(input_path, output_path, valid_ids):
    """Pass 2 — write only valid patient rows to output parquet."""
    print(f"\nPass 2: writing filtered dataset...")
    f      = pq.ParquetFile(input_path)
    writer = None
    rows_written = 0

    # Validate that all requested columns exist
    available = set(pq.read_schema(input_path).names)
    keep = [c for c in KEEP_COLS if c in available]
    missing = [c for c in KEEP_COLS if c not in available]
    if missing:
        print(f"  Warning: columns not found and skipped: {missing}")

    for batch in f.iter_batches(columns=keep, batch_size=BATCH_SIZE):
        table = batch.to_pydict()
        # Filter rows
        ids   = table['id']
        mask  = [i for i, pid in enumerate(ids) if pid in valid_ids]
        if not mask:
            continue

        filtered = {col: [vals[i] for i in mask] for col, vals in table.items()}
        out_table = pa.table(filtered)

        if writer is None:
            writer = pq.ParquetWriter(output_path, out_table.schema)

        writer.write_table(out_table)
        rows_written += len(mask)
        print(f"  Rows written so far: {rows_written:,}", end='\r')

    if writer:
        writer.close()

    print(f"\n  Done. Total rows written: {rows_written:,}")
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Output file size: {size_mb:.0f} MB")
    print(f"  Saved to: {output_path}")


def main():
    args = parse_args()

    print("=" * 60)
    print("METABONET Dataset Filter")
    print("=" * 60)
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")
    print(f"  Filter : >= {args.min_bolus} bolus events AND "
          f">= {args.min_carbs} carb events")
    print()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    valid_ids = find_valid_patients(args.input, args.min_bolus, args.min_carbs)
    write_filtered(args.input, args.output, valid_ids)

    print("\n✓ Filtering complete.")


if __name__ == '__main__':
    main()