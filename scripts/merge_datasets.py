import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

writer = None
total_patients = 0
total_rows = 0

def write_source(path, source_name):
    global writer, total_patients, total_rows
    print(f"Loading {source_name}...")
    
    if path.endswith('.parquet'):
        pf = pq.ParquetFile(path)
        patients = set()
        rows = 0
        for batch in pf.iter_batches(batch_size=500_000):
            df = batch.to_pandas()
            df['id'] = df['id'].astype(str)
            df['age'] = df['age'].astype(float)  
            df['source_file'] = source_name
            patients.update(df['id'].unique())
            rows += len(df)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    "data/interim/combined_filtered.parquet", table.schema)
            writer.write_table(table)
        print(f"  {source_name}: {len(patients)} patients, {rows:,} rows")
        return len(patients), rows

# 1. METABONET
meta_p, meta_r = write_source(
    "data/raw/metabonet_train_filtered.parquet", "METABONET")

# 2. T1DEXI adultos
t1a_p, t1a_r = write_source(
    "data/raw/t1dexi_parsed.parquet", "T1DEXI_adults")

# 3. T1DEXI pediátricos
t1p_p, t1p_r = write_source(
    "data/raw/t1dexi_p_parsed.parquet", "T1DEXI_pediatric")

writer.close()

print(f"\nCombined dataset:")
print(f"  METABONET:         {meta_p} patients, {meta_r:,} rows")
print(f"  T1DEXI adults:     {t1a_p} patients, {t1a_r:,} rows")
print(f"  T1DEXI pediatric:  {t1p_p} patients, {t1p_r:,} rows")
print(f"  TOTAL:             {meta_p+t1a_p+t1p_p} patients, "
      f"{meta_r+t1a_r+t1p_r:,} rows")
print("Saved to data/interim/combined_filtered.parquet")