import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

print("Loading T1DEXI...")
t1dexi = pd.read_parquet("data/raw/t1dexi_parsed.parquet")
t1dexi['id'] = t1dexi['id'].astype(str)
t1dexi['age'] = t1dexi['age'].astype(float)
t1dexi['source_file'] = 'T1DEXI'
print(f"  T1DEXI: {t1dexi['id'].nunique()} patients, {len(t1dexi)} rows")

print("Merging with METABONET in batches...")
pf = pq.ParquetFile("data/raw/metabonet_train_filtered.parquet")

writer = None
meta_patients = set()
meta_rows = 0

for batch in pf.iter_batches(batch_size=500_000):
    df_batch = batch.to_pandas()
    df_batch['id'] = df_batch['id'].astype(str)
    df_batch['source_file'] = 'METABONET'
    meta_patients.update(df_batch['id'].unique())
    meta_rows += len(df_batch)

    table = pa.Table.from_pandas(df_batch, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter("data/interim/combined_filtered.parquet", table.schema)
    writer.write_table(table)

# Write T1DEXI at the end
t1dexi_table = pa.Table.from_pandas(t1dexi, preserve_index=False)
writer.write_table(t1dexi_table)
writer.close()

print(f"\nCombined dataset:")
print(f"  METABONET: {len(meta_patients)} patients, {meta_rows} rows")
print(f"  T1DEXI:    {t1dexi['id'].nunique()} patients, {len(t1dexi)} rows")
print(f"  Total:     {len(meta_patients) + t1dexi['id'].nunique()} patients, {meta_rows + len(t1dexi)} rows")
print("Saved to data/interim/combined_filtered.parquet")