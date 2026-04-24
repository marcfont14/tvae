import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

writer = None
target_schema = None
total_patients = 0
total_rows = 0

def _cast_nulls(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Align columns to schema (drop extras, add missing as null) then cast."""
    schema_names = schema.names
    # Drop columns not in schema
    for col in table.schema.names:
        if col not in schema_names:
            table = table.remove_column(table.schema.get_field_index(col))
    # Add missing columns as null
    for field in schema:
        if field.name not in table.schema.names:
            table = table.append_column(field, pa.array([None] * len(table), type=field.type))
    # Reorder to match schema
    table = table.select(schema_names)
    return table.cast(schema)

def _infer_schema(table: pa.Table) -> pa.Schema:
    """Replace null-typed columns with string or double based on column name."""
    string_cols = {'id', 'gender', 'insulin_delivery_modality', 'source_file'}
    fields = []
    for field in table.schema:
        if pa.types.is_null(field.type):
            t = pa.string() if field.name in string_cols else pa.float64()
            fields.append(field.with_type(t))
        else:
            fields.append(field)
    return pa.schema(fields)

def write_source(path, source_name):
    global writer, target_schema, total_patients, total_rows
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
            if target_schema is None:
                target_schema = _infer_schema(table)
                writer = pq.ParquetWriter(
                    "data/interim/combined_filtered.parquet", target_schema)
            writer.write_table(_cast_nulls(table, target_schema))
        print(f"  {source_name}: {len(patients)} patients, {rows:,} rows")
        return len(patients), rows

# 1. METABONET train
meta_tr_p, meta_tr_r = write_source(
    "data/raw/metabonet_train_filtered.parquet", "METABONET_train")

# 2. METABONET test
meta_te_p, meta_te_r = write_source(
    "data/raw/metabonet_test_filtered.parquet", "METABONET_test")

# 3. T1DEXI adultos
t1a_p, t1a_r = write_source(
    "data/raw/t1dexi_parsed.parquet", "T1DEXI_adults")

writer.close()

print(f"\nCombined dataset:")
print(f"  METABONET train:   {meta_tr_p} patients, {meta_tr_r:,} rows")
print(f"  METABONET test:    {meta_te_p} patients, {meta_te_r:,} rows")
print(f"  T1DEXI adults:     {t1a_p} patients, {t1a_r:,} rows")
total_p = meta_tr_p + meta_te_p + t1a_p
total_r = meta_tr_r + meta_te_r + t1a_r
print(f"  TOTAL:             {total_p} patients, {total_r:,} rows")
print("Saved to data/interim/combined_filtered.parquet")