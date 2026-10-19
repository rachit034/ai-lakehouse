from __future__ import annotations

import io

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from services import sql_engine, vector_store, write_log

router = APIRouter()


# ── CSV → Iceberg ─────────────────────────────────────────────────────────────

@router.post("/csv")
async def ingest_csv(
    file: UploadFile = File(..., description="CSV file to ingest"),
    schema: str = Form(default="demo", description="Target Iceberg schema/database"),
    table: str = Form(..., description="Target table name"),
):
    """
    Upload a CSV → create Iceberg table → INSERT rows via DuckDB.

    DuckDB has no bulk-load CSV-into-Iceberg-catalog endpoint, so we:
    1. Parse CSV with PyArrow
    2. CREATE TABLE IF NOT EXISTS (infer schema from Arrow)
    3. INSERT rows in batches of 500
    """
    content = await file.read()
    try:
        sql_engine.validate_identifier(schema, "schema")
        sql_engine.validate_identifier(table, "table")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Parsing + CREATE + batched INSERTs are all blocking — run them in the
    # threadpool so a large CSV doesn't freeze every other request.
    return await run_in_threadpool(_ingest_csv_blocking, content, schema, table)


def _ingest_csv_blocking(content: bytes, schema: str, table: str) -> dict:
    try:
        arrow_table = pa_csv.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")

    # Ensure schema exists
    try:
        sql_engine.ensure_schema(schema)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Schema creation failed: {e}")

    # Create table
    try:
        sql_engine.create_table_from_arrow(schema, table, arrow_table.schema)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Table creation failed: {e}")

    # Batch INSERT
    df = arrow_table.to_pandas()
    batch_size = 500
    total = 0
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        rows_sql = _df_to_values_sql(batch, arrow_table.schema)
        insert_sql = (
            f"INSERT INTO iceberg.{schema}.{table} VALUES {rows_sql}"
        )
        try:
            sql_engine.execute_ddl(insert_sql)
            total += len(batch)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"INSERT failed at row {i}: {e}",
            )

    write_log.record_write(schema)
    return {
        "status": "ok",
        "table": f"iceberg.{schema}.{table}",
        "rows_ingested": total,
        "columns": list(df.columns),
    }


def _df_to_values_sql(df: pd.DataFrame, schema: pa.Schema) -> str:
    """Convert DataFrame rows to SQL VALUES string for a DuckDB INSERT."""
    import json

    rows = []
    for _, row in df.iterrows():
        vals = []
        for col, field in zip(df.columns, schema):
            val = row[col]
            if pd.isna(val):
                vals.append("NULL")
            elif pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
                escaped = str(val).replace("'", "''")
                vals.append(f"'{escaped}'")
            elif pa.types.is_boolean(field.type):
                vals.append("TRUE" if val else "FALSE")
            else:
                vals.append(str(val))
        rows.append(f"({', '.join(vals)})")
    return ", ".join(rows)


# ── Iceberg table → LanceDB embeddings ────────────────────────────────────────

class EmbedRequest(BaseModel):
    schema: str = "demo"
    table: str
    text_field: str               # column to embed
    dataset: str | None = None    # LanceDB dataset name (defaults to table name)
    limit: int = 10_000           # max rows to embed in prototype


@router.post("/embed")
def embed_table(req: EmbedRequest):
    """
    Read rows from an Iceberg table, embed the text_field column, store in LanceDB.
    In production this runs as a Ray task submitted via Airflow.
    """
    dataset = req.dataset or req.table

    # Fetch data via DuckDB — read_all, not execute(): execute() caps at
    # settings.query_max_rows, which silently embedded only the first 1000
    # rows of any table even when limit asked for 10,000.
    try:
        sql_engine.validate_identifier(req.schema, "schema")
        sql_engine.validate_identifier(req.table, "table")
        result = sql_engine.read_all(
            f"SELECT * FROM iceberg.{req.schema}.{req.table} LIMIT {int(req.limit)}",
            schema=req.schema,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DuckDB query failed: {e}")

    if not result["rows"]:
        raise HTTPException(status_code=404, detail="Table is empty")

    records = [
        dict(zip(result["columns"], row)) for row in result["rows"]
    ]

    if req.text_field not in result["columns"]:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{req.text_field}' not found. Available: {result['columns']}",
        )

    # Embed and store
    try:
        count = vector_store.ingest(dataset, records, text_field=req.text_field)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    return {
        "status": "ok",
        "dataset": dataset,
        "records_embedded": count,
        "text_field": req.text_field,
    }
