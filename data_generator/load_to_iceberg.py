"""
Load generated Parquet files into Iceberg tables via DuckDB.
This is the "local Glue + Athena" equivalent — same SQL runs in prod against real Athena.

Steps:
  1. Upload Parquet to MinIO  (s3://lakehouse/raw/<table>/)
  2. CREATE SCHEMA IF NOT EXISTS (via DuckDB, against the Gravitino-attached Iceberg catalog)
  3. CREATE TABLE IF NOT EXISTS (Iceberg, Parquet format)
  4. INSERT INTO ... SELECT * FROM read_parquet(...)   ← DuckDB reads, DuckDB ingests

Run:
    python load_to_iceberg.py
    python load_to_iceberg.py --schema ecommerce --gravitino-uri http://localhost:19090
"""

from __future__ import annotations

import argparse
from pathlib import Path

import boto3
import duckdb
import pandas as pd
from botocore.client import Config

PARQUET_DIR = Path(__file__).parent / "parquet"

TABLES = ["customers", "products", "orders", "order_items"]

# DuckDB/Iceberg column type mapping for each table
SCHEMAS: dict[str, list[tuple[str, str]]] = {
    "customers": [
        ("customer_id",     "INTEGER"),
        ("first_name",      "VARCHAR"),
        ("last_name",       "VARCHAR"),
        ("email",           "VARCHAR"),
        ("country",         "VARCHAR"),
        ("city",            "VARCHAR"),
        ("segment",         "VARCHAR"),
        ("signup_date",     "VARCHAR"),
        ("age_group",       "VARCHAR"),
        ("lifetime_value",  "DOUBLE"),
    ],
    "products": [
        ("product_id",      "INTEGER"),
        ("name",            "VARCHAR"),
        ("category",        "VARCHAR"),
        ("subcategory",     "VARCHAR"),
        ("price",           "DOUBLE"),
        ("cost",            "DOUBLE"),
        ("margin_pct",      "DOUBLE"),
        ("supplier",        "VARCHAR"),
        ("stock_quantity",  "INTEGER"),
        ("is_active",       "BOOLEAN"),
        ("description",     "VARCHAR"),
        ("launch_date",     "VARCHAR"),
    ],
    "orders": [
        ("order_id",        "INTEGER"),
        ("customer_id",     "INTEGER"),
        ("order_date",      "VARCHAR"),
        ("order_year",      "INTEGER"),
        ("order_month",     "INTEGER"),
        ("order_quarter",   "INTEGER"),
        ("status",          "VARCHAR"),
        ("payment_method",  "VARCHAR"),
        ("total_amount",    "DOUBLE"),
        ("item_count",      "INTEGER"),
        ("shipping_days",   "INTEGER"),
    ],
    "order_items": [
        ("item_id",         "INTEGER"),
        ("order_id",        "INTEGER"),
        ("product_id",      "INTEGER"),
        ("quantity",        "INTEGER"),
        ("unit_price",      "DOUBLE"),
        ("discount_pct",    "DOUBLE"),
        ("line_total",      "DOUBLE"),
    ],
}


# ── S3 / MinIO upload ─────────────────────────────────────────────────────────

def upload_parquet(
    table: str,
    minio_endpoint: str,
    access_key: str,
    secret_key: str,
    bucket: str = "lakehouse",
) -> str:
    path = PARQUET_DIR / f"{table}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run generate.py first.")

    s3 = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    key = f"raw/{table}/{table}.parquet"
    s3.upload_file(str(path), bucket, key)
    s3_uri = f"s3://{bucket}/raw/{table}/"
    print(f"  ✓ Uploaded {table}.parquet  →  {s3_uri}")
    return s3_uri


# ── DuckDB / Gravitino connection ─────────────────────────────────────────────

def _duckdb_conn(
    gravitino_uri: str,
    metalake: str,
    catalog: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
) -> duckdb.DuckDBPyConnection:
    """
    Standalone DuckDB connection for this CLI script (separate process from the
    gateway, so it attaches its own — see gateway/services/sql_engine.py for
    the equivalent embedded-in-the-gateway setup and why each piece is needed).
    """
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL iceberg; LOAD iceberg;")
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute(f"""
        CREATE SECRET minio_s3 (
            TYPE s3,
            ENDPOINT '{minio_endpoint.removeprefix("http://").removeprefix("https://")}',
            KEY_ID '{minio_access_key}',
            SECRET '{minio_secret_key}',
            REGION 'us-east-1',
            USE_SSL false,
            URL_STYLE 'path'
        )
    """)
    conn.execute("CREATE SECRET iceberg_rest (TYPE ICEBERG, TOKEN 'dummy')")
    conn.execute(f"""
        ATTACH '{metalake}' AS {catalog} (
            TYPE ICEBERG,
            ENDPOINT '{gravitino_uri}/iceberg/',
            SECRET iceberg_rest
        )
    """)
    return conn


def create_schema(conn, catalog: str, schema: str):
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    print(f"  ✓ Schema {catalog}.{schema}")


def create_table(conn, catalog: str, schema: str, table: str, columns: list[tuple[str, str]]):
    col_defs = ",\n    ".join(f'"{col}" {dtype}' for col, dtype in columns)
    ddl = f"CREATE TABLE IF NOT EXISTS {catalog}.{schema}.{table} (\n    {col_defs}\n)"
    conn.execute(ddl)
    print(f"  ✓ Table {catalog}.{schema}.{table}")


def insert_from_parquet(
    conn,
    catalog: str,
    schema: str,
    table: str,
    parquet_path: Path,
    batch_size: int = 500,
):
    """Read Parquet with DuckDB locally, INSERT rows into the Iceberg table in batches."""
    df = pd.read_parquet(parquet_path)
    # Replace NaN with None for SQL NULL
    df = df.where(pd.notna(df), other=None)

    columns = [col for col, _ in SCHEMAS[table]]
    col_list = ", ".join(f'"{c}"' for c in columns)

    total = 0
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size]
        rows = []
        for _, row in batch.iterrows():
            vals = []
            for col, dtype in SCHEMAS[table]:
                v = row[col]
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    vals.append("NULL")
                elif dtype in ("VARCHAR",):
                    escaped = str(v).replace("'", "''")
                    vals.append(f"'{escaped}'")
                elif dtype == "BOOLEAN":
                    vals.append("TRUE" if v else "FALSE")
                else:
                    vals.append(str(v))
            rows.append(f"({', '.join(vals)})")

        insert_sql = (
            f"INSERT INTO {catalog}.{schema}.{table} ({col_list}) "
            f"VALUES {', '.join(rows)}"
        )
        conn.execute(insert_sql)
        total += len(batch)

    print(f"  ✓ Inserted {total:,} rows into {catalog}.{schema}.{table}")
    return total


# ── Quick verify ──────────────────────────────────────────────────────────────

def verify(conn, catalog: str, schema: str):
    """Run a quick join query to verify the data loaded correctly."""
    rows = conn.execute(f"""
        SELECT
            p.category,
            COUNT(DISTINCT o.order_id) AS orders,
            ROUND(SUM(oi.line_total), 0) AS revenue
        FROM {catalog}.{schema}.order_items oi
        JOIN {catalog}.{schema}.orders   o ON oi.order_id   = o.order_id
        JOIN {catalog}.{schema}.products p ON oi.product_id = p.product_id
        WHERE o.status = 'completed'
        GROUP BY p.category
        ORDER BY revenue DESC
    """).fetchall()
    print("\n  Verification query — Revenue by category (DuckDB):")
    print(f"  {'Category':<20} {'Orders':>8} {'Revenue':>12}")
    print(f"  {'-'*20} {'-'*8} {'-'*12}")
    for row in rows:
        print(f"  {str(row[0]):<20} {row[1]:>8,} {row[2]:>12,.0f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Load Parquet files into Iceberg via DuckDB")
    parser.add_argument("--schema",         default="ecommerce")
    parser.add_argument("--gravitino-uri",  default="http://localhost:19090")
    parser.add_argument("--metalake",       default="lakehouse")
    parser.add_argument("--catalog",        default="iceberg")
    parser.add_argument("--minio-endpoint", default="http://localhost:9100")
    parser.add_argument("--minio-access-key", default="minioadmin")
    parser.add_argument("--minio-secret-key", default="minioadmin")
    parser.add_argument("--skip-upload",    action="store_true",
                        help="Skip MinIO upload (files already there)")
    parser.add_argument("--verify",         action="store_true",
                        help="Run a verification query after loading")
    args = parser.parse_args()

    print("\n── Uploading Parquet to MinIO ───────────────────────────────────")
    if not args.skip_upload:
        for table in TABLES:
            upload_parquet(
                table,
                minio_endpoint=args.minio_endpoint,
                access_key=args.minio_access_key,
                secret_key=args.minio_secret_key,
            )
    else:
        print("  Skipped (--skip-upload)")

    print("\n── Creating Iceberg schema + tables via DuckDB ──────────────────")
    conn = _duckdb_conn(
        args.gravitino_uri, args.metalake, args.catalog,
        args.minio_endpoint, args.minio_access_key, args.minio_secret_key,
    )
    create_schema(conn, args.catalog, args.schema)

    for table in TABLES:
        create_table(conn, args.catalog, args.schema, table, SCHEMAS[table])

    print("\n── Inserting data ───────────────────────────────────────────────")
    for table in TABLES:
        path = PARQUET_DIR / f"{table}.parquet"
        insert_from_parquet(conn, args.catalog, args.schema, table, path)

    if args.verify:
        print("\n── Verification ─────────────────────────────────────────────────")
        verify(conn, args.catalog, args.schema)

    print("\n── Done ─────────────────────────────────────────────────────────")
    print("  Test NL query:")
    print('  curl -X POST http://localhost:8088/query/nl \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"question": "What is total revenue by category?", "schema": "ecommerce", "explain": true}\'\n')


if __name__ == "__main__":
    main()
