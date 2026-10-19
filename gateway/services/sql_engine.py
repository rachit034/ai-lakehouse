"""
SQL execution via DuckDB, embedded in this process (no separate query-engine
service — see the docker-compose header for the full topology). DuckDB's
`iceberg` extension attaches Gravitino's Iceberg REST catalog for metadata and
reads/writes Parquet directly on MinIO via the `httpfs` extension.
In production, swap _get_conn()'s ATTACH target for the real Glue/Athena-
compatible Iceberg REST endpoint and the S3 secret for real AWS creds.
"""
from __future__ import annotations

import threading

import duckdb
import mlflow
import sqlglot
import pyarrow as pa
from mlflow.entities import SpanType
from sqlglot import exp

from core.config import settings

# Statement types that mutate data/schema. Anything sqlglot resolves to one of
# these — anywhere in the parsed tree, not just at the top level — is rejected
# by execute(). This is the boundary between the ad-hoc query surface
# (/query/sql, ask_nl's generated SQL) and writes, which only ever happen
# through execute_ddl() below, called by ingest.py with statements it built
# itself, never with arbitrary caller-supplied SQL.
_WRITE_TYPES = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.AlterTable, exp.Merge)
# sqlglot falls back to a generic Command node for syntax its DuckDB dialect
# doesn't specifically model (SHOW, EXPLAIN, DESCRIBE) — allow only the known
# safe read-only ones by keyword; reject any other unrecognized statement.
_SAFE_COMMAND_PREFIXES = ("SHOW", "EXPLAIN", "DESCRIBE", "DESC")

_conn: duckdb.DuckDBPyConnection | None = None
_lock = threading.Lock()


def _strip_scheme(url: str) -> str:
    return url.removeprefix("http://").removeprefix("https://")


def _init_conn() -> duckdb.DuckDBPyConnection:
    """
    One DuckDB connection for the process lifetime, guarded by _lock for
    every use — a prototype-appropriate way to keep a single embedded
    connection safe under concurrent FastAPI requests without standing up a
    real connection pool. Attaches Gravitino's Iceberg REST catalog (named
    to match settings.iceberg_catalog_name, "iceberg" by default, so every
    `iceberg.{schema}.{table}` reference elsewhere in the codebase keeps
    working unchanged) and configures httpfs for direct MinIO S3 access.
    """
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL iceberg; LOAD iceberg;")
    conn.execute("INSTALL httpfs; LOAD httpfs;")

    conn.execute(f"""
        CREATE SECRET minio_s3 (
            TYPE s3,
            ENDPOINT '{_strip_scheme(settings.minio_endpoint)}',
            KEY_ID '{settings.minio_access_key}',
            SECRET '{settings.minio_secret_key}',
            REGION 'us-east-1',
            USE_SSL false,
            URL_STYLE 'path'
        )
    """)

    # Gravitino's local dev deployment doesn't enforce real OAuth2 on its
    # Iceberg REST endpoint — a placeholder token satisfies the iceberg
    # extension's requirement that *some* secret be attached. Swap for a
    # real CLIENT_ID/CLIENT_SECRET/OAUTH2_SERVER_URI secret in production.
    conn.execute("""
        CREATE SECRET iceberg_rest (
            TYPE ICEBERG,
            TOKEN 'dummy'
        )
    """)

    # NOTE: Gravitino's Iceberg REST service path — verify this against the
    # deployed Gravitino version; some releases scope it per-catalog
    # (e.g. f"{settings.gravitino_uri}/iceberg/{settings.iceberg_catalog_name}/").
    conn.execute(f"""
        ATTACH '{settings.gravitino_metalake}' AS {settings.iceberg_catalog_name} (
            TYPE ICEBERG,
            ENDPOINT '{settings.gravitino_uri}/iceberg/',
            SECRET iceberg_rest
        )
    """)
    return conn


def _get_conn() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        _conn = _init_conn()
    return _conn


def _assert_read_only(sql: str) -> None:
    try:
        statements = [s for s in sqlglot.parse(sql, dialect="duckdb") if s is not None]
    except Exception as e:
        raise ValueError(f"Could not parse SQL to verify it's read-only: {e}") from e

    if len(statements) != 1:
        raise ValueError("Only a single read-only statement is allowed per call (no statement stacking).")

    stmt = statements[0]
    if isinstance(stmt, _WRITE_TYPES) or stmt.find(*_WRITE_TYPES):
        raise ValueError(
            f"Only read-only queries are allowed here (got {type(stmt).__name__}). "
            "Writes must go through /ingest/csv or /ingest/embed."
        )
    if isinstance(stmt, exp.Command):
        first_word = sql.strip().split(None, 1)[0].upper() if sql.strip() else ""
        if first_word not in _SAFE_COMMAND_PREFIXES:
            raise ValueError(f"Unrecognized statement is not allowed via this read-only endpoint: {sql[:80]!r}")


def extract_schemas(sql: str) -> set[str]:
    """
    Iceberg schemas referenced by fully-qualified table names in `sql` (the
    `.db` part of `catalog.db.table`). Callers (redaction's catalog lookup)
    can't rely on an explicit schema param here — in practice, most /query/sql
    calls write fully-qualified table names and never pass one. Best-effort:
    returns an empty set if the SQL doesn't parse.
    """
    try:
        stmt = sqlglot.parse_one(sql, dialect="duckdb")
    except Exception:
        return set()
    return {t.db for t in stmt.find_all(exp.Table) if t.db}


@mlflow.trace(name="duckdb.execute", span_type=SpanType.TOOL)
def execute(sql: str, schema: str | None = None) -> dict:
    """
    Run a read-only SQL statement on DuckDB, return {columns, rows, row_count}.
    Max 1000 rows for prototype (add pagination later).
    """
    _assert_read_only(sql)
    with _lock:
        cur = _get_conn().cursor()
        cur.execute(f"USE {settings.iceberg_catalog_name}.{schema or settings.default_schema}")
        cur.execute(sql)
        rows = cur.fetchmany(1000)
        columns = [desc[0] for desc in cur.description] if cur.description else []

    return {
        "columns": columns,
        "rows": [list(r) for r in rows],
        "row_count": len(rows),
    }


@mlflow.trace(name="duckdb.execute_ddl", span_type=SpanType.TOOL)
def execute_ddl(sql: str, schema: str | None = None) -> bool:
    """Run a DDL/DML statement (CREATE TABLE, CREATE SCHEMA, INSERT, etc.)."""
    with _lock:
        cur = _get_conn().cursor()
        if schema:
            cur.execute(f"USE {settings.iceberg_catalog_name}.{schema}")
        cur.execute(sql)
    return True


def catalog_query(sql: str, params: list | None = None) -> tuple[list[str], list[tuple]]:
    """
    Internal helper for catalog.py's introspection queries (duckdb_schemas() /
    duckdb_tables() / duckdb_columns()) — skips _assert_read_only since the SQL
    text is always built by catalog.py itself, never caller-supplied, but still
    goes through the same lock as every other use of the shared connection.
    """
    with _lock:
        cur = _get_conn().cursor()
        cur.execute(sql, params or [])
        rows = cur.fetchall()
        columns = [d[0] for d in cur.description] if cur.description else []
    return columns, rows


def ensure_schema(schema: str):
    """Create schema if it doesn't exist."""
    execute_ddl(f"CREATE SCHEMA IF NOT EXISTS {settings.iceberg_catalog_name}.{schema}")


def create_table_from_arrow(schema: str, table_name: str, arrow_schema: pa.Schema):
    """
    CREATE TABLE in Iceberg from a PyArrow schema.
    Type mapping: Arrow → DuckDB SQL types (near-identical to the Trino/Iceberg
    names this replaced — both are ANSI-flavored).
    """
    type_map = {
        pa.int8(): "TINYINT",
        pa.int16(): "SMALLINT",
        pa.int32(): "INTEGER",
        pa.int64(): "BIGINT",
        pa.float32(): "REAL",
        pa.float64(): "DOUBLE",
        pa.bool_(): "BOOLEAN",
        pa.string(): "VARCHAR",
        pa.large_string(): "VARCHAR",
        pa.date32(): "DATE",
        pa.timestamp("us"): "TIMESTAMP",
        pa.timestamp("ms"): "TIMESTAMP",
    }

    cols = []
    for field in arrow_schema:
        duckdb_type = type_map.get(field.type, "VARCHAR")
        cols.append(f'"{field.name}" {duckdb_type}')

    ddl = (
        f"CREATE TABLE IF NOT EXISTS {settings.iceberg_catalog_name}.{schema}.{table_name} "
        f"({', '.join(cols)})"
    )
    execute_ddl(ddl)
