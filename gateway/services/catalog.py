"""
Schema discovery via DuckDB's catalog introspection functions
(duckdb_schemas/duckdb_tables/duckdb_columns), against the Gravitino-backed
Iceberg REST catalog attached in sql_engine.py — Iceberg namespaces are
exposed there as ordinary DuckDB schemas.
In production this calls boto3 Glue API instead.
"""
import mlflow
from mlflow.entities import SpanType

from core.config import settings
from services import datahub_client, metrics, sql_engine


def list_schemas() -> list[str]:
    _, rows = sql_engine.catalog_query(
        "SELECT schema_name FROM duckdb_schemas() WHERE database_name = ? AND NOT internal",
        [settings.iceberg_catalog_name],
    )
    return [r[0] for r in rows]


def list_tables(schema: str) -> list[str]:
    _, rows = sql_engine.catalog_query(
        "SELECT table_name FROM duckdb_tables() WHERE database_name = ? AND schema_name = ?",
        [settings.iceberg_catalog_name, schema],
    )
    return [r[0] for r in rows]


def get_columns(schema: str, table: str) -> list[dict]:
    """Return list of {name, type} dicts for a table."""
    _, rows = sql_engine.catalog_query(
        "SELECT column_name, data_type FROM duckdb_columns() "
        "WHERE database_name = ? AND schema_name = ? AND table_name = ? "
        "ORDER BY column_index",
        [settings.iceberg_catalog_name, schema, table],
    )
    return [{"name": r[0], "type": r[1]} for r in rows]


@mlflow.trace(name="catalog.build_schema_context", span_type=SpanType.RETRIEVER)
def build_schema_context(schema: str | None = None) -> str:
    """
    Build a schema string for the NL→SQL prompt: tables + columns from the
    Iceberg catalog, enriched with human-written descriptions, business-
    glossary aliases, and deprecation warnings from DataHub where available
    (falls back to bare name/type silently if DataHub has no entry or is
    unreachable — see datahub_client.get_table_context). Glossary aliases
    matter here specifically: a question asking for "revenue" won't match any
    column literally named that, but ask_nl needs to resolve it to total_amount.
    """
    schemas = [schema] if schema else list_schemas()
    # skip internal schemas
    schemas = [s for s in schemas if s not in ("information_schema", "sys", "main", "pg_catalog")]

    lines: list[str] = []
    for s in schemas:
        for table in list_tables(s):
            cols = get_columns(s, table)
            enrichment = datahub_client.get_table_context(settings.iceberg_catalog_name, s, table)
            col_info = enrichment.get("columns", {})

            header = f"{settings.iceberg_catalog_name}.{s}.{table}"
            if enrichment.get("description"):
                header += f" — {enrichment['description']}"
            if enrichment.get("deprecated"):
                header += f" [DEPRECATED{': ' + enrichment['deprecation_note'] if enrichment.get('deprecation_note') else ''} — avoid unless explicitly asked for]"
            lines.append(header)

            for c in cols:
                info = col_info.get(c["name"], {})
                col_line = f"  {c['name']} ({c['type']})"
                if info.get("glossary_terms"):
                    col_line += f" [aka: {', '.join(info['glossary_terms'])}]"
                if info.get("description"):
                    col_line += f": {info['description']}"
                lines.append(col_line)

    metrics_block = metrics.build_metrics_context("sql", schema=schema or "ecommerce")
    if metrics_block:
        lines.append("")
        lines.append(metrics_block)

    return "\n".join(lines)
