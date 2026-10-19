"""
Gold layer: NL -> SQL against Silver (or any schema), with a choice per
query — execute and return the rows (ephemeral), or materialize as a real
Gold table registered back in the catalog with lineage.

Security note, since this is the one place in the system where an LLM's
output ends up driving a CREATE TABLE: the model never writes DDL. It only
ever produces a SELECT, which goes through the exact same read-only
assertion (sql_engine._assert_read_only) as every other ad-hoc query in
this system before anything happens to it. The CREATE TABLE ... AS SELECT
wrapper — and the target schema (always "gold", never LLM-controlled) — is
application code, not model output. This is the same defense-in-depth
principle as the rest of the gateway's guardrails: the model's untrusted
output is validated before use, not trusted because it "looks like SQL".
"""
from __future__ import annotations

import asyncio
import re

import mlflow
import sqlglot
from mlflow.entities import SpanType
from sqlglot import exp

from services import catalog, datahub_client, llm, sql_engine, write_log

_TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _ensure_column_aliases(sql: str) -> str:
    """
    CREATE TABLE ... AS SELECT requires every output column to have an
    explicit name — a plain `AVG(total_amount)` works fine as an ad-hoc
    query (DuckDB auto-names it `_col1`) but fails CTAS with "Column name
    not specified". NL-generated aggregate SQL essentially never includes
    aliases, so this isn't a one-off fix, it's a systematic requirement of
    the materialize path. Derives a readable name from the function +
    referenced column (avg_total_amount) rather than a generic col_N.
    """
    tree = sqlglot.parse_one(sql, dialect="duckdb")
    for i, e in enumerate(tree.selects):
        if isinstance(e, exp.Column) or e.alias:
            continue
        cols = list(e.find_all(exp.Column))
        func_name = e.__class__.__name__.lower() if isinstance(e, exp.Func) else "expr"
        alias = f"{func_name}_{cols[0].name}" if cols else f"{func_name}_{i}"
        e.replace(exp.alias_(e.copy(), alias))
    return tree.sql(dialect="duckdb")


def _extract_source_tables(sql: str) -> list[str]:
    tree = sqlglot.parse_one(sql, dialect="duckdb")
    seen = []
    for t in tree.find_all(exp.Table):
        qualified = f"{t.catalog}.{t.db}.{t.name}" if t.catalog and t.db else t.name
        if qualified not in seen:
            seen.append(qualified)
    return seen


def _register_gold_lineage(table_name: str, question: str, source_tables: list[str]) -> None:
    datahub_client.register_lineage(
        platform="duckdb",
        name=f"iceberg.gold.{table_name}",
        description=f'Gold table, materialized from the NL question: "{question}"',
        upstream_tables=source_tables,
        custom_properties={"layer": "gold", "built_from": "natural language"},
    )


@mlflow.trace(name="gold.run_query", span_type=SpanType.CHAIN)
async def run_gold_query(
    question: str,
    source_schema: str = "silver",
    materialize: bool = False,
    table_name: str | None = None,
) -> dict:
    # Blocking catalog/DuckDB/DataHub calls run via asyncio.to_thread so this
    # coroutine never stalls the gateway's event loop.
    schema_ctx = await asyncio.to_thread(catalog.build_schema_context, schema=source_schema)
    if not schema_ctx.strip():
        raise ValueError(f"No tables found in schema '{source_schema}'.")

    # NL -> SQL: same path as /query/nl (semantic cache, Ollama, sqlglot
    # syntax validation) — this produces at most a validated-parseable SELECT.
    nl_result = await llm.nl_to_sql(question, schema_ctx)
    select_sql = nl_result["sql"]

    # Defense-in-depth: re-validate independently that it's actually
    # read-only before anything downstream trusts it, materialize or not.
    sql_engine._assert_read_only(select_sql)

    if not materialize:
        result = await asyncio.to_thread(sql_engine.execute, select_sql, schema=source_schema)
        return {**result, "generated_sql": select_sql, "materialized": False}

    if not table_name or not _TABLE_NAME_RE.match(table_name):
        raise ValueError(
            "materialize=true requires a table_name matching ^[a-zA-Z_][a-zA-Z0-9_]*$ "
            "(never taken from the NL question or model output)."
        )

    source_tables = _extract_source_tables(select_sql)
    ctas_sql = _ensure_column_aliases(select_sql)
    # Re-validate post-rewrite: the alias pass only adds `AS name` to
    # existing SELECT expressions, but re-checking costs nothing and keeps
    # the guarantee airtight rather than assumed.
    sql_engine._assert_read_only(ctas_sql)

    def _materialize() -> int:
        sql_engine.ensure_schema("gold")
        sql_engine.execute_ddl(
            f"CREATE TABLE IF NOT EXISTS iceberg.gold.{table_name} AS {ctas_sql}"
        )
        count_result = sql_engine.execute(f"SELECT count(*) AS n FROM iceberg.gold.{table_name}")
        _register_gold_lineage(table_name, question, source_tables)
        write_log.record_write("gold")
        return count_result["rows"][0][0]

    row_count = await asyncio.to_thread(_materialize)

    return {
        "materialized": True,
        "gold_table": f"iceberg.gold.{table_name}",
        "row_count": row_count,
        "generated_sql": ctas_sql,
        "source_tables": source_tables,
    }
