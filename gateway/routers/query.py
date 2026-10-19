from __future__ import annotations

import traceback
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core import redaction
from services import catalog, llm, metrics, sql_engine, vector_store

router = APIRouter()


# ── Models ────────────────────────────────────────────────────────────────────

class SQLRequest(BaseModel):
    sql: str
    schema: str | None = None


class NLRequest(BaseModel):
    question: str
    schema: str | None = None   # restrict schema context to one schema
    explain: bool = False        # return generated SQL in response


class VectorRequest(BaseModel):
    dataset: str
    query: str
    top_k: int = 10
    filter: str | None = None   # SQL filter expression e.g. "price < 100"


class QueryResult(BaseModel):
    columns: list[str] = []
    rows: list[list[Any]] = []
    row_count: int = 0
    generated_sql: str | None = None
    cached: bool | None = None


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/sql", response_model=QueryResult)
def run_sql(req: SQLRequest, request: Request):
    """Execute a SQL query directly on DuckDB (Iceberg catalog).

    data_engineer/data_modeler get raw values back; business always gets PII
    redacted — see core/redaction.py.
    """
    try:
        result = sql_engine.execute(req.sql, schema=req.schema)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    if request.state.role == "business":
        # req.schema is rarely set in practice — most calls write fully-
        # qualified table names instead, so pull the real schema(s) out of
        # the SQL text itself too (a join can span more than one).
        schemas = sql_engine.extract_schemas(req.sql) | ({req.schema} if req.schema else set())
        result["rows"] = redaction.redact_rows(result["columns"], result["rows"], schema=schemas)
    return QueryResult(**result)


@router.post("/nl", response_model=QueryResult)
async def run_nl(req: NLRequest, request: Request):
    """
    Translate a natural language question to SQL and execute it.

    Flow: fetch schema → Ollama NL→SQL (semantically cached) → DuckDB → return results
    """
    # 1. Schema context
    try:
        schema_ctx = catalog.build_schema_context(schema=req.schema)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Schema fetch failed: {e}")

    if not schema_ctx.strip():
        raise HTTPException(
            status_code=404,
            detail="No tables found. Ingest some data first via POST /ingest/csv",
        )

    # 2. NL → SQL
    try:
        nl_result = await llm.nl_to_sql(req.question, schema_ctx)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"LLM error [{type(e).__name__}]: {e!r}")

    sql = nl_result["sql"]

    # 3. Execute
    try:
        result = sql_engine.execute(sql, schema=req.schema)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Generated SQL failed: {e}\n\nSQL was:\n{sql}",
        )

    if request.state.role == "business":
        schemas = sql_engine.extract_schemas(sql) | ({req.schema} if req.schema else set())
        result["rows"] = redaction.redact_rows(result["columns"], result["rows"], schema=schemas)
    return QueryResult(
        **result,
        generated_sql=sql if req.explain else None,
        cached=nl_result["cached"],
    )


@router.post("/vector")
def run_vector(req: VectorRequest, request: Request):
    """Semantic vector search over a LanceDB dataset."""
    try:
        results = vector_store.search(
            dataset=req.dataset,
            query=req.query,
            top_k=req.top_k,
            filter_sql=req.filter,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if request.state.role == "business":
        results = redaction.redact_records(results)
    return {"results": results, "count": len(results)}


@router.get("/schema")
def get_schema(schema: str | None = None):
    """List all tables and columns visible to DuckDB."""
    try:
        schemas = [schema] if schema else catalog.list_schemas()
        schemas = [s for s in schemas if s not in ("information_schema", "sys")]
        out = {}
        for s in schemas:
            tables = catalog.list_tables(s)
            out[s] = {t: catalog.get_columns(s, t) for t in tables}
        return out
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/datasets")
def get_datasets():
    """List LanceDB vector datasets."""
    return {"datasets": vector_store.list_datasets()}


@router.get("/metrics")
def get_metrics():
    """
    Named business metric definitions (name, version, description, aliases,
    SQL and/or Cypher expression). The same registry is injected into both
    catalog.build_schema_context() and graph_store.build_schema_context(), so
    this endpoint is a way to inspect what's currently authoritative for
    both the NL->SQL and NL->Cypher paths — not a separate source of truth.
    """
    return {"metrics": metrics.list_metrics()}
