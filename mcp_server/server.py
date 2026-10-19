"""
MCP server for the AI Lakehouse prototype.

Wraps the FastAPI gateway (http://localhost:8088 by default) as MCP tools so
any MCP client (Claude Agent SDK, Claude Desktop, etc.) can query, search and
ingest data into the lakehouse. This server does no lakehouse work itself —
every tool is a thin HTTP call to the gateway, which owns DuckDB/LanceDB/Redis.

Run standalone (stdio transport):
    GATEWAY_URL=http://localhost:8088 python server.py
"""
from __future__ import annotations

import os
import uuid
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8088").rstrip("/")
_TIMEOUT = httpx.Timeout(120.0)

# One MCP server subprocess = one agent invocation (agents spawn it fresh per
# run over stdio) — so a process-lifetime run id lets every gateway call this
# process makes be found together as one trace group in MLflow.
_RUN_ID = str(uuid.uuid4())
_AGENT_NAME = os.environ.get("AGENT_NAME", "unknown-agent")
_API_KEY = os.environ.get("GATEWAY_API_KEY", "")

mcp = FastMCP("ai-lakehouse")

if not _API_KEY:
    import sys

    print(
        "WARNING: GATEWAY_API_KEY not set — every gateway call will 401. "
        "Pass it via the agent's mcp_servers env config.",
        file=sys.stderr,
    )


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=GATEWAY_URL,
        timeout=_TIMEOUT,
        headers={"X-Agent-Name": _AGENT_NAME, "X-Run-Id": _RUN_ID, "X-API-Key": _API_KEY},
    )


def _raise_for_gateway_error(resp: httpx.Response) -> None:
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"Gateway error {resp.status_code}: {detail}")


# ── Query tools ─────────────────────────────────────────────────────────────

@mcp.tool()
def run_sql(sql: str, schema: str | None = None) -> dict[str, Any]:
    """Execute a SQL query directly against DuckDB/Iceberg.

    Args:
        sql: A SQL SELECT statement (DuckDB/ANSI syntax).
        schema: Optional schema to run in (defaults to the gateway's configured schema).
    """
    with _client() as c:
        resp = c.post("/query/sql", json={"sql": sql, "schema": schema})
    _raise_for_gateway_error(resp)
    return resp.json()


@mcp.tool()
def ask_nl(question: str, schema: str | None = None, explain: bool = True) -> dict[str, Any]:
    """Ask a natural-language question over the lakehouse. Translates to SQL via the
    gateway's LLM and executes it against DuckDB.

    Args:
        question: The question in plain English.
        schema: Optional schema to restrict table context to.
        explain: If True, include the generated SQL in the response.
    """
    with _client() as c:
        resp = c.post(
            "/query/nl",
            json={"question": question, "schema": schema, "explain": explain},
        )
    _raise_for_gateway_error(resp)
    return resp.json()


@mcp.tool()
def vector_search(
    dataset: str,
    query: str,
    top_k: int = 10,
    filter: str | None = None,
) -> dict[str, Any]:
    """Run a semantic vector search over a LanceDB dataset.

    Args:
        dataset: Name of the LanceDB dataset (see list_vector_datasets).
        query: Natural-language search text.
        top_k: Max number of results to return.
        filter: Optional SQL filter expression, e.g. "price < 100".
    """
    with _client() as c:
        resp = c.post(
            "/query/vector",
            json={"dataset": dataset, "query": query, "top_k": top_k, "filter": filter},
        )
    _raise_for_gateway_error(resp)
    return resp.json()


@mcp.tool()
def get_schema(schema: str | None = None) -> dict[str, Any]:
    """List tables and columns visible to DuckDB. Omit `schema` to list all schemas.

    Args:
        schema: Optional single schema name to restrict to.
    """
    with _client() as c:
        resp = c.get("/query/schema", params={"schema": schema} if schema else {})
    _raise_for_gateway_error(resp)
    return resp.json()


@mcp.tool()
def list_vector_datasets() -> dict[str, Any]:
    """List available LanceDB vector-search datasets."""
    with _client() as c:
        resp = c.get("/query/datasets")
    _raise_for_gateway_error(resp)
    return resp.json()


@mcp.tool()
def list_metrics() -> dict[str, Any]:
    """List named business metric definitions (name, version, description, aliases,
    SQL and/or Cypher expression) — the same definitions get_schema and
    get_graph_schema already inject into their context. Call this if you want to
    check a metric's exact formula directly rather than reading it off a schema dump.
    """
    with _client() as c:
        resp = c.get("/query/metrics")
    _raise_for_gateway_error(resp)
    return resp.json()


# ── Ingest tools ─────────────────────────────────────────────────────────────

@mcp.tool()
def ingest_csv(file_path: str, table: str, schema: str = "demo") -> dict[str, Any]:
    """Load a local CSV file into a new (or existing) Iceberg table.

    Args:
        file_path: Absolute path to a CSV file on disk.
        table: Target table name.
        schema: Target Iceberg schema/database (default "demo").
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")

    with open(file_path, "rb") as f:
        with _client() as c:
            resp = c.post(
                "/ingest/csv",
                data={"schema": schema, "table": table},
                files={"file": (os.path.basename(file_path), f, "text/csv")},
            )
    _raise_for_gateway_error(resp)
    return resp.json()


@mcp.tool()
def embed_table(
    table: str,
    text_field: str,
    schema: str = "demo",
    dataset: str | None = None,
    limit: int = 10_000,
) -> dict[str, Any]:
    """Embed a text column from an Iceberg table and store the vectors in LanceDB.

    Args:
        table: Source Iceberg table name.
        text_field: Column to embed.
        schema: Source Iceberg schema/database (default "demo").
        dataset: Target LanceDB dataset name (defaults to the table name).
        limit: Max rows to embed.
    """
    with _client() as c:
        resp = c.post(
            "/ingest/embed",
            json={
                "schema": schema,
                "table": table,
                "text_field": text_field,
                "dataset": dataset,
                "limit": limit,
            },
        )
    _raise_for_gateway_error(resp)
    return resp.json()


# ── Graph tools ───────────────────────────────────────────────────────────

@mcp.tool()
def build_knowledge_graph(schema: str = "ecommerce", limit: int = 100_000) -> dict[str, Any]:
    """Build/refresh the knowledge graph in Neo4j from an Iceberg schema's tables
    (currently wired for "ecommerce": customers, orders, order_items, products).
    Idempotent — safe to re-run after new data lands. Returns node/relationship counts.

    Args:
        schema: Source Iceberg schema to build the graph from.
        limit: Per-table row cap (raise only if a table exceeds 100k rows).
    """
    with _client() as c:
        resp = c.post("/graph/build", json={"schema": schema, "limit": limit})
    _raise_for_gateway_error(resp)
    return resp.json()


@mcp.tool()
def graph_query(cypher: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a Cypher query against the knowledge graph.

    Graph shape (schema="ecommerce"):
        (:Customer {customer_id, first_name, last_name, email, country, city,
                     segment, age_group, lifetime_value})
        (:Order {order_id, order_date, status, payment_method, total_amount, item_count})
        (:Product {product_id, name, category, subcategory, price, cost,
                    margin_pct, stock_quantity, is_active})
        (:Category {name})
        (:Supplier {name})
        (:Customer)-[:PLACED]->(:Order)
        (:Order)-[:CONTAINS {quantity, unit_price}]->(:Product)
        (:Product)-[:IN_CATEGORY]->(:Category)
        (:Product)-[:SUPPLIED_BY]->(:Supplier)

    Args:
        cypher: The Cypher query to run.
        params: Optional query parameters (referenced as $name in the query).
    """
    with _client() as c:
        resp = c.post("/graph/query", json={"cypher": cypher, "params": params})
    _raise_for_gateway_error(resp)
    return resp.json()


@mcp.tool()
def graph_stats() -> dict[str, Any]:
    """Return node and relationship counts currently in the knowledge graph."""
    with _client() as c:
        resp = c.get("/graph/stats")
    _raise_for_gateway_error(resp)
    return resp.json()


@mcp.tool()
def infer_graph_model(schema: str) -> dict[str, Any]:
    """Read-only: infer a graph model (nodes, relationships) from a schema's real
    structure via naming-convention heuristics — no hand-coding, no domain
    knowledge. Does NOT touch Neo4j. Review the result before calling
    materialize_graph_model with the same schema. data_modeler role only.

    Args:
        schema: Iceberg schema to infer a graph model from.
    """
    with _client() as c:
        resp = c.get("/graph/infer", params={"schema": schema})
    _raise_for_gateway_error(resp)
    return resp.json()


@mcp.tool()
def materialize_graph_model(schema: str, limit: int = 100_000) -> dict[str, Any]:
    """Infers the graph model from `schema` (same as infer_graph_model) and MERGEs
    it into Neo4j — idempotent, generic, driven by the inferred model rather than
    hand-written per-table Cypher. data_modeler role only.

    Args:
        schema: Iceberg schema to infer and materialize a graph model from.
        limit: Per-table row cap.
    """
    with _client() as c:
        resp = c.post("/graph/infer/materialize", json={"schema": schema, "limit": limit})
    _raise_for_gateway_error(resp)
    return resp.json()


@mcp.tool()
def graph_freshness(schema: str) -> dict[str, Any]:
    """When a schema's graph was last built/materialized, and its node/relationship
    counts at that time. graph_stats has no notion of staleness on its own — check
    this before trusting the graph for anything time-sensitive. Empty dict if it's
    never been built (or the catalog is unreachable) — that's a real signal, not
    an error: it means build_knowledge_graph or materialize_graph_model hasn't run.

    Args:
        schema: Iceberg schema whose graph build history to check.
    """
    with _client() as c:
        resp = c.get("/graph/freshness", params={"schema": schema})
    _raise_for_gateway_error(resp)
    return resp.json()


@mcp.tool()
def get_graph_schema() -> dict[str, Any]:
    """Get the live graph schema for writing Cypher: labels, properties, and
    relationship direction — verified against the actual running graph, not
    asserted — enriched with human-written descriptions from the data catalog
    where available (property meanings, valid enum values, disambiguation).
    Call this before writing Cypher rather than relying on assumed schema."""
    with _client() as c:
        resp = c.get("/graph/schema")
    _raise_for_gateway_error(resp)
    return resp.json()


if __name__ == "__main__":
    mcp.run(transport="stdio")
