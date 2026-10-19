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
# Local-LLM calls (ontology inference, /graph/nl repair loop) can run several
# minutes on a laptop-sized model — 120s cut them off mid-answer.
_TIMEOUT = httpx.Timeout(600.0)

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
    """Infer a graph ontology (nodes, relationships, business descriptions) from a
    schema's real structure AND sample rows, using the local LLM. Does not change the
    graph — the result is saved as a numbered draft; its `ontology_version` is what
    you pass to materialize_graph_model so exactly this reviewed model gets built.
    data_modeler role only.

    Args:
        schema: Iceberg schema to infer a graph model from.
    """
    with _client() as c:
        resp = c.get("/graph/infer", params={"schema": schema})
    _raise_for_gateway_error(resp)
    return resp.json()


@mcp.tool()
def materialize_graph_model(schema: str, ontology_version: int | None = None, limit: int = 100_000) -> dict[str, Any]:
    """MERGE a reviewed ontology draft into Neo4j and mark it active — idempotent.
    Always pass the ontology_version returned by infer_graph_model: inference isn't
    reproducible run to run, so omitting it infers (and commits) a fresh, unreviewed
    model. data_modeler role only.

    Args:
        schema: Iceberg schema the ontology was inferred from.
        ontology_version: Draft version from infer_graph_model.
        limit: Per-table row cap.
    """
    with _client() as c:
        resp = c.post(
            "/graph/infer/materialize",
            json={"schema": schema, "limit": limit, "ontology_version": ontology_version},
        )
    _raise_for_gateway_error(resp)
    return resp.json()


@mcp.tool()
def graph_freshness(schema: str) -> dict[str, Any]:
    """When a schema's graph was last built/materialized, when its source data was
    last written through the gateway, and `stale`: true if the data changed after
    the build (false if not, null if unknown — e.g. never built). graph_stats has
    no notion of staleness on its own — check this before trusting the graph for
    anything time-sensitive.

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


@mcp.tool()
def ask_graph_nl(question: str, schema: str | None = None, answer: bool = True) -> dict[str, Any]:
    """Ask the knowledge graph a question in plain English, answered by the LOCAL LLM
    (no data leaves the machine): it writes Cypher grounded on the live schema, the
    active ontology and verified examples, validates it with EXPLAIN, repairs it if
    needed, runs it read-only, and (if answer=true) writes a short answer from the
    rows. Returns the Cypher, rows, attempts, and the answer — check the Cypher
    before trusting the answer.

    Args:
        question: The question, in plain English.
        schema: Restrict to one source schema's part of the graph (e.g. "ecommerce").
        answer: Also return a written answer from the results.
    """
    with _client() as c:
        resp = c.post("/graph/nl", json={"question": question, "schema": schema, "answer": answer})
    _raise_for_gateway_error(resp)
    return resp.json()


@mcp.tool()
def add_graph_example(question: str, cypher: str) -> dict[str, Any]:
    """Add a verified question -> Cypher pair to the few-shot library ask_graph_nl
    learns from. Only add queries you have checked return the right answer.
    data_modeler role only.

    Args:
        question: The plain-English question.
        cypher: A Cypher query verified to answer it correctly.
    """
    with _client() as c:
        resp = c.post("/graph/nl/examples", json={"question": question, "cypher": cypher})
    _raise_for_gateway_error(resp)
    return resp.json()
