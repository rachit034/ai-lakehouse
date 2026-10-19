"""
Business metrics registry — named, versioned definitions that both the
SQL surface (llm.py's NL->SQL) and the graph surface (graph_agent.py's
NL->Cypher) resolve from the same source, instead of each path re-deriving
"revenue" or "active customer" independently and risking two different
answers to what should be one question.

Modeled after the metrics-authoring/resolution layer in AWS's
context-ontology-accelerator, scaled down to what this prototype needs: no
separate authoring service, no formal versioning history — just one
registry, injected into both schema contexts, with a version number bumped
by hand when a definition changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Metric:
    name: str
    version: int
    description: str
    aliases: tuple[str, ...]
    sql: str | None      # None if this metric has no meaningful SQL form
    cypher: str | None   # None if this metric has no meaningful Cypher form


# {schema} is substituted with the target Iceberg schema (default "ecommerce",
# matching graph_store.build_graph's default — the same domain both surfaces
# model). Bump `version` when a definition's meaning changes, not its wording.
METRICS: tuple[Metric, ...] = (
    Metric(
        name="revenue",
        version=1,
        description="Completed-order revenue.",
        aliases=("sales", "gross revenue", "total revenue"),
        sql="SUM(total_amount) FROM iceberg.{schema}.orders WHERE status = 'completed'",
        cypher="MATCH (o:Order {{status: 'completed'}}) RETURN sum(o.total_amount)",
    ),
    Metric(
        name="average_order_value",
        version=1,
        description="Mean order total across completed orders.",
        aliases=("aov", "avg order value"),
        sql="AVG(total_amount) FROM iceberg.{schema}.orders WHERE status = 'completed'",
        cypher="MATCH (o:Order {{status: 'completed'}}) RETURN avg(o.total_amount)",
    ),
    Metric(
        name="active_customer",
        version=1,
        description="A customer with at least one order in the last 90 days.",
        aliases=("active customers", "recently active"),
        sql=(
            "customer_id FROM iceberg.{schema}.orders "
            "WHERE CAST(order_date AS DATE) >= current_date - INTERVAL 90 DAY"
        ),
        cypher=(
            "MATCH (c:Customer)-[:PLACED]->(o:Order) "
            "WHERE date(o.order_date) >= date() - duration('P90D') "
            "RETURN DISTINCT c"
        ),
    ),
)


def list_metrics() -> list[dict]:
    """Raw registry, for the /query/metrics endpoint and the list_metrics MCP tool."""
    return [
        {
            "name": m.name,
            "version": m.version,
            "description": m.description,
            "aliases": list(m.aliases),
            "sql": m.sql,
            "cypher": m.cypher,
        }
        for m in METRICS
    ]


def build_metrics_context(surface: Literal["sql", "cypher"], schema: str = "ecommerce") -> str:
    """
    Formatted block for prompt injection into catalog.build_schema_context()
    (surface="sql") or graph_store.build_schema_context() (surface="cypher").
    Skips metrics that don't define an expression for this surface. Empty
    string if none apply, so callers can just concatenate unconditionally.
    """
    lines: list[str] = []
    for m in METRICS:
        expr = m.sql if surface == "sql" else m.cypher
        if not expr:
            continue
        header = f"{m.name} (v{m.version})"
        if m.aliases:
            header += f" [aka: {', '.join(m.aliases)}]"
        header += f" — {m.description}"
        lines.append(header)
        lines.append(f"  {surface.upper()}: {expr.format(schema=schema)}")

    if not lines:
        return ""
    return "Business metrics — use these exact definitions when a question names one:\n" + "\n".join(lines)
