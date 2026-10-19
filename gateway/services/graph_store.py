"""
Knowledge graph over the lakehouse's structured data, via Neo4j.
In production this points at Amazon Neptune instead.

The graph is derived from Iceberg tables (read through DuckDB) rather than
stored independently — build_graph() is idempotent (MERGE) and can be re-run
after new data lands.

Node identity is scoped per source schema via two internal properties every
node gets, set here and in graph_inference.py's materialize_inferred_graph():
  _node_key: f"{schema}:{pk_or_value}" — the actual MERGE/uniqueness-constraint
             target, so two schemas producing the same label with the same
             primary-key value (e.g. both have a product_id=1) become two
             separate nodes instead of silently overwriting each other.
  _schema:   the source schema name, plain — used to scope relationship
             MATCH queries (e.g. "Category name=Electronics AND _schema=X")
             so a value-based match can't accidentally bind to a same-named
             entity from an unrelated schema.
Both are internal bookkeeping — build_schema_context() below filters
_node_key out of what it shows the graph_agent, since it's not something
anyone should write Cypher against directly (query real properties like
product_id/name instead, optionally + _schema if you need to disambiguate).
"""
from __future__ import annotations

import re
from collections import defaultdict

import mlflow
from mlflow.entities import SpanType
from neo4j import GraphDatabase

from core.config import settings
from services import datahub_client, metrics, sql_engine

_driver = None


def _get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
    return _driver


_PAGE_SIZE = 1000  # sql_engine.execute() caps fetchmany at 1000 regardless of SQL LIMIT


def _rows(schema: str, table: str, order_by: str, limit: int) -> list[dict]:
    """Fetch up to `limit` rows from an Iceberg table, paging around the
    1000-row cap in sql_engine.execute(). Ordered by `order_by` so pages
    don't overlap or skip rows between calls."""
    rows: list[dict] = []
    offset = 0
    while len(rows) < limit:
        page_limit = min(_PAGE_SIZE, limit - len(rows))
        result = sql_engine.execute(
            f"SELECT * FROM iceberg.{schema}.{table} "
            f"ORDER BY {order_by} OFFSET {offset} LIMIT {page_limit}",
            schema=schema,
        )
        batch = [dict(zip(result["columns"], row)) for row in result["rows"]]
        rows.extend(batch)
        if len(batch) < page_limit:
            break
        offset += page_limit
    return rows


# ── Build ─────────────────────────────────────────────────────────────────

@mlflow.trace(name="graph_store.build_graph", span_type=SpanType.WORKFLOW)
def build_graph(schema: str = "ecommerce", limit: int = 100_000) -> dict:
    """
    Build/refresh the knowledge graph from the ecommerce Iceberg tables:

    (:Customer)-[:PLACED]->(:Order)-[:CONTAINS {quantity, unit_price}]->(:Product)
    (:Product)-[:IN_CATEGORY]->(:Category)
    (:Product)-[:SUPPLIED_BY]->(:Supplier)

    Node identity is scoped by `schema` (see _node_key docstring below) —
    two different Iceberg schemas both producing a Product with product_id=1
    become two distinct Neo4j nodes, not one MERGEd/overwritten node. Before
    this, testing graph_inference.py against a schema whose primary keys
    happened to overlap with this one's silently overwrote real product data.

    `limit` is a per-table safety cap, not a real-world constraint — it's paged
    around sql_engine.execute()'s 1000-row-per-call limit, so raise it if your
    tables grow past 100k rows. Returns actual node/relationship counts from
    Neo4j after the load (see graph_stats), not attempted-write counts.
    """
    customers = _rows(schema, "customers", "customer_id", limit)
    products = _rows(schema, "products", "product_id", limit)
    orders = _rows(schema, "orders", "order_id", limit)
    order_items = _rows(schema, "order_items", "item_id", limit)

    with _get_driver().session() as session:
        # Old single-property constraints (customer_id, product_id, ...) would
        # block a second schema from ever having the same id — drop them in
        # favor of a constraint on the schema-scoped _node_key instead.
        for old in ("customer_id", "product_id", "order_id", "category_name", "supplier_name"):
            session.run(f"DROP CONSTRAINT {old} IF EXISTS")
        for label in ("Customer", "Product", "Order", "Category", "Supplier"):
            session.run(f"CREATE CONSTRAINT {label.lower()}_key IF NOT EXISTS FOR (n:{label}) REQUIRE n._node_key IS UNIQUE")

        for c in customers:
            session.run(
                """
                MERGE (n:Customer {_node_key: $node_key})
                SET n._schema = $schema, n.customer_id = $customer_id,
                    n.first_name = $first_name, n.last_name = $last_name,
                    n.email = $email, n.country = $country, n.city = $city,
                    n.segment = $segment, n.age_group = $age_group,
                    n.lifetime_value = $lifetime_value
                """,
                node_key=f"{schema}:{c['customer_id']}", schema=schema, **c,
            )

        for p in products:
            session.run(
                """
                MERGE (n:Product {_node_key: $node_key})
                SET n._schema = $schema, n.product_id = $product_id, n.name = $name,
                    n.category = $category, n.subcategory = $subcategory,
                    n.price = $price, n.cost = $cost, n.margin_pct = $margin_pct,
                    n.stock_quantity = $stock_quantity, n.is_active = $is_active
                MERGE (cat:Category {_node_key: $category_key})
                SET cat._schema = $schema, cat.name = $category
                MERGE (n)-[:IN_CATEGORY]->(cat)
                MERGE (sup:Supplier {_node_key: $supplier_key})
                SET sup._schema = $schema, sup.name = $supplier
                MERGE (n)-[:SUPPLIED_BY]->(sup)
                """,
                node_key=f"{schema}:{p['product_id']}",
                category_key=f"{schema}:{p['category']}",
                supplier_key=f"{schema}:{p['supplier']}",
                schema=schema, **p,
            )

        for o in orders:
            session.run(
                """
                MERGE (n:Order {_node_key: $node_key})
                SET n._schema = $schema, n.order_id = $order_id, n.order_date = $order_date,
                    n.status = $status, n.payment_method = $payment_method,
                    n.total_amount = $total_amount, n.item_count = $item_count
                WITH n
                MATCH (c:Customer {_node_key: $customer_key})
                MERGE (c)-[:PLACED]->(n)
                """,
                node_key=f"{schema}:{o['order_id']}",
                customer_key=f"{schema}:{o['customer_id']}",
                schema=schema, **o,
            )

        for item in order_items:
            session.run(
                """
                MATCH (o:Order {_node_key: $order_key})
                MATCH (p:Product {_node_key: $product_key})
                MERGE (o)-[r:CONTAINS]->(p)
                SET r.quantity = $quantity, r.unit_price = $unit_price
                """,
                order_key=f"{schema}:{item['order_id']}",
                product_key=f"{schema}:{item['product_id']}",
                quantity=item["quantity"], unit_price=item["unit_price"],
            )

    # Scoped to this schema, not graph_stats()'s global totals — the graph
    # can hold more than one schema's nodes now (see _node_key), and
    # recording another schema's counts against this one would be wrong.
    scoped = run_cypher(
        "MATCH (n {_schema: $schema}) WITH count(n) AS nodes "
        "MATCH (a {_schema: $schema})-[r]->(b {_schema: $schema}) "
        "RETURN nodes, count(r) AS relationships",
        {"schema": schema},
    )
    counts = scoped[0] if scoped else {"nodes": 0, "relationships": 0}
    datahub_client.record_graph_build(schema, counts["nodes"], counts["relationships"])
    return graph_stats()


def get_build_freshness(schema: str) -> dict:
    """{"last_built": iso str, "node_count": int, "relationship_count": int}
    for this schema's graph, or {} if it's never been built or DataHub is
    unreachable — see datahub_client.get_graph_build_info."""
    return datahub_client.get_graph_build_info(schema)


# ── Query ────────────────────────────────────────────────────────────────

@mlflow.trace(name="neo4j.run_cypher", span_type=SpanType.RETRIEVER)
def run_cypher(cypher: str, params: dict | None = None) -> list[dict]:
    """
    Run Cypher in a READ transaction — Neo4j itself rejects any write clause
    (CREATE/MERGE/SET/DELETE/etc.) with "Writing in read access mode not
    allowed", so this is enforced by the database, not by guessing at Cypher
    grammar client-side. build_graph() above needs writes, so it uses
    session.run() directly instead of going through this function.
    """

    def _read(tx):
        result = tx.run(cypher, params or {})
        return [record.data() for record in result]

    with _get_driver().session() as session:
        return session.execute_read(_read)


@mlflow.trace(name="graph_store.graph_stats", span_type=SpanType.TOOL)
def graph_stats() -> dict:
    node_counts = run_cypher(
        "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC"
    )
    rel_counts = run_cypher(
        "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY count DESC"
    )
    return {"nodes": node_counts, "relationships": rel_counts}


# ── Schema context (for graph_agent's Cypher generation, mirrors
#    catalog.build_schema_context() on the SQL side) ────────────────────────

@mlflow.trace(name="graph_store.build_schema_context", span_type=SpanType.RETRIEVER)
def build_schema_context() -> str:
    """
    Live-introspected graph schema (labels, relationship direction verified
    against real data, not asserted) enriched with DataHub descriptions.

    Direction is the single most common failure mode found when evaluating
    LLM-generated Cypher against this graph — a model writing
    (c:Customer)<-[:PLACED]-(o:Order) instead of the correct
    (c:Customer)-[:PLACED]->(o:Order) — because a hand-typed schema comment
    only *asserts* direction, it doesn't verify it. Sourcing this from
    `MATCH (a)-[r]->(b)` against the live graph makes the direction a fact
    about the running database, not a claim a prompt author could get wrong
    or let drift out of sync after a schema change.
    """
    node_props: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rec in run_cypher("CALL db.schema.nodeTypeProperties()"):
        if rec["propertyName"] == "_node_key":  # internal — see module docstring
            continue
        label = rec["nodeLabels"][0] if rec["nodeLabels"] else "?"
        ptype = (rec["propertyTypes"] or ["ANY"])[0]
        node_props[label].append((rec["propertyName"], ptype))

    rel_props: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rec in run_cypher("CALL db.schema.relTypeProperties()"):
        rel_type = re.sub(r"[`:]", "", rec["relType"])
        if rec["propertyName"]:
            ptype = (rec["propertyTypes"] or ["ANY"])[0]
            rel_props[rel_type].append((rec["propertyName"], ptype))

    connectivity = run_cypher(
        "MATCH (a)-[r]->(b) RETURN DISTINCT labels(a)[0] AS from_label, "
        "type(r) AS rel_type, labels(b)[0] AS to_label"
    )

    lines = ["Nodes:"]
    for label in sorted(node_props):
        enrichment = datahub_client.get_graph_node_context(label)
        prop_info = enrichment.get("properties", {})

        header = f"  (:{label})"
        if enrichment.get("description"):
            header += f" — {enrichment['description']}"
        lines.append(header)

        for prop, ptype in sorted(set(node_props[label])):
            info = prop_info.get(prop, {})
            prop_line = f"    {prop} ({ptype})"
            if info.get("glossary_terms"):
                prop_line += f" [aka: {', '.join(info['glossary_terms'])}]"
            if info.get("description"):
                prop_line += f": {info['description']}"
            lines.append(prop_line)

    lines.append("\nRelationships (direction verified against live data, not asserted):")
    for rec in connectivity:
        rel = rec["rel_type"]
        prop_str = ""
        if rel_props.get(rel):
            prop_str = f" {{{', '.join(f'{p} ({t})' for p, t in sorted(set(rel_props[rel])))}}}"
        lines.append(f"  (:{rec['from_label']})-[:{rel}{prop_str}]->(:{rec['to_label']})")

    metrics_block = metrics.build_metrics_context("cypher")
    if metrics_block:
        lines.append("")
        lines.append(metrics_block)

    return "\n".join(lines)
