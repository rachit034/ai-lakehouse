"""
Infers a knowledge graph model by having the LLM read a schema's real
structure *and* a sample of real rows, and reason about what the tables and
columns actually mean — not naming-convention pattern matching. This is the
generic replacement for graph_store.build_graph() (hardcoded to one dataset's
shape) and for this file's earlier regex/FK-naming heuristic, which could
only catch a relationship if a column was literally named "{table}_id" —
it had no way to recognize that `bundle_candidates.anchor_product` (a
product *name*, not an id) refers to a Product, something obvious from a
few sample rows but invisible from the schema alone.

Real sample rows go into the prompt, which is why this stays entirely on the
local Ollama model (never a third-party API) and why /graph/infer and
/graph/infer/materialize are data_modeler-only in routers/graph.py — seeing
raw data to design the model is the same legitimate-need bypass that role
already gets on query results (core/redaction.py).
"""
from __future__ import annotations

import json
import re

import mlflow
from mlflow.entities import SpanType

from core.config import settings
from services import catalog, datahub_client, graph_store, sql_engine
from services.llm import _call_ollama

_PAGE_SIZE = 1000  # matches sql_engine.execute()'s cap — see graph_store._rows
_SAMPLE_ROWS = 5
_MAX_VALUE_CHARS = 120  # truncate long text values so the prompt stays small

_ONTOLOGY_PROMPT = """You are a data modeler inferring a property-graph model (for Neo4j) from a \
relational schema. Design nodes and relationships based on what the tables and the ACTUAL sample \
data mean — not just column-naming patterns like "_id" suffixes.

Schema (tables and columns):
{schema}

Sample rows (a few real rows per table — look at the actual values, not just column names):
{samples}

Respond with a JSON object in exactly this shape:

{{
  "nodes": [
    {{"label": "Customer", "kind": "table", "table": "customers", "pk_column": "customer_id",
      "properties": ["customer_id", "first_name", "email"]}},
    {{"label": "Category", "kind": "value", "table": "products", "value_column": "category",
      "node_property": "name"}}
  ],
  "relationships": [
    {{"type": "PLACED", "source_table": "orders",
      "from": {{"kind": "id_lookup", "label": "Customer", "row_column": "customer_id"}},
      "to":   {{"kind": "self", "label": "Order"}},
      "properties": []}},
    {{"type": "CONTAINS", "source_table": "order_items",
      "from": {{"kind": "id_lookup", "label": "Order", "row_column": "order_id"}},
      "to":   {{"kind": "id_lookup", "label": "Product", "row_column": "product_id"}},
      "properties": ["quantity", "unit_price"]}},
    {{"type": "IN_CATEGORY", "source_table": "products",
      "from": {{"kind": "self", "label": "Product"}},
      "to":   {{"kind": "value_lookup", "label": "Category", "row_column": "category",
                "match_property": "name"}},
      "properties": []}},
    {{"type": "BUNDLED_WITH", "source_table": "bundle_candidates",
      "from": {{"kind": "value_lookup", "label": "Product", "row_column": "anchor_product",
                "match_property": "name"}},
      "to":   {{"kind": "value_lookup", "label": "Product", "row_column": "addon_product",
                "match_property": "name"}},
      "properties": ["bundle_price", "rationale"]}}
  ]
}}

Rules:
- Your ENTIRE response must be exactly one JSON object with exactly two top-level keys, "nodes"
  and "relationships" — start with {{ and end with }}. Do NOT respond with a bare JSON array, and
  do NOT omit "relationships" even if you think a schema has none — use an empty list [] instead.
- "kind": "table" nodes = one node per row of a real table, keyed on its primary key. Give every
  table that represents a real-world entity (a person, a product, an order) a "table" node.
- "kind": "value" nodes = promote a column's distinct values into their own node type. Only for
  genuinely low-cardinality, entity-like columns (a category, a status, a supplier) — never for
  something closer to unique-per-row (an email, a free-text description, a timestamp).
- A table whose rows represent a fact/association between two OTHER entities rather than an
  entity itself (a join table like order_items, or a table like bundle_candidates pairing two
  products) should NOT get its own "table" node — it should only produce relationships, with any
  extra columns (quantity, price, rationale) carried as relationship properties instead.
- A relationship's "from"/"to" is one of:
  - "self": this row IS that node (only valid when source_table is that node's own table)
  - "id_lookup": row_column's value matches another node's pk_column (a classic foreign key) —
    no "match_property" needed, it's always that node's own pk_column.
  - "value_lookup": row_column's value matches another node's property, named by "match_property"
    (e.g. a column holding a product NAME matches Product's "name" property) — use this whenever
    the sample data shows a column clearly referencing another entity by a human-readable value
    instead of an id. This is exactly the case naming-convention heuristics miss; look at the
    real values to catch it. "match_property" MUST be one of that target node's declared
    properties (its "properties" list for a "table" node, or its "node_property" for a "value"
    node) — never invent a property name that isn't declared on the target node.
- Every table in the schema should be accounted for, as either a "table" node, a "value" node
  source, or a relationship's source_table (or more than one of these).
- Return ONLY the JSON object — no explanation, no markdown fences.
"""

_FIX_PROMPT = """Your previous response did not match the required format. Here is what you \
returned:

{previous_output}

This failed for the following reason:
{error}

Return a corrected response: a single JSON object with exactly two top-level keys, "nodes" and \
"relationships" (see the original instructions for the exact shape of each). Fix only the \
structural problem described above — keep your actual modeling decisions (which columns became \
which node types, which relationships you found) unless the error is specifically about one of \
those. Return ONLY the corrected JSON object — no explanation, no markdown fences.
"""


def _sample_rows(schema: str, table: str, n: int = _SAMPLE_ROWS) -> list[dict]:
    result = sql_engine.execute(f"SELECT * FROM iceberg.{schema}.{table} LIMIT {n}", schema=schema)
    rows = []
    for row in result["rows"]:
        rec = {}
        for col, val in zip(result["columns"], row):
            if isinstance(val, str) and len(val) > _MAX_VALUE_CHARS:
                val = val[:_MAX_VALUE_CHARS] + "…"
            rec[col] = val
        rows.append(rec)
    return rows


def _distinct_values(schema: str, table: str, column: str, limit: int = _PAGE_SIZE) -> list[str]:
    result = sql_engine.execute(
        f'SELECT DISTINCT "{column}" FROM iceberg.{schema}.{table} '
        f'WHERE "{column}" IS NOT NULL LIMIT {limit}',
        schema=schema,
    )
    return [row[0] for row in result["rows"]]


def _parse_ontology_json(raw: str) -> dict:
    """Strip markdown fences / stray prose, then parse the JSON object."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in the model LLM's output:\n{raw[:500]}")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"Model LLM output was not valid JSON: {e}\nOutput:\n{text[:800]}") from e


def _validate_model(model: dict, columns_by_table: dict[str, list[dict]]) -> None:
    """Fail loudly and specifically rather than materializing a broken graph."""
    if not isinstance(model.get("nodes"), list) or not isinstance(model.get("relationships"), list):
        raise ValueError(f"Model must have 'nodes' and 'relationships' lists, got: {model}")

    tables = set(columns_by_table)
    labels: dict[str, dict] = {}
    for node in model["nodes"]:
        label = node.get("label")
        if not label:
            raise ValueError(f"Node missing 'label': {node}")
        if label in labels:
            raise ValueError(f"Duplicate node label '{label}'")
        if node.get("kind") == "table":
            if node.get("table") not in tables:
                raise ValueError(f"Node '{label}' references unknown table {node.get('table')!r}")
            cols = {c["name"] for c in columns_by_table[node["table"]]}
            if node.get("pk_column") not in cols:
                raise ValueError(f"Node '{label}' pk_column {node.get('pk_column')!r} not in {node['table']}")
            for p in node.get("properties", []):
                if p not in cols:
                    raise ValueError(f"Node '{label}' property {p!r} not in table {node['table']}")
        elif node.get("kind") == "value":
            if node.get("table") not in tables:
                raise ValueError(f"Node '{label}' references unknown table {node.get('table')!r}")
            cols = {c["name"] for c in columns_by_table[node["table"]]}
            if node.get("value_column") not in cols:
                raise ValueError(f"Node '{label}' value_column {node.get('value_column')!r} not in {node['table']}")
        else:
            raise ValueError(f"Node '{label}' has unknown kind {node.get('kind')!r}")
        labels[label] = node

    for rel in model["relationships"]:
        if rel.get("source_table") not in tables:
            raise ValueError(f"Relationship {rel.get('type')} references unknown table {rel.get('source_table')!r}")
        row_cols = {c["name"] for c in columns_by_table[rel["source_table"]]}
        for end_name in ("from", "to"):
            end = rel.get(end_name)
            if not end or end.get("label") not in labels:
                raise ValueError(f"Relationship {rel.get('type')}.{end_name} references unknown label {end}")
            if end.get("kind") in ("id_lookup", "value_lookup") and end.get("row_column") not in row_cols:
                raise ValueError(
                    f"Relationship {rel.get('type')}.{end_name}.row_column "
                    f"{end.get('row_column')!r} not in {rel['source_table']}"
                )
            if end.get("kind") == "value_lookup":
                target = labels[end["label"]]
                valid_props = (
                    set(target.get("properties", [])) | {target.get("pk_column")}
                    if target["kind"] == "table"
                    else {target.get("node_property")}
                )
                if end.get("match_property") not in valid_props:
                    raise ValueError(
                        f"Relationship {rel.get('type')}.{end_name}.match_property "
                        f"{end.get('match_property')!r} is not a declared property of node "
                        f"'{end['label']}' (valid: {sorted(p for p in valid_props if p)})"
                    )
        for p in rel.get("properties", []):
            if p not in row_cols:
                raise ValueError(f"Relationship {rel.get('type')} property {p!r} not in {rel['source_table']}")


@mlflow.trace(name="graph_inference.infer_model", span_type=SpanType.CHAIN)
async def infer_graph_model(schema: str) -> dict:
    """
    Read-only — does not touch Neo4j. Returns {"nodes": [...], "relationships":
    [...]} — see _ONTOLOGY_PROMPT for the exact shape.

    temperature=0 means a bare retry of the same prompt reproduces the same
    output — no help if the model's *reasoning* was fine but it got the JSON
    envelope wrong (e.g. returned a bare array instead of {"nodes":...,
    "relationships":...}, which qwen3:4b did in testing). So on a parse/
    validation failure, one self-correction pass shows the model its own
    output plus the specific error and asks it to fix the structural problem
    — a different input, so determinism doesn't block progress. Gives up
    after that rather than looping indefinitely.
    """
    tables = catalog.list_tables(schema)
    columns_by_table = {t: catalog.get_columns(schema, t) for t in tables}
    samples_by_table = {t: _sample_rows(schema, t) for t in tables}
    # Best-effort — empty dict per table if DataHub has nothing/is unreachable,
    # same degrade-gracefully contract as everywhere else datahub_client is used.
    enrichment_by_table = {
        t: datahub_client.get_table_context(settings.iceberg_catalog_name, schema, t) for t in tables
    }

    def _col_str(c: dict, table_enrichment: dict) -> str:
        info = table_enrichment.get("columns", {}).get(c["name"], {})
        s = f"{c['name']} {c['type']}"
        if info.get("description"):
            s += f" [{info['description']}]"
        return s

    # Human-written descriptions matter here specifically: a column
    # description like "cumulative revenue attributed to this customer" is
    # exactly the kind of hint that turns a guessable relationship into a
    # confident one, the same way it already helps ask_nl.
    schema_lines = []
    for t, cols in columns_by_table.items():
        enrichment = enrichment_by_table[t]
        header = t
        if enrichment.get("description"):
            header += f" — {enrichment['description']}"
        schema_lines.append(f"{header}({', '.join(_col_str(c, enrichment) for c in cols)})")
    schema_text = "\n".join(schema_lines)
    samples_text = "\n".join(
        f"{t}:\n" + "\n".join(f"  {row}" for row in rows)
        for t, rows in samples_by_table.items()
    )

    prompt = _ONTOLOGY_PROMPT.format(schema=schema_text, samples=samples_text)
    # Ontology reasoning over a full schema + samples is heavier than a single
    # NL->SQL translation — give it real budget rather than _call_ollama's
    # 120s default (tuned for the much lighter query-translation case).
    raw = await _call_ollama(prompt, timeout=300)
    try:
        model = _parse_ontology_json(raw)
        _validate_model(model, columns_by_table)
        return model
    except ValueError as first_error:
        mlflow.update_current_trace(tags={"self_correction": "attempted"})
        fix_prompt = _FIX_PROMPT.format(previous_output=raw, error=first_error)
        raw2 = await _call_ollama(fix_prompt, timeout=300)
        try:
            model = _parse_ontology_json(raw2)
            _validate_model(model, columns_by_table)
            mlflow.update_current_trace(tags={"self_correction": "succeeded"})
            return model
        except ValueError as second_error:
            mlflow.update_current_trace(tags={"self_correction": "failed"})
            raise ValueError(
                f"Model failed to produce a valid graph model even after one self-correction "
                f"attempt. First error: {first_error}. Second error: {second_error}"
            ) from second_error


@mlflow.trace(name="graph_inference.materialize", span_type=SpanType.WORKFLOW)
async def materialize_inferred_graph(schema: str, limit: int = 100_000) -> dict:
    """Infers the model, then MERGEs it into Neo4j — idempotent."""
    model = await infer_graph_model(schema)
    nodes_by_label = {n["label"]: n for n in model["nodes"]}

    with graph_store._get_driver().session() as session:
        # Node identity is scoped per schema via _node_key (see graph_store.py's
        # module docstring for the full rationale) — this is what stopped this
        # feature's own test run from overwriting real ecommerce Product data
        # when a test schema happened to reuse the same product_id values.
        # Relationship MATCHing below still uses the real semantic property
        # (pk_column / match_property), scoped by _schema, NOT _node_key —
        # _node_key is always pk/value-based, so a value_lookup matching on a
        # different property (e.g. Product.name) couldn't reconstruct it.

        # 1. Constraints + nodes, "table" kind first (real rows), then "value"
        #    kind (promoted columns) — relationships below assume both exist.
        for node in model["nodes"]:
            session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node['label']}) REQUIRE n._node_key IS UNIQUE")

        for node in model["nodes"]:
            if node["kind"] != "table":
                continue
            rows = graph_store._rows(schema, node["table"], node["pk_column"], limit)
            # Always set pk_column as a real property, even if the model's own
            # "properties" list happened to omit it (observed varying between
            # otherwise-identical inference runs) — the node's own id should
            # always be queryable directly, not only reconstructable from
            # _node_key.
            props = list({node["pk_column"], *node["properties"]})
            set_clause = ", ".join(f"n.{p} = $row.{p}" for p in props)
            for row in rows:
                session.run(
                    f"MERGE (n:{node['label']} {{_node_key: $node_key}}) SET n._schema = $schema, {set_clause}",
                    row=row, node_key=f"{schema}:{row[node['pk_column']]}", schema=schema,
                )

        for node in model["nodes"]:
            if node["kind"] != "value":
                continue
            values = _distinct_values(schema, node["table"], node["value_column"], limit)
            for v in values:
                session.run(
                    f"MERGE (n:{node['label']} {{_node_key: $node_key}}) "
                    f"SET n._schema = $schema, n.{node['node_property']} = $v",
                    node_key=f"{schema}:{v}", schema=schema, v=v,
                )

        # 2. Relationships — iterate each relationship's source_table once,
        #    resolving "from"/"to" per-row via whichever kind it declares.
        def _match_clause(var: str, end: dict) -> str:
            label = end["label"]
            target = nodes_by_label[label]
            match_prop = target["pk_column"] if end["kind"] in ("self", "id_lookup") else end["match_property"]
            return f"({var}:{label} {{{match_prop}: ${var}_value, _schema: $schema}})"

        def _end_value(end: dict, row: dict) -> object:
            if end["kind"] == "self":
                return row[nodes_by_label[end["label"]]["pk_column"]]
            return row[end["row_column"]]

        for rel in model["relationships"]:
            # Order rows by a real pk if source_table is itself a "table" node,
            # otherwise any deterministic column works — paging just needs
            # rows not to overlap or skip between pages, not a "real" key.
            order_col = next(
                (n["pk_column"] for n in model["nodes"] if n.get("table") == rel["source_table"] and n["kind"] == "table"),
                None,
            ) or catalog.get_columns(schema, rel["source_table"])[0]["name"]
            rows = graph_store._rows(schema, rel["source_table"], order_col, limit)

            from_clause = _match_clause("a", rel["from"])
            to_clause = _match_clause("b", rel["to"])
            set_clause = (
                ("SET " + ", ".join(f"r.{p} = $row.{p}" for p in rel["properties"])) if rel["properties"] else ""
            )
            for row in rows:
                session.run(
                    f"MATCH {from_clause} MATCH {to_clause} MERGE (a)-[r:{rel['type']}]->(b) {set_clause}",
                    row=row,
                    a_value=_end_value(rel["from"], row),
                    b_value=_end_value(rel["to"], row),
                    schema=schema,
                )

    # Scoped to this schema, not graph_stats()'s global totals — same
    # reasoning as graph_store.build_graph()'s equivalent tracking call.
    scoped = graph_store.run_cypher(
        "MATCH (n {_schema: $schema}) WITH count(n) AS nodes "
        "MATCH (a {_schema: $schema})-[r]->(b {_schema: $schema}) "
        "RETURN nodes, count(r) AS relationships",
        {"schema": schema},
    )
    counts = scoped[0] if scoped else {"nodes": 0, "relationships": 0}
    datahub_client.record_graph_build(schema, counts["nodes"], counts["relationships"])
    return {"model": model, "stats": graph_store.graph_stats()}
