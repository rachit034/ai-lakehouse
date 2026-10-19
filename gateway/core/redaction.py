"""
PII redaction for ad-hoc query responses.

Applied at the router layer (the API response boundary), not inside
services/sql_engine.py or services/graph_store.py — those are also used
internally for the knowledge graph build and vector-search embedding, where
real values are required for correct operation (a Customer node needs a real
name to join on; embedding a redacted string would embed junk). Redacting
there would silently corrupt those pipelines.

So: DuckDB and Neo4j hold real values. What's masked is the response handed
back to any caller — human, agent, or MCP tool — through /query/sql,
/query/nl, /query/vector, and /graph/query.

Two sources feed the redacted-field set:
  PII_FIELDS      static, hardcoded — the safety net if the catalog is down
                  or a table was never tagged.
  DataHub tags    dynamic, governance-driven (core/services/datahub_client.py
                  get_masked_columns) — any column tagged Consumer.Modeler
                  or Consumer.Engineer in the catalog is masked for business
                  too, without a code change. This is the actual point of
                  having a catalog: cost/margin_pct/lifetime_value are masked
                  today purely because someone tagged them, not because a
                  developer remembered to add them to PII_FIELDS.
"""
from __future__ import annotations

import re
import time

import sqlglot
from sqlglot import exp

from core.config import settings
from services import datahub_client

# Column/key names treated as PII, matched case-insensitively and exactly —
# not a substring match, so "product name" / "category name" aren't caught.
# This is the static fallback; see module docstring for the DataHub-driven
# dynamic path that supersedes it in practice.
PII_FIELDS = {
    "email",
    "first_name",
    "last_name",
    "phone",
    "phone_number",
    "ssn",
    "address",
    "street_address",
    "credit_card",
    "date_of_birth",
    "dob",
}

_catalog_cache: dict[str, tuple[float, set[str]]] = {}
_CATALOG_CACHE_TTL = 300  # seconds — same order as settings.nl_cache_ttl


def _masked_fields_for_schema(schema: str) -> set[str]:
    """Cached per-schema lookup — see _catalog_masked_fields for the public,
    multi-schema entry point callers should actually use."""
    now = time.monotonic()
    cached = _catalog_cache.get(schema)
    if cached and now - cached[0] < _CATALOG_CACHE_TTL:
        return cached[1]

    from services import catalog  # local import: catalog.py doesn't import this module, no cycle

    masked: set[str] = set()
    try:
        for table in catalog.list_tables(schema):
            masked |= datahub_client.get_masked_columns(settings.iceberg_catalog_name, schema, table)
    except Exception:
        masked = set()
    _catalog_cache[schema] = (now, masked)
    return masked


def _catalog_masked_fields(schemas: str | set[str] | None) -> set[str]:
    """Column names DataHub tags say the business role shouldn't see raw,
    unioned across every table in every given schema. Accepts a single
    schema string or a set (a query can reference more than one). Empty set
    if nothing given, DataHub unreachable, or nothing tagged — degrades to
    PII_FIELDS-only redaction either way."""
    if not schemas:
        return set()
    schema_set = {schemas} if isinstance(schemas, str) else schemas
    masked: set[str] = set()
    for s in schema_set:
        masked |= _masked_fields_for_schema(s)
    return masked

_EMAIL_RE = re.compile(r"^([^@]{1,3})[^@]*(@.+)$")

# Column/key names alone are trivial to dodge — `SELECT email AS x`,
# `concat(first_name, last_name)`, `RETURN c.first_name AS n` all rename the
# field. So the query text is traced too: any output column whose expression
# derives from a masked column (through aliases, functions, subqueries, CTEs,
# WITH clauses) is masked — see sql_derived_masked_columns() and
# cypher_derived_masked_keys(). count(...) of a masked column is exempt: a
# count isn't the value. As a last net, every string value is also scanned
# for email shape regardless of where it came from. The durable fix — the
# business role reading PII-masked views so raw columns are unreachable —
# is in docs/ARCHITECTURE-PROPOSAL.md.
_EMAIL_VALUE_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$")


def is_pii_field(field: str) -> bool:
    return field.lower() in PII_FIELDS


def _mask_email_text(text: str) -> str:
    m = _EMAIL_RE.match(text)
    return f"{m.group(1)}***{m.group(2)}" if m else "***@***"


def _mask(field: str, value):
    if value is None:
        return value
    text = str(value)
    field_l = field.lower()
    if field_l == "email":
        return _mask_email_text(text)
    if field_l in ("first_name", "last_name"):
        return f"{text[0]}***" if text else text
    return "***REDACTED***"


def _mask_value_if_email_shaped(value):
    if isinstance(value, str) and _EMAIL_VALUE_RE.match(value.strip()):
        return _mask_email_text(value.strip())
    return value


# ── Query-text lineage ───────────────────────────────────────────────────────

def _under_count(node: exp.Expression, stop: exp.Expression) -> bool:
    cur = node.parent
    while cur is not None and cur is not stop:
        if isinstance(cur, exp.Count):
            return True
        cur = cur.parent
    return False


def _refs_masked(proj: exp.Expression, masked: set[str]) -> bool:
    return any(
        col.name.lower() in masked and not _under_count(col, proj)
        for col in proj.find_all(exp.Column)
    )


def _output_positions(stmt: exp.Expression, masked: set[str]) -> list[bool] | None:
    """Per-output-column masked flags for the outermost query, or None when
    positions can't be known statically (SELECT *)."""
    if isinstance(stmt, exp.Union):
        left = _output_positions(stmt.left, masked)
        right = _output_positions(stmt.right, masked)
        if left is None or right is None or len(left) != len(right):
            return None
        return [a or b for a, b in zip(left, right)]
    if isinstance(stmt, exp.Subquery):
        return _output_positions(stmt.this, masked)
    if not isinstance(stmt, exp.Select):
        return None
    if any(isinstance(p, exp.Star) or (isinstance(p, exp.Column) and isinstance(p.this, exp.Star))
           for p in stmt.expressions):
        return None
    return [_refs_masked(p, masked) for p in stmt.expressions]


def sql_derived_masked_columns(sql: str, base_masked: set[str]) -> tuple[set[str], list[bool] | None]:
    """
    Returns (masked_names, masked_positions) for a SELECT:
      masked_names     every column name/alias anywhere in the query that
                       derives from a base masked column (fixpoint over all
                       SELECTs, so a CTE aliasing email AS x, selected again
                       by an outer query, still taints x)
      masked_positions per-output-column flags for the outermost SELECT, so
                       unaliased expressions like concat(first_name, ...) —
                       whose DuckDB column name can't be predicted — are still
                       caught; None when SELECT * makes positions unknowable
    Unparseable SQL returns the base set unchanged (name matching still applies).
    """
    masked = {m.lower() for m in base_masked}
    try:
        stmt = sqlglot.parse_one(sql, dialect="duckdb")
    except Exception:
        return masked, None
    if stmt is None:
        return masked, None

    changed = True
    while changed:
        changed = False
        for select in stmt.find_all(exp.Select):
            for proj in select.expressions:
                name = (proj.alias_or_name or "").lower()
                if name and name not in masked and _refs_masked(proj, masked):
                    masked.add(name)
                    changed = True
    return masked, _output_positions(stmt, masked)


_CYPHER_CLAUSE_RE = re.compile(
    r"\b(WITH|RETURN)\b(?:\s+DISTINCT\b)?(.*?)(?=\b(?:MATCH|OPTIONAL|WHERE|ORDER\s+BY|SKIP|LIMIT|WITH|RETURN|UNWIND|CALL|UNION|CREATE|MERGE|SET|DELETE)\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_CYPHER_ALIAS_RE = re.compile(r"\s+AS\s+`?(\w+)`?\s*$", re.IGNORECASE)


def _split_top_level(text: str) -> list[str]:
    items, depth, cur = [], 0, []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            items.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    items.append("".join(cur))
    return [i.strip() for i in items if i.strip()]


def cypher_derived_masked_keys(cypher: str, base_masked: set[str]) -> set[str]:
    """Result keys of a Cypher query that derive from masked properties,
    following aliases through WITH clauses to the final RETURN."""
    masked = {m.lower() for m in base_masked}
    keys: set[str] = set()
    without_count = re.compile(r"count\s*\([^()]*\)", re.IGNORECASE)
    for _, body in _CYPHER_CLAUSE_RE.findall(cypher):
        for item in _split_top_level(body):
            m = _CYPHER_ALIAS_RE.search(item)
            expr = item[: m.start()] if m else item
            name = (m.group(1) if m else item).strip()
            probe = without_count.sub("", expr)
            props = {p.lower() for p in re.findall(r"\.\s*`?(\w+)`?", probe)}
            bare = {w.lower() for w in re.findall(r"\b(\w+)\b", probe)}
            if props & masked or bare & masked:
                masked.add(name.lower())
                keys.add(name)
    return keys


# ── Redaction ────────────────────────────────────────────────────────────────

def _key_is_masked(key: str, masked: set[str]) -> bool:
    k = key.lower()
    return k in masked or k.rsplit(".", 1)[-1] in masked


def _redact_value(value, masked: set[str]):
    """Recursive: a Cypher `RETURN c` yields the whole node as a dict of
    properties, and collect()/lists nest further — mask PII keys at any depth."""
    if isinstance(value, dict):
        return {k: (_mask(k.rsplit(".", 1)[-1], v) if _key_is_masked(k, masked) and not isinstance(v, (dict, list))
                    else _redact_value(v, masked))
                for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(v, masked) for v in value]
    return _mask_value_if_email_shaped(value)


def redact_rows(
    columns: list[str],
    rows: list[list],
    schema: str | set[str] | None = None,
    sql: str | None = None,
) -> list[list]:
    """Redact PII columns in SQL-style {columns, rows} results (/query/sql, /query/nl).

    `schema` (a schema name, or a set of them for a multi-schema join) pulls
    in DataHub-tagged Consumer.Modeler/Engineer columns for those schemas'
    tables in addition to the static PII_FIELDS — see module docstring.
    `sql`, when given, extends masking to every output column derived from
    a masked column — see sql_derived_masked_columns().
    """
    base = PII_FIELDS | {f.lower() for f in _catalog_masked_fields(schema)}
    masked, positions = (sql_derived_masked_columns(sql, base) if sql else (base, None))
    pii_idx = {i for i, c in enumerate(columns) if _key_is_masked(c, masked)}
    if positions is not None and len(positions) == len(columns):
        pii_idx |= {i for i, flag in enumerate(positions) if flag}
    redacted = []
    for row in rows:
        row = list(row)
        for i in range(len(row)):
            if i in pii_idx:
                row[i] = _mask(columns[i], row[i])
            else:
                row[i] = _redact_value(row[i], masked)
        redacted.append(row)
    return redacted


def redact_records(records: list[dict], schema: str | None = None, cypher: str | None = None) -> list[dict]:
    """Redact PII keys in dict-style results (/query/vector, /graph/query, /graph/nl).

    Recursive (whole nodes returned as dicts are masked inside), matches
    dotted keys like "c.first_name" on their last segment, and — when the
    Cypher text is given — masks aliases derived from PII properties.
    """
    masked = PII_FIELDS | {f.lower() for f in _catalog_masked_fields(schema)}
    if cypher:
        masked = masked | {k.lower() for k in cypher_derived_masked_keys(cypher, masked)}
    return [_redact_value(rec, masked) for rec in records]
