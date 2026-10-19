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

# Column/key names are trivial to dodge — `RETURN c.email AS x` or
# `SELECT email AS x` renames the field the redactor keys off of. Email has a
# distinctive enough format to catch by VALUE regardless of column name, so
# every string value gets scanned for that shape as a second pass. First/last
# names have no such format (any short word could be one) — there's no
# reliable value-based catch for those, so an aliased `first_name AS x` *will*
# slip through unredacted. Column-name matching is a real but incomplete
# guardrail, not a hardened one; it stops the default case, not someone
# deliberately trying to route around it.
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


def redact_rows(columns: list[str], rows: list[list], schema: str | set[str] | None = None) -> list[list]:
    """Redact PII columns in SQL-style {columns, rows} results (/query/sql, /query/nl).

    `schema` (a schema name, or a set of them for a multi-schema join) pulls
    in DataHub-tagged Consumer.Modeler/Engineer columns for those schemas'
    tables in addition to the static PII_FIELDS — see module docstring.
    """
    dynamic = {f.lower() for f in _catalog_masked_fields(schema)}
    pii_idx = {i for i, c in enumerate(columns) if is_pii_field(c) or c.lower() in dynamic}
    redacted = []
    for row in rows:
        row = list(row)
        for i in range(len(row)):
            if i in pii_idx:
                row[i] = _mask(columns[i], row[i])
            else:
                row[i] = _mask_value_if_email_shaped(row[i])
        redacted.append(row)
    return redacted


def redact_records(records: list[dict], schema: str | None = None) -> list[dict]:
    """Redact PII keys in dict-style results (/query/vector, /graph/query).

    `schema` is optional and unused by current callers (neither vector
    datasets nor Cypher results carry a clean single Iceberg schema to key
    catalog lookups off of) — kept for API symmetry with redact_rows, and so
    a future caller that does have a schema can opt in without a signature
    change.
    """
    dynamic = {f.lower() for f in _catalog_masked_fields(schema)}
    return [
        {
            k: (_mask(k, v) if (is_pii_field(k) or k.lower() in dynamic) else _mask_value_if_email_shaped(v))
            for k, v in rec.items()
        }
        for rec in records
    ]
