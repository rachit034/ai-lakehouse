"""
Reads AND writes DataHub metadata over its plain REST API — deliberately
not the acryl-datahub SDK, which pulls ~60 packages and 70MB+ into what's
otherwise a lean gateway image (no PyTorch, ONNX-only embeddings, etc. —
see vector_store.py/core/embeddings.py for the same design choice). The
MCP JSON shape below was captured once from a real SDK emit call and is
stable; hand-constructing it is a few lines, not worth the dependency.

Reads (get_table_context / get_graph_node_context) enrich the NL->SQL and
NL->Cypher schema context — see catalog.build_schema_context() and
graph_store.build_schema_context(). Writes (register_lineage) are used by
gold.py to register materialized Gold tables with provenance.

DataHub is optional everywhere it's used here, not a hard dependency: reads
return {} and writes silently no-op on any failure — a DataHub outage must
never break NL->SQL, NL->Cypher, or Gold materialization.
"""
from __future__ import annotations

import urllib.parse

import httpx
import mlflow
from mlflow.entities import SpanType

from core.config import settings

_TIMEOUT = 3.0


def _make_urn(platform: str, name: str) -> str:
    return f"urn:li:dataset:(urn:li:dataPlatform:{platform},{name},PROD)"


def _fetch_context(urn: str) -> dict:
    """
    Returns:
      {"description": str | None,
       "deprecated": bool, "deprecation_note": str | None,
       "fields": {field_name: {"description": str | None,
                                "tags": [str, ...],           # e.g. ["PII.Direct", "Consumer.Modeler"]
                                "glossary_terms": [str, ...]  # e.g. ["Revenue"]
                               }}}
    Returns {} (not an error) if DataHub is unreachable or the entity isn't
    cataloged there — callers should treat this as "no enrichment available".
    Tag/term URNs (urn:li:tag:X, urn:li:glossaryTerm:X) are stripped down to
    the bare name X for callers.
    """
    if not settings.datahub_gms_url:
        return {}

    encoded = urllib.parse.quote(urn, safe="")
    try:
        resp = httpx.get(f"{settings.datahub_gms_url}/entitiesV2/{encoded}", timeout=_TIMEOUT)
        if resp.status_code != 200:
            return {}
        aspects = resp.json().get("aspects", {})
    except Exception:
        return {}

    ds_props = aspects.get("datasetProperties", {}).get("value", {})
    description = ds_props.get("description")
    custom_properties = ds_props.get("customProperties", {})

    dep = aspects.get("deprecation", {}).get("value", {})
    deprecated = bool(dep.get("deprecated", False))
    deprecation_note = dep.get("note") or None

    field_infos = (
        aspects.get("editableSchemaMetadata", {})
        .get("value", {})
        .get("editableSchemaFieldInfo", [])
    )
    fields = {}
    for f in field_infos:
        tags = [t["tag"].rsplit(":", 1)[-1] for t in f.get("globalTags", {}).get("tags", [])]
        terms = [t["urn"].rsplit(":", 1)[-1] for t in f.get("glossaryTerms", {}).get("terms", [])]
        if f.get("description") or tags or terms:
            fields[f["fieldPath"]] = {
                "description": f.get("description"),
                "tags": tags,
                "glossary_terms": terms,
            }

    return {
        "description": description,
        "deprecated": deprecated,
        "deprecation_note": deprecation_note,
        "custom_properties": custom_properties,
        "fields": fields,
    }


@mlflow.trace(name="datahub.get_table_context", span_type=SpanType.RETRIEVER)
def get_table_context(catalog: str, schema: str, table: str, platform: str = "duckdb") -> dict:
    """
    {"description": str | None, "deprecated": bool, "deprecation_note": str | None,
     "columns": {col_name: {"description", "tags", "glossary_terms"}}} for a SQL table.
    """
    urn = _make_urn(platform, f"{catalog}.{schema}.{table}")
    result = _fetch_context(urn)
    if not result:
        return {}
    return {
        "description": result["description"],
        "deprecated": result["deprecated"],
        "deprecation_note": result["deprecation_note"],
        "columns": result["fields"],
    }


@mlflow.trace(name="datahub.get_graph_node_context", span_type=SpanType.RETRIEVER)
def get_graph_node_context(label: str) -> dict:
    """{"description": str | None, "properties": {prop_name: {...}}} for a graph node label."""
    urn = _make_urn("neo4j", label)
    result = _fetch_context(urn)
    if not result:
        return {}
    return {"description": result["description"], "properties": result["fields"]}


# Roles whose tag means "business shouldn't see this raw" — see the taxonomy
# note in redaction.py. Kept here (not hardcoded per-caller) so the mapping
# from catalog tag -> access decision lives in one place.
_MASK_FROM_BUSINESS_TAGS = {"Consumer.Modeler", "Consumer.Engineer"}


@mlflow.trace(name="datahub.get_masked_columns", span_type=SpanType.RETRIEVER)
def get_masked_columns(catalog: str, schema: str, table: str, platform: str = "duckdb") -> set[str]:
    """
    Column names this table's DataHub tags say the business role shouldn't
    see raw (tagged Consumer.Modeler or Consumer.Engineer — see the tagging
    taxonomy applied to iceberg.ecommerce.{customers,products,orders}).
    Empty set if DataHub is unreachable or nothing is tagged — same
    optional/best-effort contract as get_table_context.
    """
    ctx = get_table_context(catalog, schema, table, platform)
    if not ctx:
        return set()
    return {
        col
        for col, info in ctx["columns"].items()
        if _MASK_FROM_BUSINESS_TAGS & set(info.get("tags", []))
    }


def _emit_mcp(entity_urn: str, aspect_name: str, aspect_value: dict) -> None:
    """POSTs one MetadataChangeProposal — the shape ingestProposal expects,
    captured from a real acryl-datahub emit call. Best-effort: swallows any
    failure, since every caller treats DataHub registration as optional."""
    import json

    if not settings.datahub_gms_url:
        return
    try:
        httpx.post(
            f"{settings.datahub_gms_url}/aspects?action=ingestProposal",
            json={
                "proposal": {
                    "entityType": "dataset",
                    "entityUrn": entity_urn,
                    "changeType": "UPSERT",
                    "aspectName": aspect_name,
                    "aspect": {
                        "value": json.dumps(aspect_value),
                        "contentType": "application/json",
                    },
                },
                "async": "true",
            },
            timeout=_TIMEOUT,
        )
    except Exception:
        pass


def register_lineage(
    platform: str,
    name: str,
    description: str,
    upstream_tables: list[str],
    upstream_platform: str = "duckdb",
    custom_properties: dict | None = None,
) -> None:
    """Registers a dataset's description + upstream lineage in one call —
    used by gold.py after materializing a Gold table. Best-effort/optional,
    same as everywhere else DataHub is touched in this gateway."""
    urn = _make_urn(platform, name)
    _emit_mcp(
        urn,
        "datasetProperties",
        {
            "customProperties": custom_properties or {},
            "name": name.split(".")[-1],
            "description": description,
            "tags": [],
        },
    )
    if upstream_tables:
        _emit_mcp(
            urn,
            "upstreamLineage",
            {
                "upstreams": [
                    {
                        "auditStamp": {"time": 0, "actor": "urn:li:corpuser:unknown"},
                        "dataset": _make_urn(upstream_platform, src),
                        "type": "TRANSFORMED",
                    }
                    for src in upstream_tables
                ]
            },
        )


def _graph_urn(schema: str) -> str:
    return _make_urn("neo4j", f"{schema}_graph")


def record_graph_build(schema: str, node_count: int, relationship_count: int) -> None:
    """
    Records when a schema's knowledge graph was last built/materialized.
    graph_stats() has no notion of staleness on its own — nothing
    invalidates or refreshes it when new data lands upstream in Iceberg, so
    without this a caller has no way to tell a fresh graph from one built
    a week ago. Called at the end of graph_store.build_graph() and
    graph_inference.materialize_inferred_graph(). Best-effort/optional, same
    as everywhere else DataHub is touched in this gateway.
    """
    import datetime

    _emit_mcp(
        _graph_urn(schema),
        "datasetProperties",
        {
            "name": f"{schema}_graph",
            "description": f"Neo4j knowledge graph built from iceberg.{schema}",
            "customProperties": {
                "last_built": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "node_count": str(node_count),
                "relationship_count": str(relationship_count),
            },
            "tags": [],
        },
    )


@mlflow.trace(name="datahub.get_graph_build_info", span_type=SpanType.RETRIEVER)
def get_graph_build_info(schema: str) -> dict:
    """{"last_built": iso str, "node_count": int, "relationship_count": int} for
    a schema's graph, or {} if it's never been recorded / DataHub is unreachable."""
    result = _fetch_context(_graph_urn(schema))
    if not result:
        return {}
    props = result["custom_properties"]
    if "last_built" not in props:
        return {}
    return {
        "last_built": props["last_built"],
        "node_count": int(props.get("node_count", 0)),
        "relationship_count": int(props.get("relationship_count", 0)),
    }
