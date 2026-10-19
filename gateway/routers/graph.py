from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from core import redaction
from core.auth import ALL_ROLES, require_role
from services import graph_inference, graph_store

router = APIRouter()

# /build, /infer, /infer/materialize commit or preview a *model* — that's the
# data modeler's call, not something ingestion or a business question should
# be able to trigger. /query, /stats, /schema are the read surface every role
# (including modeler/engineer verifying their own work) can reach.
_modeler_only = [Depends(require_role("data_modeler"))]
_any_role = [Depends(require_role(*ALL_ROLES))]


class BuildRequest(BaseModel):
    schema: str = "ecommerce"
    limit: int = 100_000


class CypherRequest(BaseModel):
    cypher: str
    params: dict[str, Any] | None = None


class InferRequest(BaseModel):
    schema: str
    limit: int = 100_000


@router.post("/build", dependencies=_modeler_only)
def build(req: BuildRequest):
    """Build/refresh the knowledge graph from the ecommerce Iceberg tables. data_modeler only."""
    try:
        stats = graph_store.build_graph(schema=req.schema, limit=req.limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph build failed: {e}")
    return {"status": "ok", **stats}


@router.post("/query", dependencies=_any_role)
def query(req: CypherRequest, request: Request):
    """Run a Cypher query against the knowledge graph.

    data_engineer/data_modeler get raw values back (validating their own
    ingested/modeled data is a legitimate need); business always gets PII
    redacted — see core/redaction.py.
    """
    try:
        records = graph_store.run_cypher(req.cypher, req.params)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    if request.state.role == "business":
        records = redaction.redact_records(records)
    return {"records": records, "count": len(records)}


@router.get("/stats", dependencies=_any_role)
def stats():
    """Node/relationship counts in the knowledge graph."""
    try:
        return graph_store.graph_stats()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/freshness", dependencies=_any_role)
def freshness(schema: str):
    """
    When this schema's graph was last built/materialized, and its node/
    relationship counts at that time — graph_stats() has no notion of
    staleness on its own, so check this before trusting the graph for
    anything time-sensitive. Empty dict if it's never been built (or
    DataHub is unreachable) — that itself is a signal worth surfacing to
    the caller, not an error.
    """
    try:
        return graph_store.get_build_freshness(schema)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/schema", dependencies=_any_role)
def schema():
    """
    Live graph schema (labels, properties, relationship direction verified
    against real data) enriched with DataHub descriptions where available.
    For NL->Cypher generation — see graph_store.build_schema_context().
    """
    try:
        return {"schema_context": graph_store.build_schema_context()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/infer", dependencies=_modeler_only)
async def infer(schema: str):
    """
    Read-only preview: an LLM reads the schema's real structure AND a sample
    of real rows, and reasons about what the tables/columns mean (see
    graph_inference.py's docstring — this catches value-based references,
    like a column holding a product name rather than an id, that naming
    conventions alone can't). Does NOT touch Neo4j. Review this before
    calling POST /graph/infer/materialize. data_modeler only — seeing raw
    sample data to design the model is a model-design decision even though
    nothing is written yet.
    """
    try:
        return await graph_inference.infer_graph_model(schema)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")


@router.post("/infer/materialize", dependencies=_modeler_only)
async def infer_materialize(req: InferRequest):
    """
    Infers the graph model from `req.schema` (same LLM-driven inference as
    GET /infer) and MERGEs it into Neo4j — idempotent, generic (see
    graph_inference.py), not hand-coded per dataset. data_modeler only.
    """
    try:
        return await graph_inference.materialize_inferred_graph(req.schema, req.limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Materialization failed: {e}")
