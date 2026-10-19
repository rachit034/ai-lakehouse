from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from core import redaction
from core.auth import ALL_ROLES, require_role
from fastapi.concurrency import run_in_threadpool

from services import graph_inference, graph_qa, graph_store, ontology_store, write_log

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


class GraphNLRequest(BaseModel):
    question: str
    # Restrict to one source schema's nodes (and ground on its active ontology).
    schema: str | None = None
    # Also return a natural-language answer written from the result rows.
    answer: bool = True


class ExampleRequest(BaseModel):
    question: str
    cypher: str


class InferRequest(BaseModel):
    schema: str
    limit: int = 100_000
    # The draft version returned by GET /graph/infer. Omit to infer afresh
    # (not recommended — inference isn't reproducible, so you'd commit a
    # model nobody reviewed).
    ontology_version: int | None = None


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
        records = redaction.redact_records(records, cypher=req.cypher)
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
        info = graph_store.get_build_freshness(schema)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    # DataHub is optional — fall back to the gateway's own build record, then
    # compare against the last write to the source schema through the gateway.
    last_built = info.get("last_built") or write_log.last_graph_build(schema)
    last_source_write = write_log.last_write(schema)
    return {
        **info,
        "last_built": last_built,
        "last_source_write": last_source_write,
        "stale": write_log.is_stale(last_built, last_source_write),
    }


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
        return await graph_inference.infer_and_save_draft(schema)
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
        return await graph_inference.materialize_inferred_graph(req.schema, req.limit, req.ontology_version)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Materialization failed: {e}")


@router.get("/ontology", dependencies=_any_role)
def ontology(schema: str, version: int | None = None):
    """The active ontology for `schema` (or a specific version) plus the version
    history — what /graph/nl grounds its Cypher generation on."""
    try:
        current = ontology_store.get(schema, version) if version else ontology_store.get_active(schema)
        return {"ontology": current, "versions": ontology_store.list_versions(schema)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/nl", dependencies=_any_role)
async def graph_nl(req: GraphNLRequest, request: Request):
    """
    Ask the knowledge graph a question in plain English, answered entirely by
    the local LLM: schema- and ontology-grounded Cypher generation, EXPLAIN
    validation with a repair loop, bounded read-only execution, and an
    optional written answer. See services/graph_qa.py for the pipeline.

    business gets results redacted *before* the answer is written, so the
    answer can't restate what redaction hid.
    """
    redact = None
    if request.state.role == "business":
        redact = lambda records, cypher: redaction.redact_records(records, cypher=cypher)  # noqa: E731
    try:
        return await graph_qa.answer(req.question, schema=req.schema, synthesize=req.answer, redact=redact)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Graph Q&A failed [{type(e).__name__}]: {e}")


@router.post("/nl/examples", dependencies=_modeler_only)
async def add_example(req: ExampleRequest):
    """
    Add a verified question→Cypher pair to the few-shot library /graph/nl
    retrieves from. data_modeler only: an example teaches the model what
    "correct" means on this graph, which is a modeling decision.
    """
    try:
        await run_in_threadpool(graph_qa.add_verified_example, req.question, req.cypher)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok"}
