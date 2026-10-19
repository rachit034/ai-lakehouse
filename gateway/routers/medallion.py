from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from core.auth import ALL_ROLES, require_role
from services import entity_resolution, gold

router = APIRouter()

# Silver preview/build decide what counts as "the same entity" — a modeling
# call, same reasoning as graph /build and /infer being modeler-only. /gold
# is open to every role by default (it's a query, like /query/nl); whether a
# given call may *persist* a new Gold table is checked inside the handler,
# since that depends on the request body's materialize flag, not the route.
_modeler_only = [Depends(require_role("data_modeler"))]
_any_role = [Depends(require_role(*ALL_ROLES))]


class SilverRequest(BaseModel):
    schema: str = "bronze"
    table: str
    target_schema: str = "silver"
    threshold: float = 0.85


class GoldRequest(BaseModel):
    question: str
    source_schema: str = "silver"
    materialize: bool = False
    table_name: str | None = None


@router.post("/silver/preview", dependencies=_modeler_only)
def silver_preview(req: SilverRequest):
    """
    Read-only: runs entity resolution (real SIMILAR_TO edges written to
    Neo4j, connected-component grouping) and returns the groups it found —
    does not write anything to DuckDB/Iceberg. Review before /silver/build.
    data_modeler only — deciding what counts as a duplicate is a modeling call.
    """
    try:
        return entity_resolution.find_duplicate_groups(req.schema, req.table, req.threshold)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Resolution failed: {e}")


@router.post("/silver/build", dependencies=_modeler_only)
def silver_build(req: SilverRequest):
    """Runs entity resolution and materializes the canonical Silver table + id-map table.
    data_modeler only."""
    try:
        return entity_resolution.build_silver_table(
            req.schema, req.table, req.target_schema, req.threshold
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Silver build failed: {e}")


@router.post("/gold", dependencies=_any_role)
async def gold_query(req: GoldRequest, request: Request):
    """
    NL -> SQL against `source_schema` (Silver by default). materialize=false
    (default) executes and returns rows, like /query/nl — any role.
    materialize=true persists as iceberg.gold.{table_name} and registers it
    in the catalog with lineage — that's a modeling decision (a new
    permanent, catalogued table other people will query), so it requires
    data_modeler even though the query itself is open to everyone.
    table_name is caller-supplied and validated, never taken from the NL
    question or model output.
    """
    if req.materialize and request.state.role != "data_modeler":
        raise HTTPException(
            status_code=403,
            detail="Persisting a Gold table (materialize=true) requires the data_modeler role.",
        )
    try:
        return await gold.run_gold_query(
            req.question, req.source_schema, req.materialize, req.table_name
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gold query failed: {e}")
