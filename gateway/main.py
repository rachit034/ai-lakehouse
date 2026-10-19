from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.auth import ALL_ROLES, require_role
from core.config import settings
from core.tracing import TracingMiddleware, configure_tracing
from routers import graph, ingest, medallion, query


@asynccontextmanager
async def lifespan(app: FastAPI):
    # llm.py owns its own Redis connection (via RedisVL's SemanticCache), so
    # there's no app-level Redis handle to manage here anymore.
    configure_tracing()
    yield


app = FastAPI(
    title="AI Lakehouse Gateway",
    description=(
        "Local prototype — DuckDB (embedded, Athena substitute) + Iceberg "
        "(Gravitino catalog) + LanceDB + Ollama.\n\n"
        "**Endpoints**\n"
        "- `/query/sql`    — direct SQL on DuckDB/Iceberg\n"
        "- `/query/nl`     — natural language → SQL via Ollama\n"
        "- `/query/vector` — semantic vector search via LanceDB\n"
        "- `/ingest/csv`   — load CSV into an Iceberg table via DuckDB\n"
        "- `/ingest/embed` — embed an Iceberg table and store in LanceDB\n"
        "- `/graph/build`  — build the knowledge graph from Iceberg tables into Neo4j\n"
        "- `/graph/query`  — run Cypher against the knowledge graph\n"
        "- `/graph/nl`     — ask the knowledge graph in plain English (local LLM)\n"
        "- `/graph/infer`  — infer a versioned ontology draft from a schema (local LLM)\n"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Explicit origins, not "*": the gateway accepts API keys, and a wildcard
# lets any web page a user visits call it from their browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type", "X-Agent-Name", "X-Run-Id"],
)
app.add_middleware(TracingMiddleware)

# query.router is the shared read surface — every role can reach it (see
# core/auth.py's docstring for why). ingest.router is data_engineer-only.
# graph.router and medallion.router mix roles per-route (e.g. /graph/query is
# open to everyone, /graph/build is data_modeler-only), so their role
# dependencies live on the individual routes instead of here.
app.include_router(
    query.router, prefix="/query", tags=["Query"], dependencies=[Depends(require_role(*ALL_ROLES))]
)
app.include_router(
    ingest.router, prefix="/ingest", tags=["Ingest"], dependencies=[Depends(require_role("data_engineer"))]
)
app.include_router(graph.router, prefix="/graph", tags=["Graph"])
app.include_router(medallion.router, prefix="/medallion", tags=["Medallion"])


@app.get("/health", tags=["Ops"])
async def health():
    return {"status": "ok"}


@app.get("/ready", tags=["Ops"])
async def ready(request_state=None):
    """Check that all downstream services are reachable."""
    import httpx

    checks = {}

    # Gravitino (catalog) + DuckDB (embedded — check the actual ATTACHed
    # connection rather than an HTTP endpoint, since there's no separate
    # query-engine service to ping)
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{settings.gravitino_uri}/api/metalakes")
            checks["gravitino"] = "ok" if r.status_code == 200 else "degraded"
    except Exception:
        checks["gravitino"] = "unreachable"

    try:
        from services import sql_engine

        sql_engine.catalog_query("SELECT 1")
        checks["duckdb"] = "ok"
    except Exception:
        checks["duckdb"] = "unreachable"

    # Ollama
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            checks["ollama"] = "ok" if r.status_code == 200 else "degraded"
    except Exception:
        checks["ollama"] = "unreachable"

    # Redis
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(settings.redis_url)
        checks["redis"] = "ok" if await client.ping() else "degraded"
        await client.aclose()
    except Exception:
        checks["redis"] = "unreachable"

    # Neo4j
    try:
        neo4j_host = settings.neo4j_uri.split("://")[1].split(":")[0]
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"http://{neo4j_host}:7474")
            checks["neo4j"] = "ok" if r.status_code == 200 else "degraded"
    except Exception:
        checks["neo4j"] = "unreachable"

    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "ok" if all_ok else "degraded", "services": checks}
