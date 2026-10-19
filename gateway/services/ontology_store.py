"""
Versioned graph ontologies, stored in Neo4j next to the graph they describe.

Why: LLM ontology inference isn't reproducible run to run (qwen3:4b was seen
varying property lists between otherwise identical runs), so "preview, then
materialize" used to materialize a *different* model from the one reviewed —
materialize re-ran inference. Now /graph/infer saves what it showed you as a
numbered draft, /graph/infer/materialize commits exactly that version and marks
it active, and /graph/nl grounds Cypher generation on the active ontology.

Stored as (:_Ontology {schema, version, status, model_json, created_at}).
Labels starting with "_" are internal and hidden from the schema context the
LLM sees (graph_store.build_schema_context).
"""
from __future__ import annotations

import datetime as dt
import json

from services import graph_store


def save_draft(schema: str, model: dict) -> int:
    with graph_store._get_driver().session() as session:
        rec = session.run(
            """
            OPTIONAL MATCH (o:_Ontology {schema: $schema})
            WITH coalesce(max(o.version), 0) + 1 AS next
            CREATE (n:_Ontology {schema: $schema, version: next, status: 'draft',
                                 model_json: $model_json, created_at: $now})
            RETURN next AS version
            """,
            schema=schema,
            model_json=json.dumps(model, sort_keys=True),
            now=dt.datetime.now(dt.timezone.utc).isoformat(),
        ).single()
    return rec["version"]


def get(schema: str, version: int) -> dict | None:
    rows = graph_store.run_cypher(
        "MATCH (o:_Ontology {schema: $schema, version: $version}) "
        "RETURN o.model_json AS model_json, o.status AS status, o.created_at AS created_at",
        {"schema": schema, "version": version},
    )
    return _decode(rows[0], version) if rows else None


def get_active(schema: str) -> dict | None:
    rows = graph_store.run_cypher(
        "MATCH (o:_Ontology {schema: $schema, status: 'active'}) "
        "RETURN o.model_json AS model_json, o.status AS status, o.created_at AS created_at, "
        "o.version AS version ORDER BY o.version DESC LIMIT 1",
        {"schema": schema},
    )
    return _decode(rows[0], rows[0]["version"]) if rows else None


def list_versions(schema: str) -> list[dict]:
    return graph_store.run_cypher(
        "MATCH (o:_Ontology {schema: $schema}) "
        "RETURN o.version AS version, o.status AS status, o.created_at AS created_at "
        "ORDER BY o.version DESC",
        {"schema": schema},
    )


def activate(schema: str, version: int) -> None:
    with graph_store._get_driver().session() as session:
        session.run(
            "MATCH (o:_Ontology {schema: $schema}) WHERE o.status = 'active' SET o.status = 'superseded'",
            schema=schema,
        )
        session.run(
            "MATCH (o:_Ontology {schema: $schema, version: $version}) SET o.status = 'active'",
            schema=schema, version=version,
        )


def _decode(row: dict, version: int) -> dict:
    return {
        "version": version,
        "status": row["status"],
        "created_at": row["created_at"],
        "model": json.loads(row["model_json"]),
    }
