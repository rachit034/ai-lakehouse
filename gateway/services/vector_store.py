"""
Vector storage and search via LanceDB on MinIO.
Uses fastembed (ONNX, no PyTorch) for embeddings — lightweight for local dev.
In production point LANCE_URI at s3://your-bucket/lance/ with real AWS creds.

Embeddings are computed directly with fastembed rather than through lancedb's
embedding-function registry: no version of lancedb (checked 0.13 through 0.34)
actually ships a "fastembed" backend in that registry, so relying on it always
raised KeyError. Vectors are precomputed and stored on the model explicitly.
"""
from __future__ import annotations

import os

import lancedb
import mlflow
from lancedb.pydantic import LanceModel, Vector
from mlflow.entities import SpanType

from core.config import settings
from core.embeddings import embed_many

# lancedb 0.13's connect() doesn't accept a storage_options kwarg for non-cloud
# URIs (see LanceDBConnection.__init__) — S3-compatible endpoints like MinIO
# must instead be configured via env vars. Two separate S3 clients read these:
# the Rust object_store crate (writes, table_names) reads AWS_ENDPOINT_URL;
# lancedb.util.fs_from_uri's pyarrow S3FileSystem (open_table, i.e. reads)
# reads plain AWS_ENDPOINT — both must be set or reads 403 against MinIO.
os.environ.setdefault("AWS_ENDPOINT_URL", settings.minio_endpoint)
os.environ.setdefault("AWS_ENDPOINT", settings.minio_endpoint)
os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.minio_access_key)
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.minio_secret_key)
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ALLOW_HTTP", "true")

@mlflow.trace(name="fastembed.embed", span_type=SpanType.EMBEDDING)
def _embed(texts: list[str]) -> list[list[float]]:
    return embed_many(texts)


def _db():
    return lancedb.connect(settings.lance_uri)


def _make_model(ndims: int):
    """Dynamically create a LanceModel with the right vector dimension."""

    class TextRecord(LanceModel):
        id: str
        text: str
        vector: Vector(ndims)  # type: ignore[valid-type]
        metadata: str = ""  # JSON string for extra fields

    return TextRecord


@mlflow.trace(name="lancedb.ingest", span_type=SpanType.TOOL)
def ingest(dataset: str, records: list[dict], text_field: str) -> int:
    """
    Embed records[text_field] and write to LanceDB table.
    Creates the table if it doesn't exist, appends otherwise.
    Returns number of records written.
    """
    import json

    texts = [str(rec.get(text_field, "")) for rec in records]
    vectors = _embed(texts)
    TextRecord = _make_model(len(vectors[0]))

    rows = []
    for i, (rec, text, vector) in enumerate(zip(records, texts, vectors)):
        extra = {k: v for k, v in rec.items() if k != text_field}
        rows.append(
            TextRecord(
                id=str(rec.get("id", i)),
                text=text,
                vector=vector,
                metadata=json.dumps(extra),
            )
        )

    db = _db()
    if dataset in db.table_names():
        table = db.open_table(dataset)
        table.add(rows)
    else:
        db.create_table(dataset, data=rows, schema=TextRecord)

    return len(rows)


@mlflow.trace(name="lancedb.search", span_type=SpanType.RETRIEVER)
def search(dataset: str, query: str, top_k: int = 10, filter_sql: str | None = None) -> list[dict]:
    """
    Semantic search over a LanceDB dataset.
    Returns list of {id, text, metadata, _distance}.
    """
    import json

    db = _db()
    if dataset not in db.table_names():
        raise ValueError(f"Dataset '{dataset}' not found. Ingest data first.")

    table = db.open_table(dataset)
    query_vector = _embed([query])[0]
    q = table.search(query_vector).limit(top_k)
    if filter_sql:
        q = q.where(filter_sql)

    results = q.to_list()
    out = []
    for r in results:
        extra = json.loads(r.get("metadata", "{}"))
        out.append(
            {
                "id": r["id"],
                "text": r["text"],
                "_distance": r.get("_distance"),
                **extra,
            }
        )
    return out


def list_datasets() -> list[str]:
    return _db().table_names()
