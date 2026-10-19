"""
Last-write timestamps per Iceberg schema, so graph freshness can say "stale"
instead of only "last built at X".

The knowledge graph is derived from Iceberg — every gateway write path
(/ingest/csv, Silver build, Gold materialize) calls record_write(schema), and
/graph/freshness compares that against the schema's last graph build. Writes
that bypass the gateway (data_generator/load_to_iceberg.py) aren't seen; the
proposal doc covers moving this to Iceberg snapshot metadata.

Best-effort: Redis being down must never fail a write, only make freshness
report stale=None (unknown).
"""
from __future__ import annotations

import datetime as dt

import redis

from core.config import settings

_KEY = "lakehouse:last_write:{schema}"
# Also kept here (not only in DataHub, which is optional) so freshness works
# without a catalog running.
_BUILD_KEY = "lakehouse:last_graph_build:{schema}"
_client: redis.Redis | None = None


def _redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.Redis.from_url(settings.redis_url, socket_timeout=2)
    return _client


def record_write(schema: str) -> None:
    try:
        _redis().set(_KEY.format(schema=schema), dt.datetime.now(dt.timezone.utc).isoformat())
    except Exception:
        pass


def last_write(schema: str) -> str | None:
    try:
        value = _redis().get(_KEY.format(schema=schema))
    except Exception:
        return None
    return value.decode() if value else None


def record_graph_build(schema: str) -> None:
    try:
        _redis().set(_BUILD_KEY.format(schema=schema), dt.datetime.now(dt.timezone.utc).isoformat())
    except Exception:
        pass


def last_graph_build(schema: str) -> str | None:
    try:
        value = _redis().get(_BUILD_KEY.format(schema=schema))
    except Exception:
        return None
    return value.decode() if value else None


def is_stale(last_built: str | None, last_source_write: str | None) -> bool | None:
    """True when the source schema changed after the graph was built; None if unknown."""
    if not last_built:
        return None
    if not last_source_write:
        return False
    try:
        built = dt.datetime.fromisoformat(last_built.replace("Z", "+00:00"))
        wrote = dt.datetime.fromisoformat(last_source_write)
    except ValueError:
        return None
    if built.tzinfo is None:
        built = built.replace(tzinfo=dt.timezone.utc)
    return wrote > built
