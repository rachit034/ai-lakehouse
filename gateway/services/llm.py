"""
NL → SQL via Ollama.
In production point OLLAMA_BASE_URL at an internal LLM endpoint or swap
httpx calls for boto3 Bedrock invoke_model.

Caching is semantic (RedisVL SemanticCache), not exact-match: "find me the
customer with maximum orders" and "customer with max orders" hash to
different strings but embed to nearly the same vector, so the old
hash-keyed Redis cache always missed on rephrasing. Requires the Redis
Stack search module (redis/redis-stack-server), not plain redis:7-alpine.
"""
from __future__ import annotations

import hashlib
import re

import httpx
import mlflow
import sqlglot
from mlflow.entities import SpanType
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.query.filter import Tag
from redisvl.utils.vectorize.text.custom import CustomVectorizer

from core.config import settings
from core.embeddings import embed_one

_NL_TO_SQL_PROMPT = """You are a SQL expert for DuckDB.
Generate a single SQL SELECT query based on the question below.

Available tables:
{schema}

Rules:
- Return ONLY the SQL query — no explanation, no markdown fences
- Use fully qualified table names: catalog.schema.table
- Use DuckDB SQL syntax (ANSI SQL, no backtick quoting — use double quotes if needed)
- Do not use LIMIT unless the question asks for top-N results
- If the question names a metric listed under "Business metrics" below (or one of
  its aliases), use that metric's exact SQL expression rather than deriving your own

Question: {question}

SQL:"""

_cache: SemanticCache | None = None


def _get_cache() -> SemanticCache:
    """Lazy singleton — constructing SemanticCache/CustomVectorizer eagerly
    loads the fastembed ONNX model, same lazy pattern as core/embeddings.py."""
    global _cache
    if _cache is None:
        _cache = SemanticCache(
            name="nl_sql_cache",
            redis_url=settings.redis_url,
            vectorizer=CustomVectorizer(embed=embed_one),
            distance_threshold=settings.nl_cache_distance_threshold,
            ttl=settings.nl_cache_ttl,
            # schema_hash-scoped so a cached translation never leaks across
            # schemas — the same question can mean a different query, or no
            # longer be valid SQL at all, if the underlying tables differ.
            filterable_fields=[{"name": "schema_hash", "type": "tag"}],
        )
    return _cache


def _parse_sql(raw: str) -> str:
    """Strip markdown fences and whitespace from LLM output."""
    raw = raw.strip()
    # Remove ```sql ... ``` or ``` ... ```
    raw = re.sub(r"^```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _validate_sql(sql: str) -> str:
    """Parse with sqlglot to catch obvious syntax errors. Returns sql unchanged if valid."""
    try:
        sqlglot.parse_one(sql, dialect="duckdb")
    except Exception as e:
        raise ValueError(f"Generated SQL failed validation: {e}\nSQL: {sql}") from e
    return sql


@mlflow.trace(name="ollama.generate", span_type=SpanType.LLM)
async def _call_ollama(prompt: str, timeout: float = 120) -> str:
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                # no num_predict cap — thinking models (qwen3) need budget for CoT
                "options": {"temperature": 0},
            },
        )
        resp.raise_for_status()
        # Ollama returns thinking in a separate field; response = final answer only
        return resp.json()["response"]


@mlflow.trace(name="llm.nl_to_sql", span_type=SpanType.CHAIN)
async def nl_to_sql(question: str, schema_context: str) -> dict:
    """
    Translate a natural language question to SQL.
    Returns {sql, cached: bool}.
    """
    schema_hash = hashlib.md5(schema_context.encode()).hexdigest()[:8]
    cache = _get_cache()

    cached = await cache.acheck(prompt=question, filter_expression=(Tag("schema_hash") == schema_hash))
    if cached:
        mlflow.update_current_trace(tags={"cache": "hit"})
        return {"sql": cached[0]["response"], "cached": True}

    mlflow.update_current_trace(tags={"cache": "miss"})
    prompt = _NL_TO_SQL_PROMPT.format(schema=schema_context, question=question)
    raw_sql = await _call_ollama(prompt)
    sql = _validate_sql(_parse_sql(raw_sql))

    await cache.astore(prompt=question, response=sql, filters={"schema_hash": schema_hash})

    return {"sql": sql, "cached": False}
