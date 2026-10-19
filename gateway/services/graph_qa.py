"""
Local-LLM question answering over the knowledge graph (/graph/nl).

The goal this serves: a small local model (Ollama) answering complex,
multi-hop questions from the graph without a frontier API. Small models fail
at Cypher in predictable ways — wrong relationship direction, invented labels
or properties, ignoring the schema scope, writing prose around the query — so
instead of hoping a bigger model fixes that, every one of those failure modes
gets a mechanical guard:

  1. Grounding     live graph schema (direction verified against data, see
                   graph_store.build_schema_context) + the active ontology's
                   business descriptions + the most similar verified
                   question→Cypher examples (few-shot, retrieved by embedding)
  2. Constrained   Ollama structured output: {"reasoning", "cypher"} JSON,
     output        never free text to scrape a query out of
  3. Validation    read-only clause check, no internal labels, schema scope
                   present, then Neo4j EXPLAIN — which plans without executing
                   and reports unknown labels/relationships/properties
  4. Repair loop   validation problems (or an empty result) are fed back for
                   another attempt, up to settings.graph_nl_max_attempts
  5. Bounded       a LIMIT is enforced, and the query runs in a READ transaction
                   (Neo4j itself rejects writes — see graph_store.run_cypher)
  6. Answer        optional natural-language answer synthesized only from the
                   (already redacted) rows, so it can't leak what redaction hid

Examples come from gateway/graph_examples.json plus ones a data_modeler
verified via POST /graph/nl/examples — the library grows as people confirm
good answers, which is how a small model gets better on *this* graph without
fine-tuning. Measure it with scripts/eval_graph_nl.py.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Callable

import mlflow
from mlflow.entities import SpanType

from core.config import settings
from services import graph_store, llm, ontology_store

_EXAMPLES_FILE = Path(__file__).resolve().parent.parent / "graph_examples.json"
_FORBIDDEN_RE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV|CALL)\b", re.IGNORECASE
)
_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+\s*$", re.IGNORECASE)
_PROBLEM_NOTIFICATIONS = ("UnknownLabel", "UnknownRelationship", "UnknownPropertyKey", "CartesianProduct")

CYPHER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "cypher": {"type": "string"},
    },
    "required": ["reasoning", "cypher"],
}
ANSWER_JSON_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

_PROMPT = """You translate questions into a single read-only Cypher query for Neo4j.

Graph schema (labels, properties, and relationship DIRECTIONS verified against the live data —
follow the arrows exactly as shown):
{schema}
{ontology}
Verified examples of correct queries on this graph:
{examples}

Rules:
- Use ONLY labels, relationship types and properties that appear in the schema above.
- Follow relationship directions exactly as written in the schema.
- Read-only: MATCH / OPTIONAL MATCH / WITH / WHERE / RETURN / ORDER BY / LIMIT / UNWIND only.
  Never CREATE, MERGE, SET, DELETE, REMOVE, or CALL.
- Put literal values directly in the query (no $parameters).
- Alias every returned column with AS and a readable name.
{scope_rule}- Think step by step in "reasoning" (which nodes, which path, which aggregation),
  then put only the query in "cypher".

Question: {question}
"""

_FIX_PROMPT = """{base}

Your previous query was:
{cypher}

It has these problems:
{problems}

Return a corrected query that fixes every problem listed.
"""

_ANSWER_PROMPT = """Answer the question using ONLY the query results below. If the results are
empty or don't answer it, say so plainly. Be concise and specific; quote the numbers.

Question: {question}

Cypher that produced the results:
{cypher}

Results ({count} rows{shown}):
{rows}
"""


# ── Examples ──────────────────────────────────────────────────────────────────

_example_vectors: list[tuple[dict, list[float]]] | None = None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _load_examples() -> list[dict]:
    examples = json.loads(_EXAMPLES_FILE.read_text()) if _EXAMPLES_FILE.exists() else []
    try:
        examples += graph_store.run_cypher(
            "MATCH (e:_CypherExample) RETURN e.question AS question, e.cypher AS cypher"
        )
    except Exception:
        pass  # Neo4j down: file examples still work
    return examples


def _example_index() -> list[tuple[dict, list[float]]]:
    global _example_vectors
    if _example_vectors is None:
        from core.embeddings import embed_many

        examples = _load_examples()
        vectors = embed_many([e["question"] for e in examples]) if examples else []
        _example_vectors = list(zip(examples, vectors))
    return _example_vectors


def retrieve_examples(question: str, k: int = 4) -> list[dict]:
    index = _example_index()
    if not index:
        return []
    from core.embeddings import embed_one

    qv = embed_one(question)
    ranked = sorted(index, key=lambda ev: _cosine(qv, ev[1]), reverse=True)
    return [e for e, _ in ranked[:k]]


def add_verified_example(question: str, cypher: str) -> None:
    """Store a data_modeler-confirmed question→Cypher pair for future few-shot use."""
    problems = validate_cypher(cypher, schema=None)
    if problems:
        raise ValueError("Example query fails validation: " + "; ".join(problems))
    with graph_store._get_driver().session() as session:
        session.run(
            "MERGE (e:_CypherExample {question: $question}) SET e.cypher = $cypher, e.created_at = $now",
            question=question, cypher=cypher, now=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
    global _example_vectors
    _example_vectors = None  # rebuild on next retrieval


# ── Validation ────────────────────────────────────────────────────────────────

def _strip_strings(cypher: str) -> str:
    """Drop quoted literals so a product named 'Set of 3' can't trip the keyword check."""
    return re.sub(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", "''", cypher)


def static_problems(cypher: str, schema: str | None) -> list[str]:
    problems = []
    code = _strip_strings(cypher)
    if not code.strip():
        return ["The query is empty."]
    if m := _FORBIDDEN_RE.search(code):
        problems.append(f"Uses {m.group(1).upper()}, which is not allowed — the query must be read-only.")
    if re.search(r":\s*`?_", code) or "_node_key" in code:
        problems.append("References an internal label or _node_key — use only schema labels/properties.")
    if "$" in code:
        problems.append("Uses $parameters — put literal values directly in the query.")
    if ";" in code.strip().rstrip(";"):
        problems.append("Contains more than one statement — return exactly one query.")
    if schema and "_schema" not in code:
        problems.append(f"Doesn't restrict to schema '{schema}' — add {{_schema: '{schema}'}} to the node patterns.")
    return problems


def validate_cypher(cypher: str, schema: str | None) -> list[str]:
    """Static checks, then Neo4j EXPLAIN (plans the query without running it)."""
    problems = static_problems(cypher, schema)
    if problems:
        return problems
    try:
        notifications = graph_store.explain_cypher(cypher)
    except Exception as e:  # syntax errors, type errors
        return [f"Neo4j rejected the query: {str(e).splitlines()[0][:300]}"]
    return [
        f"{n.get('title') or n.get('code')}: {n.get('description', '')[:200]}"
        for n in notifications
        if any(tag in (n.get("code") or "") for tag in _PROBLEM_NOTIFICATIONS)
    ]


def ensure_limit(cypher: str, limit: int) -> str:
    c = cypher.strip().rstrip(";")
    if _LIMIT_RE.search(c) or re.search(r"\bUNION\b", c, re.IGNORECASE):
        return c
    return f"{c}\nLIMIT {limit}"


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _ontology_block(schema: str | None) -> tuple[str, int | None]:
    if not schema:
        return "", None
    try:
        active = ontology_store.get_active(schema)
    except Exception:
        return "", None
    if not active:
        return "", None
    lines = ["Business meaning of the graph (from the reviewed ontology):"]
    for node in active["model"].get("nodes", []):
        if node.get("description"):
            lines.append(f"  (:{node['label']}) — {node['description']}")
    for rel in active["model"].get("relationships", []):
        if rel.get("description"):
            lines.append(f"  [:{rel['type']}] — {rel['description']}")
    block = "\n".join(lines) + "\n" if len(lines) > 1 else ""
    return block, active["version"]


@mlflow.trace(name="graph_qa.answer", span_type=SpanType.CHAIN)
async def answer(
    question: str,
    schema: str | None = None,
    synthesize: bool = True,
    redact: Callable[[list[dict], str], list[dict]] | None = None,
) -> dict:
    schema_ctx = await asyncio.to_thread(graph_store.build_schema_context)
    ontology_text, ontology_version = await asyncio.to_thread(_ontology_block, schema)
    examples = await asyncio.to_thread(retrieve_examples, question)
    examples_text = "\n".join(f"Q: {e['question']}\nCypher: {e['cypher']}\n" for e in examples) or "(none yet)"
    scope_rule = (
        f"- Restrict every node pattern to this dataset with {{_schema: '{schema}'}}.\n" if schema else ""
    )
    base = _PROMPT.format(
        schema=schema_ctx, ontology=ontology_text, examples=examples_text,
        scope_rule=scope_rule, question=question,
    )

    prompt, attempts, history = base, 0, []
    records: list[dict] = []
    cypher = ""
    while attempts < settings.graph_nl_max_attempts:
        attempts += 1
        out = await llm.generate_json(prompt, CYPHER_JSON_SCHEMA, timeout=300)
        cypher = (out.get("cypher") or "").strip()
        problems = await asyncio.to_thread(validate_cypher, cypher, schema)
        if not problems:
            cypher = ensure_limit(cypher, settings.graph_nl_row_limit)
            records = await asyncio.to_thread(graph_store.run_cypher, cypher)
            if records or attempts >= settings.graph_nl_max_attempts:
                history.append({"cypher": cypher, "problems": [], "rows": len(records)})
                break
            # An empty result is the classic silent failure (wrong direction,
            # wrong property value casing) — give the model one more look.
            problems = [
                "The query ran but returned no rows. Re-check relationship directions against the "
                "schema, and whether literal values (names, categories) match the data's exact spelling."
            ]
        history.append({"cypher": cypher, "problems": problems})
        prompt = _FIX_PROMPT.format(base=base, cypher=cypher, problems="\n".join(f"- {p}" for p in problems))
    else:
        raise ValueError(
            f"Couldn't produce a valid query in {attempts} attempts. Last problems: "
            + "; ".join(history[-1]["problems"]) if history else "no output"
        )

    mlflow.update_current_trace(tags={"graph_nl_attempts": str(attempts)})
    if redact is not None:
        records = redact(records, cypher)

    result = {
        "question": question,
        "cypher": cypher,
        "records": records,
        "count": len(records),
        "attempts": attempts,
        "attempt_history": history,
        "ontology_version": ontology_version,
        "examples_used": [e["question"] for e in examples],
    }
    if synthesize:
        shown = records[:50]
        result["answer"] = (
            await llm.generate_json(
                _ANSWER_PROMPT.format(
                    question=question, cypher=cypher, count=len(records),
                    shown=f", first {len(shown)} shown" if len(shown) < len(records) else "",
                    rows=json.dumps(shown, default=str, indent=1),
                ),
                ANSWER_JSON_SCHEMA,
                timeout=300,
            )
        ).get("answer")
    return result
