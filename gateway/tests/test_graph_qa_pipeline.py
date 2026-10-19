"""The /graph/nl pipeline with the local LLM and Neo4j mocked: the repair
loop must feed validation problems back and only run a query that passes."""
import asyncio
from unittest import mock

from services import graph_qa


def _run(coro):
    return asyncio.run(coro)


def test_repair_loop_fixes_unknown_label_then_answers():
    bad = {"reasoning": "", "cypher": "MATCH (c:Client)-[:PLACED]->(o:Order) RETURN count(o) AS n"}
    good = {"reasoning": "", "cypher": "MATCH (c:Customer)-[:PLACED]->(o:Order) RETURN count(o) AS n"}
    answer = {"answer": "There are 8000 orders."}
    llm_outputs = iter([bad, good, answer])
    prompts = []

    async def fake_generate_json(prompt, schema, timeout=300):
        prompts.append(prompt)
        return next(llm_outputs)

    def fake_explain(cypher):
        if ":Client" in cypher:
            return [{"code": "Neo.ClientNotification.Statement.UnknownLabelWarning",
                     "title": "Unknown label", "description": "Client does not exist"}]
        return []

    with mock.patch.object(graph_qa.llm, "generate_json", fake_generate_json), \
         mock.patch.object(graph_qa.graph_store, "build_schema_context", return_value="(:Customer)-[:PLACED]->(:Order)"), \
         mock.patch.object(graph_qa.graph_store, "explain_cypher", side_effect=fake_explain), \
         mock.patch.object(graph_qa.graph_store, "run_cypher", return_value=[{"n": 8000}]) as run, \
         mock.patch.object(graph_qa, "retrieve_examples", return_value=[]):
        out = _run(graph_qa.answer("How many orders are there?"))

    assert out["attempts"] == 2
    assert ":Customer" in out["cypher"] and out["cypher"].endswith("LIMIT 200")
    assert "Unknown label" in prompts[1]          # the problem was fed back
    run.assert_called_once()                       # the bad query never executed
    assert out["answer"] == "There are 8000 orders."


def test_redaction_happens_before_answer():
    seen_by_answer = []

    async def fake_generate_json(prompt, schema, timeout=300):
        if "cypher" in schema["properties"]:
            return {"reasoning": "", "cypher": "MATCH (c:Customer) RETURN c.first_name AS name"}
        seen_by_answer.append(prompt)
        return {"answer": "ok"}

    with mock.patch.object(graph_qa.llm, "generate_json", fake_generate_json), \
         mock.patch.object(graph_qa.graph_store, "build_schema_context", return_value=""), \
         mock.patch.object(graph_qa.graph_store, "explain_cypher", return_value=[]), \
         mock.patch.object(graph_qa.graph_store, "run_cypher", return_value=[{"name": "Alice"}]), \
         mock.patch.object(graph_qa, "retrieve_examples", return_value=[]):
        from core import redaction
        out = _run(graph_qa.answer(
            "names?", redact=lambda recs, cy: redaction.redact_records(recs, cypher=cy)))

    assert out["records"][0]["name"] != "Alice"
    assert "Alice" not in seen_by_answer[0]


def test_gives_up_after_max_attempts():
    async def always_bad(prompt, schema, timeout=300):
        return {"reasoning": "", "cypher": "MATCH (n) DELETE n"}

    with mock.patch.object(graph_qa.llm, "generate_json", always_bad), \
         mock.patch.object(graph_qa.graph_store, "build_schema_context", return_value=""), \
         mock.patch.object(graph_qa, "retrieve_examples", return_value=[]):
        try:
            _run(graph_qa.answer("delete everything"))
        except ValueError as e:
            assert "attempts" in str(e)
        else:
            raise AssertionError("expected ValueError")
