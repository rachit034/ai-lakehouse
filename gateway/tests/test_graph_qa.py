import pytest

from services.graph_qa import ensure_limit, static_problems


def test_valid_query_has_no_static_problems():
    q = "MATCH (c:Customer)-[:PLACED]->(o:Order) RETURN c.segment AS s, count(o) AS n"
    assert static_problems(q, None) == []


@pytest.mark.parametrize("q", [
    "MATCH (n) DETACH DELETE n",
    "MATCH (n:Product) SET n.price = 0 RETURN n",
    "CALL db.labels()",
    "MERGE (n:Product {name: 'x'}) RETURN n",
])
def test_write_or_call_rejected(q):
    assert any("not allowed" in p for p in static_problems(q, None))


def test_keyword_inside_string_literal_is_fine():
    q = "MATCH (p:Product {name: 'Set of 3 Knives'}) RETURN p.name AS name"
    assert static_problems(q, None) == []


def test_parameters_rejected():
    assert static_problems("MATCH (p:Product {name: $n}) RETURN p", None)


def test_internal_labels_rejected():
    assert static_problems("MATCH (o:_Ontology) RETURN o", None)
    assert static_problems("MATCH (p:Product) RETURN p._node_key AS k", None)


def test_schema_scope_required_when_given():
    q = "MATCH (p:Product) RETURN p.name AS name"
    assert static_problems(q, "ecommerce")
    assert static_problems("MATCH (p:Product {_schema: 'ecommerce'}) RETURN p.name AS name", "ecommerce") == []


def test_ensure_limit():
    assert ensure_limit("MATCH (n) RETURN n", 200).endswith("LIMIT 200")
    assert ensure_limit("MATCH (n) RETURN n LIMIT 5", 200).endswith("LIMIT 5")
    assert ensure_limit("MATCH (n) RETURN n;", 200).endswith("LIMIT 200")
