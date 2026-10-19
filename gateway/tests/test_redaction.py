from core import redaction


def _redact(sql, columns, row):
    return redaction.redact_rows(columns, [row], sql=sql)[0]


def test_plain_pii_column_masked():
    out = _redact("SELECT first_name, city FROM t", ["first_name", "city"], ["Alice", "Paris"])
    assert out == ["A***", "Paris"]


def test_alias_does_not_bypass():
    out = _redact("SELECT first_name AS x, city FROM t", ["x", "city"], ["Alice", "Paris"])
    assert out[0] != "Alice" and out[1] == "Paris"


def test_unaliased_expression_caught_by_position():
    # DuckDB names this column something like concat(first_name, last_name)
    out = _redact(
        "SELECT concat(first_name, ' ', last_name), city FROM t",
        ["concat(first_name, ' ', last_name)", "city"],
        ["Alice Smith", "Paris"],
    )
    assert out[0] != "Alice Smith" and out[1] == "Paris"


def test_cte_alias_propagates():
    sql = "WITH a AS (SELECT email AS contact, city FROM t) SELECT contact AS c2, city FROM a"
    out = _redact(sql, ["c2", "city"], ["alice@example.com", "Paris"])
    assert out[0] != "alice@example.com" and out[1] == "Paris"


def test_count_of_pii_is_not_masked():
    out = _redact("SELECT count(DISTINCT email) AS n FROM t", ["n"], [42])
    assert out == [42]


def test_union_masks_if_either_side_is_pii():
    sql = "SELECT city AS v FROM t UNION ALL SELECT last_name AS v FROM t"
    out = _redact(sql, ["v"], ["Smith"])
    assert out[0] != "Smith"


def test_star_falls_back_to_names():
    out = _redact("SELECT * FROM t", ["email", "city"], ["alice@example.com", "Paris"])
    assert out[0] != "alice@example.com" and out[1] == "Paris"


def test_email_shaped_value_masked_anywhere():
    out = _redact("SELECT note FROM t", ["note"], ["bob@example.org"])
    assert out[0] != "bob@example.org"


def test_cypher_alias_through_with():
    cypher = "MATCH (c:Customer) WITH c.first_name AS fn, c.city AS city RETURN fn AS who, city"
    recs = redaction.redact_records([{"who": "Alice", "city": "Paris"}], cypher=cypher)
    assert recs[0]["who"] != "Alice" and recs[0]["city"] == "Paris"


def test_cypher_dotted_key_without_alias():
    recs = redaction.redact_records([{"c.first_name": "Alice", "c.city": "Paris"}])
    assert recs[0]["c.first_name"] != "Alice" and recs[0]["c.city"] == "Paris"


def test_whole_node_returned_is_masked_inside():
    recs = redaction.redact_records([{"c": {"first_name": "Alice", "email": "a@b.co", "city": "Paris"}}])
    node = recs[0]["c"]
    assert node["first_name"] != "Alice" and node["email"] != "a@b.co" and node["city"] == "Paris"


def test_collected_nodes_masked():
    recs = redaction.redact_records([{"people": [{"last_name": "Smith"}, {"last_name": "Jones"}]}])
    assert all(p["last_name"] not in ("Smith", "Jones") for p in recs[0]["people"])


def test_cypher_count_not_masked():
    cypher = "MATCH (c:Customer) RETURN count(c.email) AS emails"
    recs = redaction.redact_records([{"emails": 10}], cypher=cypher)
    assert recs[0]["emails"] == 10
