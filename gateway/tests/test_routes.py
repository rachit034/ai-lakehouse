"""HTTP-level checks through the real FastAPI app: auth, truncation flag, and
business-role redaction. DuckDB is local in-memory; LanceDB/vector store is
stubbed (not exercised here)."""
import sys
from unittest import mock

import duckdb
import pytest

sys.modules.setdefault("services.vector_store", mock.MagicMock())

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from core.config import settings  # noqa: E402
from services import sql_engine  # noqa: E402


@pytest.fixture(autouse=True)
def local_duckdb(monkeypatch):
    conn = duckdb.connect(":memory:")
    conn.execute("ATTACH ':memory:' AS iceberg")
    conn.execute("CREATE SCHEMA iceberg.demo")
    conn.execute(
        "CREATE TABLE iceberg.demo.customers AS SELECT range AS customer_id, "
        "'Name' || range AS first_name, 'u' || range || '@x.com' AS email, 'Paris' AS city FROM range(1200)"
    )
    monkeypatch.setattr(sql_engine, "_conn", conn)


client = TestClient(main.app)


def _sql(key, sql):
    return client.post("/query/sql", json={"sql": sql, "schema": "demo"}, headers={"X-API-Key": key})


def test_missing_key_401():
    assert client.post("/query/sql", json={"sql": "SELECT 1"}).status_code == 401


def test_business_cannot_ingest():
    r = client.post("/ingest/embed", json={"table": "customers", "text_field": "city"},
                    headers={"X-API-Key": settings.api_key_business})
    assert r.status_code == 403


def test_truncated_flag_over_http():
    r = _sql(settings.api_key_data_engineer, "SELECT * FROM customers")
    body = r.json()
    assert r.status_code == 200 and body["row_count"] == 1000 and body["truncated"] is True


def test_business_alias_bypass_is_closed():
    r = _sql(settings.api_key_business, "SELECT first_name AS n, email AS e, city FROM customers LIMIT 1")
    row = r.json()["rows"][0]
    assert row[0] != "Name0" and row[1] != "u0@x.com" and row[2] == "Paris"


def test_engineer_sees_raw_values():
    r = _sql(settings.api_key_data_engineer, "SELECT first_name FROM customers ORDER BY customer_id LIMIT 1")
    assert r.json()["rows"][0][0] == "Name0"


def test_named_key_works():
    r = _sql("alice-key", "SELECT count(*) AS n FROM customers")
    assert r.status_code == 200 and r.json()["rows"][0][0] == 1200


def test_bad_identifier_rejected_on_embed():
    r = client.post("/ingest/embed", json={"schema": "demo; DROP", "table": "customers", "text_field": "city"},
                    headers={"X-API-Key": settings.api_key_data_engineer})
    assert r.status_code == 400
