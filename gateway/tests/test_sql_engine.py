import threading

import duckdb
import pytest

from services import sql_engine


@pytest.fixture(autouse=True)
def local_duckdb(monkeypatch):
    conn = duckdb.connect(":memory:")
    conn.execute("ATTACH ':memory:' AS iceberg")
    conn.execute("CREATE SCHEMA iceberg.demo")
    conn.execute("CREATE TABLE iceberg.demo.t AS SELECT range AS x FROM range(1500)")
    monkeypatch.setattr(sql_engine, "_conn", conn)
    yield conn


def test_truncation_is_reported():
    r = sql_engine.execute("SELECT x FROM t ORDER BY x", schema="demo")
    assert r["row_count"] == 1000 and r["truncated"] is True


def test_small_result_not_truncated():
    r = sql_engine.execute("SELECT x FROM t WHERE x < 10", schema="demo")
    assert r["row_count"] == 10 and r["truncated"] is False


def test_read_all_returns_everything():
    r = sql_engine.read_all("SELECT x FROM t", schema="demo")
    assert r["row_count"] == 1500 and r["truncated"] is False


def test_concurrent_reads():
    errors, counts = [], []

    def worker():
        try:
            counts.append(sql_engine.execute("SELECT count(*) FROM t", schema="demo")["rows"][0][0])
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and counts == [1500] * 16


@pytest.mark.parametrize("sql", ["DROP TABLE t", "SELECT 1; DROP TABLE t", "INSERT INTO t VALUES (1)"])
def test_writes_rejected(sql):
    with pytest.raises(ValueError):
        sql_engine.execute(sql, schema="demo")


@pytest.mark.parametrize("name", ["orders", "_tmp1", "Gold_2"])
def test_valid_identifiers(name):
    assert sql_engine.validate_identifier(name) == name


@pytest.mark.parametrize("name", ["a;drop", "x y", "1abc", "t--", "a.b", ""])
def test_invalid_identifiers(name):
    with pytest.raises(ValueError):
        sql_engine.validate_identifier(name)
