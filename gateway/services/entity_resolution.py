"""
Builds Silver from Bronze using the knowledge graph for entity resolution —
the "bronze to silver built via knowledge graph" layer of the medallion
pipeline. Generic: no domain knowledge of what a "lead" or "customer" is,
same as graph_inference.py.

How it actually uses the graph, not just an in-memory algorithm with graph
vocabulary borrowed for the name: candidate-duplicate rows are loaded into
Neo4j as real nodes, SIMILAR_TO relationships (with a score) are written
between rows that match, and those relationships are genuinely inspectable
in Neo4j Browser afterward — they aren't discarded scaffolding. Connected-
component grouping then runs in Python (Union-Find) rather than Cypher,
because Neo4j Community edition doesn't ship the Graph Data Science library
that would provide it natively; the relationships it operates on are real
graph state, not a database, this is the same distinction call/production
teams make when they don't have GDS available.

Scaling: candidate pairs come from *blocking*, not all-pairs. Rows are only
compared when they share a block key — normalized email, or the first/last
three letters of a name column — so work grows with block sizes instead of
n². Prefix AND suffix keys mean a typo at either end of a name still lands
the pair in a shared block; a row with typos at both ends of every name and
no matching email will be missed, which is the standard blocking trade-off.

Merge rule for building each canonical Silver record: first non-null value
per column, in ascending primary-key order — deterministic and simple, not
claimed to be the "right" survivorship rule for every domain (a real
platform would let a company configure this per entity type).
"""
from __future__ import annotations

import difflib
from collections import defaultdict

import mlflow
from mlflow.entities import SpanType

from services import catalog, graph_store, sql_engine, write_log

_STAGING_LABEL_PREFIX = "_Resolve"
_WRITE_BATCH = 1000


def _pk_column(schema: str, table: str) -> str:
    cols = {c["name"] for c in catalog.get_columns(schema, table)}
    last_word = table.split("_")[-1]
    for candidate in (f"{table.rstrip('s')}_id", f"{last_word.rstrip('s')}_id", "id"):
        if candidate in cols:
            return candidate
    raise ValueError(f"Could not infer a primary key for {schema}.{table}")


def _similarity(a: dict, b: dict, email_col: str | None, name_cols: list[str]) -> float:
    if email_col and a.get(email_col) and b.get(email_col):
        if str(a[email_col]).strip().lower() == str(b[email_col]).strip().lower():
            return 1.0
    best = 0.0
    for col in name_cols:
        va, vb = a.get(col), b.get(col)
        if va and vb:
            best = max(best, difflib.SequenceMatcher(None, str(va).lower(), str(vb).lower()).ratio())
    return best


def _block_keys(row: dict, email_col: str | None, name_cols: list[str]) -> set[str]:
    keys = set()
    if email_col and row.get(email_col):
        keys.add("e:" + str(row[email_col]).strip().lower())
    for col in name_cols:
        v = "".join(str(row.get(col) or "").lower().split())
        if len(v) >= 3:
            keys.add(f"{col}:p:{v[:3]}")
            keys.add(f"{col}:s:{v[-3:]}")
    return keys


def candidate_pairs(rows: list[dict], email_col: str | None, name_cols: list[str]) -> set[tuple[int, int]]:
    """Index pairs (i < j) of rows sharing at least one block key."""
    blocks: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        for key in _block_keys(row, email_col, name_cols):
            blocks[key].append(i)
    pairs: set[tuple[int, int]] = set()
    for members in blocks.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                pairs.add((members[a], members[b]))
    return pairs


class _UnionFind:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


@mlflow.trace(name="entity_resolution.find_duplicate_groups", span_type=SpanType.WORKFLOW)
def find_duplicate_groups(schema: str, table: str, threshold: float = 0.85) -> dict:
    """
    Loads rows into Neo4j as staging nodes, writes real SIMILAR_TO edges
    between rows scoring above `threshold`, then groups via connected
    components (Union-Find over those edges, computed in Python).
    Returns {"pk_column": str, "groups": [[row, ...], ...], "edges_written": int}.
    """
    sql_engine.validate_identifier(schema, "schema")
    sql_engine.validate_identifier(table, "table")
    pk = _pk_column(schema, table)
    # read_all: execute()'s response cap would silently drop every row past
    # the first 1000 from Silver.
    result = sql_engine.read_all(f"SELECT * FROM iceberg.{schema}.{table}", schema=schema)
    rows = [dict(zip(result["columns"], r)) for r in result["rows"]]
    if not rows:
        return {"pk_column": pk, "groups": [], "edges_written": 0}

    columns = result["columns"]
    email_col = next((c for c in columns if c.lower() == "email"), None)
    name_cols = [c for c in columns if "name" in c.lower() and c != email_col]

    edges = []
    for i, j in candidate_pairs(rows, email_col, name_cols):
        score = _similarity(rows[i], rows[j], email_col, name_cols)
        if score >= threshold:
            edges.append({"a": rows[i][pk], "b": rows[j][pk], "score": round(score, 3)})

    label = f"{_STAGING_LABEL_PREFIX}_{table}"
    with graph_store._get_driver().session() as session:
        # Staging is rebuilt from scratch each run: leftover SIMILAR_TO edges
        # from an earlier run at a lower threshold would otherwise still be
        # read back below and merge groups this threshold should keep apart.
        session.run(f"MATCH (n:{label}) DETACH DELETE n")
        session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{pk} IS UNIQUE")
        for start in range(0, len(rows), _WRITE_BATCH):
            session.run(
                f"UNWIND $rows AS row MERGE (n:{label} {{{pk}: row.{pk}}}) SET n = row",
                rows=rows[start : start + _WRITE_BATCH],
            )
        for start in range(0, len(edges), _WRITE_BATCH):
            session.run(
                f"UNWIND $edges AS e "
                f"MATCH (a:{label} {{{pk}: e.a}}), (b:{label} {{{pk}: e.b}}) "
                f"MERGE (a)-[r:SIMILAR_TO]->(b) SET r.score = e.score",
                edges=edges[start : start + _WRITE_BATCH],
            )
        edges_written = len(edges)

        edge_pairs = session.run(
            f"MATCH (a:{label})-[r:SIMILAR_TO]->(b) WHERE r.score >= $threshold "
            f"RETURN a.{pk} AS a, b.{pk} AS b",
            threshold=threshold,
        ).data()

    uf = _UnionFind([r[pk] for r in rows])
    for e in edge_pairs:
        uf.union(e["a"], e["b"])

    groups_by_root: dict = defaultdict(list)
    rows_by_pk = {r[pk]: r for r in rows}
    for pk_value in rows_by_pk:
        groups_by_root[uf.find(pk_value)].append(rows_by_pk[pk_value])

    return {"pk_column": pk, "groups": list(groups_by_root.values()), "edges_written": edges_written}


def _sql_type(value) -> str:
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return "BIGINT"
    if isinstance(value, float):
        return "DOUBLE"
    return "VARCHAR"


def _sql_literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


@mlflow.trace(name="entity_resolution.build_silver_table", span_type=SpanType.WORKFLOW)
def build_silver_table(schema: str, table: str, target_schema: str = "silver", threshold: float = 0.85) -> dict:
    """
    Runs find_duplicate_groups(), merges each group into one canonical row
    (first non-null value per column, ascending PK order), writes the result
    to {target_schema}.{table} and a {target_schema}.{table}_id_map table
    (bronze_pk -> silver_pk, one row per original Bronze row) so downstream
    tables can re-point references at canonical IDs.
    """
    resolved = find_duplicate_groups(schema, table, threshold)
    pk = resolved["pk_column"]
    groups = resolved["groups"]
    if not groups:
        return {"bronze_rows": 0, "silver_rows": 0, "groups_merged": 0}

    columns = list(groups[0][0].keys())
    canonical_rows = []
    id_map_rows = []
    for group in groups:
        group_sorted = sorted(group, key=lambda r: r[pk])
        canonical = {}
        for col in columns:
            canonical[col] = next((r[col] for r in group_sorted if r.get(col) is not None), None)
        canonical_rows.append(canonical)
        for r in group_sorted:
            id_map_rows.append({"bronze_" + pk: r[pk], "silver_" + pk: canonical[pk]})

    sql_engine.validate_identifier(target_schema, "schema")
    sql_engine.ensure_schema(target_schema)

    col_defs = ", ".join(f'"{c}" {_sql_type(canonical_rows[0][c])}' for c in columns)
    sql_engine.execute_ddl(
        f"CREATE TABLE IF NOT EXISTS iceberg.{target_schema}.{table} ({col_defs})"
    )
    # Rebuild, don't append: re-running Silver used to INSERT the canonical
    # rows a second time on top of the first run's.
    sql_engine.execute_ddl(f"DELETE FROM iceberg.{target_schema}.{table}")
    values = ", ".join(
        "(" + ", ".join(_sql_literal(row[c]) for c in columns) + ")" for row in canonical_rows
    )
    sql_engine.execute_ddl(f"INSERT INTO iceberg.{target_schema}.{table} VALUES {values}")

    map_table = f"{table}_id_map"
    sql_engine.execute_ddl(
        f"CREATE TABLE IF NOT EXISTS iceberg.{target_schema}.{map_table} "
        f'("bronze_{pk}" BIGINT, "silver_{pk}" BIGINT)'
    )
    sql_engine.execute_ddl(f"DELETE FROM iceberg.{target_schema}.{map_table}")
    map_values = ", ".join(
        f"({r['bronze_' + pk]}, {r['silver_' + pk]})" for r in id_map_rows
    )
    sql_engine.execute_ddl(f"INSERT INTO iceberg.{target_schema}.{map_table} VALUES {map_values}")
    write_log.record_write(target_schema)

    return {
        "bronze_rows": sum(len(g) for g in groups),
        "silver_rows": len(canonical_rows),
        "groups_merged": sum(1 for g in groups if len(g) > 1),
        "silver_table": f"iceberg.{target_schema}.{table}",
        "id_map_table": f"iceberg.{target_schema}.{map_table}",
    }
