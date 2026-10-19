"""
Query console for the AI Lakehouse prototype — run NL, SQL, and Cypher
queries against the gateway from one browser tab.

Talks only to the gateway (GATEWAY_URL) over HTTP — no direct connections to
DuckDB/Neo4j, same boundary every other client (MCP server, agents) respects.
"""
from __future__ import annotations

import html
import os

import requests
import streamlit as st

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8088").rstrip("/")

st.set_page_config(page_title="AI Lakehouse Query Console", page_icon="🌊", layout="wide")

# One key per role — see gateway/core/auth.py for what each can reach.
# data_engineer/data_modeler additionally get raw (unredacted) query results;
# business always gets PII masked.
_ROLE_KEYS = {
    "business": os.environ.get("API_KEY_BUSINESS", ""),
    "data_engineer": os.environ.get("API_KEY_DATA_ENGINEER", ""),
    "data_modeler": os.environ.get("API_KEY_DATA_MODELER", ""),
}

if "role" not in st.session_state:
    st.session_state["role"] = "business"


def call(method: str, path: str, **kwargs):
    headers = kwargs.pop("headers", {})
    headers["X-API-Key"] = _ROLE_KEYS.get(st.session_state.get("role", "business"), "")
    resp = requests.request(method, f"{GATEWAY_URL}{path}", timeout=120, headers=headers, **kwargs)
    if resp.status_code == 401:
        raise RuntimeError("401: Missing or invalid API key for the selected role.")
    if resp.status_code == 403:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"403: {detail} — switch role in the sidebar if this should be allowed.")
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"{resp.status_code}: {detail}")
    return resp.json()


def _render_table(columns: list[str], row_dicts: list[dict]) -> None:
    """
    Plain HTML table — deliberately not pandas/st.dataframe. Streamlit's
    st.dataframe converts through pandas -> pyarrow (pyarrow.pandas_compat
    .convert_column), and that conversion segfaults natively on real result
    shapes from both /query/sql (aggregate float columns) and /graph/query
    (heterogeneous record shapes) in this environment — reproduced against
    live data, not a fluke. An HTML table sidesteps pyarrow entirely.
    """
    header = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    body_rows = []
    for row in row_dicts:
        cells = "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    table_html = (
        "<div style='overflow-x:auto'><table style='width:100%'>"
        f"<thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def show_rows(columns: list[str], rows: list[list]) -> None:
    if not rows:
        st.info("No rows returned.")
        return
    _render_table(columns, [dict(zip(columns, row)) for row in rows])


def show_records(records: list[dict]) -> None:
    if not records:
        st.info("No records returned.")
        return
    columns = list({k for rec in records for k in rec.keys()})
    _render_table(columns, records)


st.title("🌊 AI Lakehouse Query Console")
st.caption(f"Gateway: `{GATEWAY_URL}`")

# ── Sidebar: role + schema + graph shape ────────────────────────────────────
with st.sidebar:
    st.header("Role")
    st.selectbox(
        "Acting as",
        options=["business", "data_engineer", "data_modeler"],
        key="role",
        help=(
            "business: read-only everywhere, PII always masked. "
            "data_engineer: + /ingest/*, unmasked query results. "
            "data_modeler: + /graph/build, /graph/infer*, /medallion/silver/*, "
            "unmasked query results. Missing the key for a role 401s; using "
            "the wrong role for a route 403s — see gateway/core/auth.py."
        ),
    )
    if not _ROLE_KEYS.get(st.session_state["role"]):
        st.error(f"No API key configured for role '{st.session_state['role']}' — check the container's env.")

    st.header("Tables")
    if st.button("Refresh tables"):
        st.session_state.pop("schema", None)
    if "schema" not in st.session_state:
        try:
            st.session_state["schema"] = call("GET", "/query/schema")
        except Exception as e:
            st.session_state["schema"] = None
            st.error(f"Schema fetch failed: {e}")
    schema = st.session_state.get("schema")
    if schema:
        for sch, tables in schema.items():
            with st.expander(sch, expanded=False):
                for table, cols in tables.items():
                    st.markdown(f"**{table}**")
                    st.caption(", ".join(c["name"] for c in cols))

    st.header("Knowledge graph")
    if st.button("Refresh graph stats"):
        st.session_state.pop("graph_stats", None)
    if "graph_stats" not in st.session_state:
        try:
            st.session_state["graph_stats"] = call("GET", "/graph/stats")
        except Exception as e:
            st.session_state["graph_stats"] = None
    gs = st.session_state.get("graph_stats")
    if gs and gs.get("nodes"):
        st.caption("Nodes: " + ", ".join(f"{n['label']} ({n['count']})" for n in gs["nodes"]))
        st.caption("Relationships: " + ", ".join(f"{r['type']} ({r['count']})" for r in gs["relationships"]))
    elif gs is not None:
        st.caption("Graph is empty — run build_knowledge_graph first.")

# ── Query tabs ────────────────────────────────────────────────────────────
tab_nl, tab_sql, tab_cypher = st.tabs(["🗣 Natural language", "🧮 SQL", "🕸 Cypher"])

with tab_nl:
    st.subheader("Ask a question in plain English")
    st.caption("Translated to SQL by the gateway's LLM, then run on DuckDB.")
    question = st.text_area(
        "Question",
        placeholder="e.g. Which product category has the highest average price?",
        key="nl_question",
    )
    c1, c2 = st.columns([3, 1])
    with c1:
        nl_schema = st.text_input("Schema (optional)", key="nl_schema")
    with c2:
        explain = st.checkbox("Show generated SQL", value=True, key="nl_explain")
    if st.button("Run", key="nl_run", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Translating and running..."):
                try:
                    result = call(
                        "POST",
                        "/query/nl",
                        json={"question": question, "schema": nl_schema or None, "explain": explain},
                    )
                    if result.get("generated_sql"):
                        st.code(result["generated_sql"], language="sql")
                    st.caption(f"{result['row_count']} rows · {'cached' if result.get('cached') else 'fresh'}")
                    show_rows(result["columns"], result["rows"])
                except Exception as e:
                    st.error(str(e))

with tab_sql:
    st.subheader("Run SQL directly on DuckDB/Iceberg")
    sql = st.text_area(
        "SQL",
        placeholder="SELECT * FROM iceberg.ecommerce.products LIMIT 10",
        height=140,
        key="sql_text",
    )
    sql_schema = st.text_input("Schema (optional)", key="sql_schema")
    if st.button("Run", key="sql_run", type="primary"):
        if not sql.strip():
            st.warning("Enter a SQL statement first.")
        else:
            with st.spinner("Running..."):
                try:
                    result = call("POST", "/query/sql", json={"sql": sql, "schema": sql_schema or None})
                    st.caption(f"{result['row_count']} rows")
                    show_rows(result["columns"], result["rows"])
                except Exception as e:
                    st.error(str(e))

with tab_cypher:
    st.subheader("Run Cypher on the knowledge graph")
    st.caption(
        "Nodes: Customer, Order, Product, Category, Supplier · "
        "Relationships: PLACED, CONTAINS, IN_CATEGORY, SUPPLIED_BY"
    )
    cypher = st.text_area(
        "Cypher",
        placeholder="MATCH (c:Customer)-[:PLACED]->(:Order) RETURN c.first_name, count(*) AS orders "
        "ORDER BY orders DESC LIMIT 10",
        height=140,
        key="cypher_text",
    )
    if st.button("Run", key="cypher_run", type="primary"):
        if not cypher.strip():
            st.warning("Enter a Cypher query first.")
        else:
            with st.spinner("Running..."):
                try:
                    result = call("POST", "/graph/query", json={"cypher": cypher})
                    st.caption(f"{result['count']} records")
                    show_records(result["records"])
                except Exception as e:
                    st.error(str(e))
