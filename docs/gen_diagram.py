"""
Generate the architecture diagrams as draw.io files plus SVGs that render on
GitHub (each SVG embeds its draw.io model, so draw.io can open either file).

    python3 docs/gen_diagram.py docs

Writes architecture.drawio/.svg (current system) and
architecture-target.drawio/.svg (proposed, see ARCHITECTURE-PROPOSAL.md).
"""
import html
import sys
from pathlib import Path

PAL = {
    "client": ("#dae8fc", "#6c8ebf"),
    "clientbox": ("#ffffff", "#6c8ebf"),
    "gw": ("#d5e8d4", "#82b366"),
    "gwbox": ("#ffffff", "#82b366"),
    "infra": ("#fff2cc", "#d6b656"),
    "infrabox": ("#ffffff", "#d6b656"),
    "pipe": ("#e1d5e7", "#9673a6"),
    "pipebox": ("#ffffff", "#9673a6"),
    "work": ("#f8cecc", "#b85450"),
    "workbox": ("#ffffff", "#b85450"),
    "new": ("#ffffff", "#d79b00"),
}
FONT = "Helvetica, Arial, sans-serif"


class Diagram:
    def __init__(self, title: str, width: int, height: int):
        self.title, self.W, self.H = title, width, height
        self.nodes, self.edges = [], []

    def lane(self, i, x, y, w, h, pal, title):
        self.nodes.append((i, x, y, w, h, pal, title, [], "lane"))

    def box(self, i, x, y, w, h, pal, title, lines=(), kind="box"):
        self.nodes.append((i, x, y, w, h, pal, title, list(lines), kind))

    def edge(self, src, tgt, exit_, entry, label="", dashed=False):
        self.edges.append((src, tgt, exit_, entry, label, dashed))

    def _pt(self, nid, f):
        _, x, y, w, h, *_ = next(n for n in self.nodes if n[0] == nid)
        return x + f[0] * w, y + f[1] * h

    def drawio(self) -> str:
        cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>',
                 f'<mxCell id="title" value="{html.escape(self.title)}" style="text;html=1;fontSize=22;fontStyle=1;'
                 f'align=left;verticalAlign=middle;" vertex="1" parent="1"><mxGeometry x="20" y="10" width="900" '
                 f'height="40" as="geometry"/></mxCell>']
        for i, x, y, w, h, pal, title, lines, kind in self.nodes:
            fill, stroke = PAL[pal]
            if kind == "lane":
                val = html.escape(f"<b>{html.escape(title)}</b>")
                style = (f"rounded=1;arcSize=2;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
                         "align=left;verticalAlign=top;spacingLeft=10;spacingTop=4;fontSize=14;")
            else:
                inner = f"<b>{html.escape(title)}</b>" + "".join(
                    f'<br><font style="font-size:11px">{html.escape(l)}</font>' for l in lines)
                val = html.escape(inner)
                extra = "dashed=1;" if kind == "host" else ("strokeWidth=2;" if kind == "new" else "")
                style = f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontSize=13;{extra}"
            cells.append(f'<mxCell id="{i}" value="{val}" style="{style}" vertex="1" parent="1"><mxGeometry '
                         f'x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        for k, (s, t, ex, en, label, dashed) in enumerate(self.edges):
            style = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeColor=#555555;"
                     f"fontSize=11;exitX={ex[0]:.4f};exitY={ex[1]};exitDx=0;exitDy=0;labelPosition=center;"
                     f"entryX={en[0]:.4f};entryY={en[1]};entryDx=0;entryDy=0;"
                     + ("dashed=1;" if dashed else "") + "labelBackgroundColor=#ffffff;")
            cells.append(f'<mxCell id="e{k}" value="{html.escape(label)}" style="{style}" edge="1" parent="1" '
                         f'source="{s}" target="{t}"><mxGeometry relative="1" as="geometry"/></mxCell>')
        return (f'<mxfile host="app.diagrams.net"><diagram id="arch" name="{html.escape(self.title)}">'
                f'<mxGraphModel dx="{self.W}" dy="{self.H}" grid="1" gridSize="10" guides="1" tooltips="1" '
                f'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{self.W}" '
                f'pageHeight="{self.H}" math="0" shadow="0"><root>{"".join(cells)}</root></mxGraphModel>'
                f'</diagram></mxfile>')

    def svg(self, content: str) -> str:
        o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.W}" height="{self.H}" '
             f'viewBox="0 0 {self.W} {self.H}" content="{html.escape(content)}">',
             '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
             'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#555"/></marker></defs>',
             f'<rect width="{self.W}" height="{self.H}" fill="#ffffff"/>',
             f'<text x="20" y="38" font-family="{FONT}" font-size="22" font-weight="bold" fill="#222">'
             f'{html.escape(self.title)}</text>']
        for i, x, y, w, h, pal, title, lines, kind in self.nodes:
            fill, stroke = PAL[pal]
            if kind == "lane":
                o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}"/>')
                o.append(f'<text x="{x + 10}" y="{y + 20}" font-family="{FONT}" font-size="14" font-weight="bold" '
                         f'fill="#222">{html.escape(title)}</text>')
                continue
            extra = (' stroke-dasharray="6 4"' if kind == "host" else "") + (
                ' stroke-width="2.4"' if kind == "new" else ' stroke-width="1.2"')
            o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}"{extra}/>')
            total = 17 + 15 * len(lines)
            cy, cx = y + h / 2 - total / 2 + 12, x + w / 2
            o.append(f'<text x="{cx}" y="{cy}" text-anchor="middle" font-family="{FONT}" font-size="13" '
                     f'font-weight="bold" fill="#222">{html.escape(title)}</text>')
            cy += 17
            for l in lines:
                o.append(f'<text x="{cx}" y="{cy}" text-anchor="middle" font-family="{FONT}" font-size="11" '
                         f'fill="#444">{html.escape(l)}</text>')
                cy += 15
        for s, t, ex, en, label, dashed in self.edges:
            x1, y1 = self._pt(s, ex)
            x2, y2 = self._pt(t, en)
            if ex[1] in (0, 1) and en[1] in (0, 1):
                my = (y1 + y2) / 2
                d = f"M{x1},{y1} L{x1},{my} L{x2},{my} L{x2},{y2}"
                lx, ly = x1, y1 + (14 if y2 > y1 else -14)
            else:
                d = f"M{x1},{y1} L{x2},{y2}"
                lx, ly = (x1 + x2) / 2, y1 - 6
            da = ' stroke-dasharray="6 4"' if dashed else ""
            o.append(f'<path d="{d}" fill="none" stroke="#555" stroke-width="1.4" marker-end="url(#arr)"{da}/>')
            if label:
                tw = 6.2 * len(label) + 8
                o.append(f'<rect x="{lx - tw / 2}" y="{ly - 9}" width="{tw}" height="16" fill="#ffffff" opacity="0.9"/>')
                o.append(f'<text x="{lx}" y="{ly + 3}" text-anchor="middle" font-family="{FONT}" font-size="11" '
                         f'fill="#333">{html.escape(label)}</text>')
        o.append("</svg>")
        return "\n".join(o)

    def write(self, out: Path, stem: str) -> None:
        xml = self.drawio()
        (out / f"{stem}.drawio").write_text(xml)
        (out / f"{stem}.svg").write_text(self.svg(xml))


# ── Current architecture ─────────────────────────────────────────────────────

def current() -> Diagram:
    d = Diagram("ai-lakehouse — Architecture", 1500, 1110)
    d.lane("L1", 20, 60, 1460, 150, "client", "Clients")
    d.box("streamlit", 40, 100, 260, 90, "clientbox", "Streamlit Query Console",
          ["NL · graph question · SQL · Cypher", ":8501"])
    d.box("agents", 340, 100, 520, 90, "clientbox", "Claude Agent SDK agents",
          ["ingestion (engineer) · modeler (modeler)", "data_analyst · graph · router (business)"])
    d.box("mcp", 900, 100, 260, 90, "clientbox", "MCP Server (FastMCP)", ["17 tools → thin HTTP calls", "to the gateway"])
    d.box("rest", 1200, 100, 260, 90, "clientbox", "Any HTTP client",
          ["Swagger UI /docs · curl", "scripts/eval_graph_nl.py"])

    d.lane("L2", 20, 250, 1460, 370, "gw", "AI Gateway — FastAPI :8088  (DuckDB runs embedded in this process)")
    d.box("auth", 40, 290, 460, 60, "gwbox", "Auth (X-API-Key → principal + role)",
          ["data_engineer · data_modeler · business"])
    d.box("trace", 520, 290, 440, 60, "gwbox", "Tracing middleware",
          ["every LLM / SQL / Cypher call → MLflow span"])
    d.box("pii", 980, 290, 480, 60, "gwbox", "PII redaction (query lineage-aware)",
          ["static fields + DataHub tags, aliases traced"])
    d.box("r_query", 40, 375, 340, 75, "gwbox", "/query", ["sql · nl · vector · schema · metrics", "all roles"])
    d.box("r_ingest", 400, 375, 340, 75, "gwbox", "/ingest", ["csv → Iceberg · embed → LanceDB", "data_engineer"])
    d.box("r_graph", 760, 375, 340, 75, "gwbox", "/graph",
          ["infer · materialize · examples (modeler)", "nl · query · ontology · freshness (all)"])
    d.box("r_med", 1120, 375, 340, 75, "gwbox", "/medallion", ["silver preview/build (modeler)", "gold query/materialize (all)"])
    svc = [
        ("s_duck", "DuckDB engine", ["concurrent reads, serial writes", "read-only guard · row cap flag"]),
        ("s_llm", "Local LLM service", ["NL → SQL (semantic cache)", "JSON-schema outputs"]),
        ("s_vec", "Vector store", ["fastembed bge-small", "LanceDB on s3://lance"]),
        ("s_graph", "Graph Q&A + ontology", ["versioned ontology drafts", "NL→Cypher, EXPLAIN, repair"]),
        ("s_er", "Entity resolution + Gold", ["blocked matching → SIMILAR_TO", "Gold CTAS + lineage"]),
        ("s_cat", "Catalog + metrics", ["schema context, business", "metrics, DataHub client"]),
    ]
    for k, (i, t, ls) in enumerate(svc):
        d.box(i, 40 + k * 239, 475, 225, 120, "gwbox", t, ls)

    d.lane("L3", 20, 660, 1460, 250, "infra", "Infrastructure — docker-compose (local stand-ins for AWS)")
    d.box("gravitino", 40, 700, 166, 95, "infrabox", "Apache Gravitino", ["Iceberg REST catalog", "≈ AWS Glue · :19090"])
    d.box("postgres", 40, 820, 166, 70, "infrabox", "PostgreSQL", ["catalog store · :5433"])
    d.box("minio", 218, 700, 166, 95, "infrabox", "MinIO", ["Iceberg data + Lance", "≈ Amazon S3 · :9100"])
    d.box("redis", 396, 700, 166, 95, "infrabox", "Redis Stack", ["NL→SQL cache ·", "write log · :6379"])
    d.box("ollama", 574, 700, 166, 95, "infrabox", "Ollama (host)", ["local LLM, qwen3:4b", ":11434"], kind="host")
    d.box("neo4j", 840, 700, 166, 95, "infrabox", "Neo4j", ["graph + ontologies", "≈ Amazon Neptune · :7687"])
    d.box("mlflow", 1030, 700, 166, 95, "infrabox", "MLflow", ["traces + experiments", ":5050"])
    d.box("datahub", 1286, 700, 166, 95, "infrabox", "DataHub (host)", ["optional catalog,", "tags · lineage · :18080"], kind="host")

    d.lane("L4", 20, 940, 1460, 150, "pipe", "Data pipeline / medallion layers (Iceberg schemas in one catalog)")
    pipe = [
        ("p_gen", "generate.py", ["Faker star schema:", "customers · products · orders"]),
        ("p_load", "load_to_iceberg.py", ["Parquet → MinIO →", "Iceberg via DuckDB"]),
        ("p_bronze", "Bronze", ["raw Iceberg tables", "(+ /ingest/csv)"]),
        ("p_silver", "Silver", ["deduplicated entities,", "resolved via the KG"]),
        ("p_gold", "Gold", ["NL-defined aggregates,", "materialized + lineage"]),
        ("p_kg", "Knowledge graph", ["LLM ontology (reviewed) →", "Neo4j; stale flag on writes"]),
    ]
    for k, (i, t, ls) in enumerate(pipe):
        d.box(i, 40 + k * 244, 985, 200, 80, "pipebox", t, ls)

    d.edge("agents", "mcp", (1, .5), (0, .5), "stdio")
    d.edge("streamlit", "L2", (.5, 1), ((170 - 20) / 1460, 0), "HTTP + X-API-Key")
    d.edge("mcp", "L2", (.5, 1), ((1030 - 20) / 1460, 0), "HTTP + per-role key")
    d.edge("rest", "L2", (.5, 1), ((1330 - 20) / 1460, 0))
    d.edge("s_duck", "gravitino", (.3, 1), (.5, 0), "Iceberg REST")
    d.edge("s_duck", "minio", (.8, 1), (.5, 0), "S3 (httpfs)")
    d.edge("gravitino", "postgres", (.5, 1), (.5, 0))
    d.edge("s_llm", "redis", (.4, 1), (.5, 0))
    d.edge("s_llm", "ollama", (.8, 1), (.5, 0))
    d.edge("s_graph", "neo4j", (.5, 1), (.4, 0), "Bolt")
    d.edge("s_er", "neo4j", (.2, 1), (.8, 0))
    d.edge("L2", "mlflow", ((1113 - 20) / 1460, 1), (.5, 0), "traces", dashed=True)
    d.edge("s_cat", "datahub", (.6, 1), (.5, 0), "REST (optional)", dashed=True)
    for a, b in (("p_gen", "p_load"), ("p_load", "p_bronze"), ("p_bronze", "p_silver"), ("p_silver", "p_gold")):
        d.edge(a, b, (1, .5), (0, .5))
    d.edge("p_load", "minio", (.2, 0), (.5, 1), dashed=True)
    return d


# ── Target architecture (ARCHITECTURE-PROPOSAL.md) ───────────────────────────

def target() -> Diagram:
    d = Diagram("ai-lakehouse — Target architecture (local-LLM knowledge graph)", 1500, 1250)
    d.lane("T1", 20, 60, 1460, 140, "client", "Clients + identity")
    d.box("t_ui", 40, 100, 300, 80, "clientbox", "Query console", ["NL · graph Q&A · SQL · Cypher"])
    d.box("t_agents", 360, 100, 400, 80, "clientbox", "Agents over MCP",
          ["Claude today → local agent model", "when evals say it's ready"])
    d.box("t_oidc", 780, 100, 320, 80, "new", "OIDC identity (Keycloak)",
          ["per-person tokens + groups → roles"], kind="new")
    d.box("t_eval", 1120, 100, 340, 80, "new", "Eval suite (CI-gated)",
          ["graph Q&A + NL→SQL cases", "blocks model/prompt regressions"], kind="new")

    d.lane("T2", 20, 240, 1460, 300, "gw", "Gateway — stateless API, N replicas (no heavy work in requests)")
    d.box("t_policy", 40, 280, 460, 60, "gwbox", "AuthN/Z + policy",
          ["JWT → role; business reads masked views only"])
    d.box("t_router", 520, 280, 440, 60, "new", "Question router + planner",
          ["graph · SQL · vector · or multi-step plan"], kind="new")
    d.box("t_jobsapi", 980, 280, 480, 60, "gwbox", "Job API",
          ["submit build/materialize/Silver → job id + status"])
    d.box("t_link", 40, 370, 340, 140, "new", "Entity linking",
          ["vector search maps 'headphones'", "→ exact node names before", "Cypher is written"], kind="new")
    d.box("t_gqa", 400, 370, 380, 140, "gwbox", "Graph Q&A (local LLM)",
          ["ontology + schema grounding", "verified few-shot examples", "constrained JSON · EXPLAIN check",
           "repair loop · read-only · LIMIT"])
    d.box("t_sql", 800, 370, 300, 140, "gwbox", "NL → SQL (local LLM)",
          ["metrics layer · literal-safe", "semantic cache", "read-only guard"])
    d.box("t_onto", 1120, 370, 340, 140, "gwbox", "Ontology registry",
          ["draft → review → active", "version diff · DataHub glossary", "descriptions feed Q&A"])

    d.lane("T3", 20, 580, 1460, 170, "work", "Workers — job queue (arq/Redis), retries, progress")
    d.box("t_w_onto", 40, 620, 340, 110, "workbox", "Ontology inference",
          ["larger local model (14–32B)", "offline, quality over speed"])
    d.box("t_w_sync", 400, 620, 380, 110, "new", "Incremental graph sync",
          ["Iceberg snapshot diff → upsert", "changed rows only; replaces", "full rebuilds + stale flags"], kind="new")
    d.box("t_w_er", 800, 620, 300, 110, "workbox", "Silver entity resolution",
          ["blocking + scoring", "graph-native components (GDS)"])
    d.box("t_w_embed", 1120, 620, 340, 110, "workbox", "Embeddings",
          ["batch, resumable", "node names + descriptions"])

    d.lane("T4", 20, 790, 1460, 200, "infra", "Engines")
    d.box("t_llm", 40, 830, 300, 130, "infrabox", "Local LLM server",
          ["Ollama → vLLM", "small model: Q&A (fast)", "large model: ontology"])
    d.box("t_qe", 360, 830, 260, 130, "infrabox", "Query engine",
          ["DuckDB (local, 1 user)", "Trino / Athena when", "multi-user"])
    d.box("t_neo", 640, 830, 260, 130, "infrabox", "Neo4j (+ GDS)",
          ["graph + ontologies", "+ example library", "≈ Neptune"])
    d.box("t_lance", 920, 830, 240, 130, "infrabox", "LanceDB", ["entity + text vectors"])
    d.box("t_obs", 1180, 830, 280, 130, "infrabox", "MLflow + Redis",
          ["traces, eval runs", "queue, cache"])

    d.lane("T5", 20, 1030, 1460, 190, "pipe", "Data (Iceberg on MinIO/S3 · Gravitino catalog · DataHub governance)")
    d.box("t_bronze", 40, 1075, 260, 110, "pipebox", "Bronze", ["raw tables"])
    d.box("t_silver", 330, 1075, 260, 110, "pipebox", "Silver", ["resolved entities"])
    d.box("t_gold", 620, 1075, 260, 110, "pipebox", "Gold", ["metrics, lineage"])
    d.box("t_views", 910, 1075, 260, 110, "new", "Masked views per role",
          ["PII removed in the engine,", "not after the query"], kind="new")
    d.box("t_snap", 1200, 1075, 260, 110, "pipebox", "Snapshots / changelog",
          ["read by the incremental", "graph sync worker"])

    d.edge("t_ui", "T2", (.5, 1), ((190 - 20) / 1460, 0), "JWT")
    d.edge("t_agents", "T2", (.5, 1), ((560 - 20) / 1460, 0), "MCP → HTTP")
    d.edge("t_router", "t_gqa", (.2, 1), (.6, 0))
    d.edge("t_router", "t_sql", (.8, 1), (.5, 0))
    d.edge("t_link", "t_gqa", (1, .5), (0, .5))
    d.edge("t_jobsapi", "T3", ((1110 - 980) / 480, 1), ((1110 - 20) / 1460, 0), "enqueue")
    d.edge("t_w_sync", "t_neo", (.8, 1), (.3, 0))
    d.edge("t_w_onto", "t_llm", (.5, 1), (.5, 0))
    for a, b in (("t_bronze", "t_silver"), ("t_silver", "t_gold"), ("t_gold", "t_views")):
        d.edge(a, b, (1, .5), (0, .5))
    return d


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs")
    out.mkdir(parents=True, exist_ok=True)
    current().write(out, "architecture")
    target().write(out, "architecture-target")
    print("ok")
