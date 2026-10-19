"""Generate docs/architecture.drawio and docs/architecture.svg from one layout spec."""
import html
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
W, H = 1500, 1110

PAL = {
    "client": ("#dae8fc", "#6c8ebf"),
    "gw": ("#d5e8d4", "#82b366"),
    "gwbox": ("#ffffff", "#82b366"),
    "infra": ("#fff2cc", "#d6b656"),
    "infrabox": ("#ffffff", "#d6b656"),
    "pipe": ("#e1d5e7", "#9673a6"),
    "pipebox": ("#ffffff", "#9673a6"),
    "clientbox": ("#ffffff", "#6c8ebf"),
}

# id, x, y, w, h, palette, title, [lines], kind ("lane" | "box" | "host")
NODES = []
def lane(i, x, y, w, h, pal, title):
    NODES.append((i, x, y, w, h, pal, title, [], "lane"))
def box(i, x, y, w, h, pal, title, lines=(), kind="box"):
    NODES.append((i, x, y, w, h, pal, title, list(lines), kind))

# ── Clients ──
lane("L1", 20, 60, 1460, 150, "client", "Clients")
box("streamlit", 40, 100, 260, 90, "clientbox", "Streamlit Query Console", ["NL · SQL · Cypher · role picker", ":8501"])
box("agents", 340, 100, 520, 90, "clientbox", "Claude Agent SDK agents",
    ["ingestion (engineer) · modeler (modeler)", "data_analyst · graph · router (business)"])
box("mcp", 900, 100, 260, 90, "clientbox", "MCP Server (FastMCP)", ["15 tools → thin HTTP calls", "to the gateway"])
box("rest", 1200, 100, 260, 90, "clientbox", "Any HTTP client", ["Swagger UI /docs · curl", "scripts/test-metrics.sh"])

# ── Gateway ──
lane("L2", 20, 250, 1460, 370, "gw", "AI Gateway — FastAPI :8088  (DuckDB runs embedded in this process)")
box("auth", 40, 290, 460, 60, "gwbox", "Role-based auth (X-API-Key)", ["data_engineer · data_modeler · business"])
box("trace", 520, 290, 440, 60, "gwbox", "Tracing middleware", ["every LLM / SQL / Cypher call → MLflow span"])
box("pii", 980, 290, 480, 60, "gwbox", "PII redaction at response boundary", ["static PII fields + DataHub governance tags"])
box("r_query", 40, 375, 340, 75, "gwbox", "/query", ["sql · nl · vector · schema · metrics", "all roles"])
box("r_ingest", 400, 375, 340, 75, "gwbox", "/ingest", ["csv → Iceberg · embed → LanceDB", "data_engineer"])
box("r_graph", 760, 375, 340, 75, "gwbox", "/graph", ["build · infer · materialize (modeler)", "query · stats · freshness (all)"])
box("r_med", 1120, 375, 340, 75, "gwbox", "/medallion", ["silver preview/build (modeler)", "gold query/materialize (all)"])
SVC_W = 225
svc = [
    ("s_duck", "DuckDB engine", ["iceberg + httpfs ext.", "read-only SQL guard (sqlglot)"]),
    ("s_llm", "LLM service", ["NL → SQL / Cypher", "Redis semantic cache"]),
    ("s_vec", "Vector store", ["fastembed bge-small", "LanceDB on s3://lance"]),
    ("s_graph", "Graph store + inference", ["Neo4j driver", "LLM-inferred graph model"]),
    ("s_er", "Entity resolution + Gold", ["Silver via SIMILAR_TO edges", "Gold CTAS + lineage"]),
    ("s_cat", "Catalog + metrics", ["schema context, business", "metrics, DataHub client"]),
]
for k, (i, t, ls) in enumerate(svc):
    box(i, 40 + k * (SVC_W + 14), 475, SVC_W, 120, "gwbox", t, ls)

# ── Infrastructure ──
lane("L3", 20, 660, 1460, 250, "infra", "Infrastructure — docker-compose (local stand-ins for AWS)")
IW = 166
box("gravitino", 40, 700, IW, 95, "infrabox", "Apache Gravitino", ["Iceberg REST catalog", "≈ AWS Glue · :19090"])
box("postgres", 40, 820, IW, 70, "infrabox", "PostgreSQL", ["catalog store · :5433"])
box("minio", 218, 700, IW, 95, "infrabox", "MinIO", ["Iceberg data + Lance", "≈ Amazon S3 · :9100"])
box("redis", 396, 700, IW, 95, "infrabox", "Redis Stack", ["semantic NL→SQL", "cache · :6379"])
box("ollama", 574, 700, IW, 95, "infrabox", "Ollama (host)", ["local LLM, qwen3:4b", ":11434"], kind="host")
box("neo4j", 840, 700, IW, 95, "infrabox", "Neo4j", ["knowledge graph", "≈ Amazon Neptune · :7687"])
box("mlflow", 1030, 700, IW, 95, "infrabox", "MLflow", ["traces + experiments", ":5050"])
box("datahub", 1286, 700, IW, 95, "infrabox", "DataHub (host)", ["optional catalog,", "tags · lineage · :18080"], kind="host")

# ── Data pipeline ──
lane("L4", 20, 940, 1460, 150, "pipe", "Data pipeline / medallion layers (Iceberg schemas in one catalog)")
PW = 200
pipe = [
    ("p_gen", "generate.py", ["Faker star schema:", "customers · products · orders"]),
    ("p_load", "load_to_iceberg.py", ["Parquet → MinIO →", "Iceberg via DuckDB"]),
    ("p_bronze", "Bronze", ["raw Iceberg tables", "(+ /ingest/csv)"]),
    ("p_silver", "Silver", ["deduplicated entities,", "resolved via the KG"]),
    ("p_gold", "Gold", ["NL-defined aggregates,", "materialized + lineage"]),
    ("p_kg", "Knowledge graph", ["inferred from schema +", "sample rows → Neo4j"]),
]
for k, (i, t, ls) in enumerate(pipe):
    box(i, 40 + k * (PW + 44), 985, PW, 80, "pipebox", t, ls)

# src, tgt, exit(x,y), entry(x,y), label, dashed
EDGES = [
    ("agents", "mcp", (1, .5), (0, .5), "stdio", False),
    ("streamlit", "L2", (.5, 1), ((170 - 20) / 1460, 0), "HTTP + X-API-Key", False),
    ("mcp", "L2", (.5, 1), ((1030 - 20) / 1460, 0), "HTTP + per-role key", False),
    ("rest", "L2", (.5, 1), ((1330 - 20) / 1460, 0), "", False),
    ("s_duck", "gravitino", (.3, 1), (.5, 0), "Iceberg REST", False),
    ("s_duck", "minio", (.8, 1), (.5, 0), "S3 (httpfs)", False),
    ("gravitino", "postgres", (.5, 1), (.5, 0), "", False),
    ("s_llm", "redis", (.4, 1), (.5, 0), "", False),
    ("s_llm", "ollama", (.8, 1), (.5, 0), "", False),
    ("s_graph", "neo4j", (.5, 1), (.4, 0), "Bolt", False),
    ("s_er", "neo4j", (.2, 1), (.8, 0), "", False),
    ("L2", "mlflow", ((1113 - 20) / 1460, 1), (.5, 0), "traces", True),
    ("s_cat", "datahub", (.6, 1), (.5, 0), "REST (optional)", True),
    ("p_gen", "p_load", (1, .5), (0, .5), "", False),
    ("p_load", "p_bronze", (1, .5), (0, .5), "", False),
    ("p_bronze", "p_silver", (1, .5), (0, .5), "", False),
    ("p_silver", "p_gold", (1, .5), (0, .5), "", False),
    ("p_load", "minio", (.2, 0), (.5, 1), "", True),
]
BY_ID = {n[0]: n for n in NODES}

def pt(nid, f):
    _, x, y, w, h, *_ = BY_ID[nid]
    return x + f[0] * w, y + f[1] * h

TITLE = "ai-lakehouse — Architecture"

# ── draw.io ──
def drawio_xml():
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>',
             f'<mxCell id="title" value="{html.escape(TITLE)}" style="text;html=1;fontSize=22;fontStyle=1;align=left;verticalAlign=middle;" vertex="1" parent="1"><mxGeometry x="20" y="10" width="800" height="40" as="geometry"/></mxCell>']
    for i, x, y, w, h, pal, title, lines, kind in NODES:
        fill, stroke = PAL[pal]
        if kind == "lane":
            val = html.escape(f"<b>{html.escape(title)}</b>")
            style = f"rounded=1;arcSize=2;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};align=left;verticalAlign=top;spacingLeft=10;spacingTop=4;fontSize=14;"
        else:
            inner = f"<b>{html.escape(title)}</b>" + "".join(f"<br><font style=\"font-size:11px\">{html.escape(l)}</font>" for l in lines)
            val = html.escape(inner)
            dash = "dashed=1;" if kind == "host" else ""
            style = f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontSize=13;{dash}"
        cells.append(f'<mxCell id="{i}" value="{val}" style="{style}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    for k, (s, t, ex, en, label, dashed) in enumerate(EDGES):
        style = (f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;strokeColor=#555555;fontSize=11;"
                 f"exitX={ex[0]:.4f};exitY={ex[1]};exitDx=0;exitDy=0;entryX={en[0]:.4f};entryY={en[1]};entryDx=0;entryDy=0;"
                 + ("dashed=1;" if dashed else "") + "labelBackgroundColor=#ffffff;")
        cells.append(f'<mxCell id="e{k}" value="{html.escape(label)}" style="{style}" edge="1" parent="1" source="{s}" target="{t}"><mxGeometry relative="1" as="geometry"/></mxCell>')
    model = "".join(cells)
    return (f'<mxfile host="app.diagrams.net"><diagram id="arch" name="Architecture">'
            f'<mxGraphModel dx="{W}" dy="{H}" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">'
            f'<root>{model}</root></mxGraphModel></diagram></mxfile>')

# ── SVG ──
def svg(content_attr):
    F = "Helvetica, Arial, sans-serif"
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" content="{html.escape(content_attr)}">',
         '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#555"/></marker></defs>',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="20" y="38" font-family="{F}" font-size="22" font-weight="bold" fill="#222">{html.escape(TITLE)}</text>']
    for i, x, y, w, h, pal, title, lines, kind in NODES:
        fill, stroke = PAL[pal]
        if kind == "lane":
            o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}"/>')
            o.append(f'<text x="{x + 10}" y="{y + 20}" font-family="{F}" font-size="14" font-weight="bold" fill="#222">{html.escape(title)}</text>')
            continue
        dash = ' stroke-dasharray="6 4"' if kind == "host" else ""
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.2"{dash}/>')
        n = 1 + len(lines)
        lh = [17] + [15] * len(lines)
        total = sum(lh)
        cy = y + h / 2 - total / 2 + 12
        cx = x + w / 2
        o.append(f'<text x="{cx}" y="{cy}" text-anchor="middle" font-family="{F}" font-size="13" font-weight="bold" fill="#222">{html.escape(title)}</text>')
        cy += 17
        for l in lines:
            o.append(f'<text x="{cx}" y="{cy}" text-anchor="middle" font-family="{F}" font-size="11" fill="#444">{html.escape(l)}</text>')
            cy += 15
    for s, t, ex, en, label, dashed in EDGES:
        x1, y1 = pt(s, ex)
        x2, y2 = pt(t, en)
        if ex[1] in (0, 1) and en[1] in (0, 1):  # vertical exit/entry
            my = (y1 + y2) / 2
            d = f"M{x1},{y1} L{x1},{my} L{x2},{my} L{x2},{y2}"
            lx, ly = x1, y1 + 14
        else:
            d = f"M{x1},{y1} L{x2},{y2}"
            lx, ly = (x1 + x2) / 2, y1 - 6
        da = ' stroke-dasharray="6 4"' if dashed else ""
        o.append(f'<path d="{d}" fill="none" stroke="#555" stroke-width="1.4" marker-end="url(#arr)"{da}/>')
        if label:
            tw = 6.2 * len(label) + 8
            o.append(f'<rect x="{lx - tw / 2}" y="{ly - 9}" width="{tw}" height="16" fill="#ffffff" opacity="0.9"/>')
            o.append(f'<text x="{lx}" y="{ly + 3}" text-anchor="middle" font-family="{F}" font-size="11" fill="#333">{html.escape(label)}</text>')
    o.append("</svg>")
    return "\n".join(o)

xml = drawio_xml()
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "architecture.drawio").write_text(xml)
(OUT / "architecture.svg").write_text(svg(xml))
print("ok")
