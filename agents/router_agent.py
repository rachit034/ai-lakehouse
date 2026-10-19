"""
Router agent — single natural-language entrypoint that decides for itself
whether a question is relational (SQL/DuckDB) or relationship-shaped (Cypher/
Neo4j), and reaches for the matching tools. Combines the toolsets of
data_analyst_agent.py and graph_agent.py behind one prompt instead of
requiring the caller to pick an agent up front.

The "router" is Claude's own tool-choice reasoning, not a separate
classifier model — deliberately. A bespoke classifier would be one more
component to keep in sync with what each surface is actually good at;
Claude already has to read the question either way, and the same model
that reliably writes correct multi-hop Cypher (see graph_agent.py's
docstring/system prompt for why Ollama couldn't) is well-placed to decide
which surface a question belongs on in the first place.

Usage:
    python router_agent.py "Which product category has the highest average price?"
    python router_agent.py "What else did customers who bought Wireless \
Noise-Cancelling Headphones also buy?"
    python router_agent.py            # interactive REPL
"""
from __future__ import annotations

import asyncio
import json
import sys

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)

from _lakehouse_mcp import lakehouse_mcp_config

SYSTEM_PROMPT = """You are the single front door for questions about the AI Lakehouse \
prototype. It has two query surfaces, and your first job on every question is deciding \
which one actually answers it — most questions belong on exactly one, not both.

SQL surface (DuckDB/Iceberg — relational, aggregate, filter, sort, join-a-few-tables):
- get_schema / list_vector_datasets — discover tables and vector datasets
- ask_nl — ask in plain English, let the gateway's LLM translate to SQL
- run_sql — run SQL yourself for precise control
- vector_search — semantic search over embedded text

Graph surface (Neo4j — relationships, multi-hop traversal, "who else / what else" \
questions, shortest-path, overlap-across-entities):
- graph_stats — check what's currently in the graph
- build_knowledge_graph — (re)build it if empty or stale
- get_graph_schema — live, verified labels/properties/relationship direction — call this \
before writing any Cypher, every time, not from memory
- graph_query — run Cypher

Both surfaces: list_metrics returns named business metric definitions (revenue, \
active_customer, etc.) with their exact SQL and Cypher expressions — get_schema and \
get_graph_schema already include these inline, so a question naming a known metric should \
get the same answer regardless of which surface you pick.

How to decide:
1. Read the question. If it asks for a total, average, count, filter, ranking, or a join \
across a small fixed set of tables — that's SQL. Use ask_nl first; fall back to run_sql \
if you need exact control or ask_nl gets it wrong.
2. If it asks about a relationship radiating outward from an entity ("also bought", \
"customers who...", "shortest path between", "which suppliers overlap") — that's the \
graph. Call get_graph_schema, then write real Cypher and run it via graph_query. Call \
graph_stats/build_knowledge_graph first only if you suspect the graph is empty or stale.
3. If a question genuinely needs both (e.g. "which high-value customers also bought \
product X" — a SQL aggregate to define "high-value", then a graph traversal for "also \
bought") — answer the SQL part first, then the graph part, and combine the results \
yourself. Say which surface answered which part if it's not obvious.
4. If you're genuinely unsure, prefer the SQL surface — it has sqlglot validation as a \
safety net; the graph surface doesn't, so guess there only when the question is clearly \
relationship-shaped.
5. Always answer in plain English with the concrete numbers/rows, not just a description \
of the query you ran. Show the SQL or Cypher briefly if it helps. If a query fails, read \
the error, adjust, and retry — don't give up after one attempt, and don't silently switch \
surfaces without saying so.
"""


def _options() -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers=lakehouse_mcp_config("router_agent"),
        tools=[],  # no built-in Claude Code tools — only the lakehouse MCP tools
        allowed_tools=[
            "mcp__lakehouse__get_schema",
            "mcp__lakehouse__list_vector_datasets",
            "mcp__lakehouse__list_metrics",
            "mcp__lakehouse__ask_nl",
            "mcp__lakehouse__run_sql",
            "mcp__lakehouse__vector_search",
            "mcp__lakehouse__graph_stats",
            "mcp__lakehouse__build_knowledge_graph",
            "mcp__lakehouse__get_graph_schema",
            "mcp__lakehouse__graph_query",
        ],
        permission_mode="bypassPermissions",
    )


async def ask(question: str) -> None:
    # Maps tool_use_id -> short tool name, so a later ToolResultBlock (ask_nl's
    # generated SQL only shows up in the *result*, not the call) can be traced
    # back to which tool produced it. Queries are collected here and printed
    # once, alongside the final answer, rather than interleaved mid-stream.
    pending_tools: dict[str, str] = {}
    queries: list[tuple[str, str]] = []  # [(surface, generated_query), ...]

    async for message in query(prompt=question, options=_options()):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)
                elif isinstance(block, ToolUseBlock):
                    name = block.name.rsplit("__", 1)[-1]
                    pending_tools[block.id] = name
                    if name == "run_sql":
                        queries.append(("SQL", block.input.get("sql", "")))
                    elif name == "graph_query":
                        queries.append(("Cypher", block.input.get("cypher", "")))
        elif isinstance(message, UserMessage):
            content = message.content if isinstance(message.content, list) else []
            for block in content:
                if not isinstance(block, ToolResultBlock):
                    continue
                name = pending_tools.get(block.tool_use_id)
                if name != "ask_nl" or not isinstance(block.content, str):
                    continue
                try:
                    result = json.loads(block.content)
                except (json.JSONDecodeError, TypeError):
                    continue
                generated_sql = result.get("generated_sql")
                if generated_sql:
                    queries.append(("SQL", generated_sql))
        elif isinstance(message, ResultMessage):
            print()
            if queries:
                print("\n---")
                for surface, generated_query in queries:
                    print(f"[{surface}]\n{generated_query}\n")
            if message.is_error:
                print(f"[error] {message.result}", file=sys.stderr)


async def repl() -> None:
    print("Lakehouse router agent — ask anything, SQL or graph (Ctrl-D to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except EOFError:
            print()
            break
        if not question:
            continue
        await ask(question)


def main() -> None:
    if len(sys.argv) > 1:
        asyncio.run(ask(" ".join(sys.argv[1:])))
    else:
        asyncio.run(repl())


if __name__ == "__main__":
    main()
