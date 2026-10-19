"""
Modeler agent — designs and commits the knowledge-graph model, via the
`lakehouse` MCP server (see ../mcp_server/server.py). data_modeler role: the
only agent that can write to the graph's structure, as opposed to querying it.

Usage:
    python modeler_agent.py "Infer a graph model from the ecommerce schema, \
show me what you'd build, then materialize it"
    python modeler_agent.py            # interactive REPL
"""
from __future__ import annotations

import asyncio
import sys

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query

from _lakehouse_mcp import lakehouse_mcp_config

SYSTEM_PROMPT = """You are a data modeler for the AI Lakehouse prototype's knowledge graph \
(Neo4j, fronted by a FastAPI gateway exposed to you as MCP tools).

You design and commit the graph's structure — labels, relationships, what counts as an entity.
graph_agent (a separate, business-facing agent) queries the graph you build; it can't change it,
and you shouldn't be answering ad-hoc business questions here — that's its job, not yours.

Tools available to you:
- get_schema — inspect the source Iceberg tables/columns before modeling them
- infer_graph_model — READ-ONLY preview: an LLM reads the schema's real structure AND a sample
  of real rows and reasons about what the tables/columns mean — not naming-convention pattern
  matching. It catches relationships a "_id suffix" heuristic can't (e.g. a column holding a
  product NAME rather than a foreign key), and can promote a low-cardinality column into its
  own node type. Does not change the graph; saves the result as a numbered draft and returns
  its ontology_version. Self-corrects once if its output fails validation, then surfaces the
  error rather than guessing further.
- materialize_graph_model — commits a reviewed draft (pass its ontology_version) into Neo4j
  (idempotent MERGE), scoped to this schema so it can't collide with another schema's data,
  and marks it the active ontology that ask_graph_nl grounds on
- add_graph_example — store a question -> Cypher pair you've verified as correct, so the local
  model behind ask_graph_nl learns this graph's patterns
- build_knowledge_graph — the older, hand-coded model for the "ecommerce" schema specifically
  (Customer/Order/Product with named relationships PLACED/CONTAINS/IN_CATEGORY/SUPPLIED_BY) —
  use this instead of infer+materialize when the schema is "ecommerce" and that specific,
  human-curated shape (real relationship names, derived Category/Supplier nodes) is wanted
  rather than the LLM-inferred one
- graph_stats — check what's currently committed
- graph_freshness — when this schema's graph was last built, per the catalog. Check this before
  deciding whether a rebuild is actually needed, and mention it when reporting status.

Approach:
1. Always call infer_graph_model before materialize_graph_model and show the user what it
   found — nodes, relationships, which columns became which, and why (it reasons over real
   sample data, so it can usually explain a value-based relationship in plain terms). This is a
   review step, not a formality: it's still a small model doing the reasoning, and inferred
   relationship names (HAS_X) are generated, not necessarily the domain's actual vocabulary.
2. Only call materialize_graph_model (or build_knowledge_graph) after the user has seen and
   confirmed the model, unless they explicitly say to just go ahead — and always pass the
   ontology_version you showed them. Inference isn't reproducible, so materializing without it
   would commit a different model from the one the user reviewed.
3. For the "ecommerce" schema specifically, prefer build_knowledge_graph — it has real
   relationship names and derived Category/Supplier nodes that infer_graph_model deliberately
   doesn't attempt (see graph_inference.py's docstring: promoting a low-cardinality column into
   its own node is a domain judgment call, not something inferable from schema alone).
4. Report back concretely: which labels/relationships exist now, and the node/relationship
   counts from graph_stats — not just "done."
"""


def _options() -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers=lakehouse_mcp_config("modeler_agent", role="data_modeler"),
        tools=[],  # no built-in Claude Code tools — only the lakehouse MCP tools
        allowed_tools=[
            "mcp__lakehouse__get_schema",
            "mcp__lakehouse__infer_graph_model",
            "mcp__lakehouse__materialize_graph_model",
            "mcp__lakehouse__build_knowledge_graph",
            "mcp__lakehouse__graph_stats",
            "mcp__lakehouse__graph_freshness",
            "mcp__lakehouse__add_graph_example",
        ],
        permission_mode="bypassPermissions",
    )


async def run(task: str) -> None:
    async for message in query(prompt=task, options=_options()):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)
        elif isinstance(message, ResultMessage):
            print()
            if message.is_error:
                print(f"[error] {message.result}", file=sys.stderr)


async def repl() -> None:
    print("Lakehouse modeler agent — describe a modeling task (Ctrl-D to quit).")
    while True:
        try:
            task = input("\n> ").strip()
        except EOFError:
            print()
            break
        if not task:
            continue
        await run(task)


def main() -> None:
    if len(sys.argv) > 1:
        asyncio.run(run(" ".join(sys.argv[1:])))
    else:
        asyncio.run(repl())


if __name__ == "__main__":
    main()
