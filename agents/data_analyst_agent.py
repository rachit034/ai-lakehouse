"""
Data analyst agent — answers natural-language questions over the AI Lakehouse
by chaining schema discovery, NL->SQL, raw SQL and vector search through the
`lakehouse` MCP server (see ../mcp_server/server.py).

Usage:
    python data_analyst_agent.py "Which product category has the highest average price?"
    python data_analyst_agent.py            # interactive REPL
"""
from __future__ import annotations

import asyncio
import sys

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query

from _lakehouse_mcp import lakehouse_mcp_config

SYSTEM_PROMPT = """You are a data analyst for the AI Lakehouse prototype (DuckDB/Iceberg + \
LanceDB, fronted by a FastAPI gateway exposed to you as MCP tools).

Tools available to you:
- get_schema / list_vector_datasets — discover what tables and vector datasets exist
- list_metrics — named business metric definitions (revenue, active_customer, etc.); get_schema
  already includes these inline, call this only to double-check a formula directly
- ask_nl — ask a question in plain English and let the gateway's LLM translate it to SQL
- run_sql — run SQL yourself when you need precise control
- vector_search — semantic search over embedded text datasets

Approach:
1. If you don't already know the schema, call get_schema (and list_vector_datasets if the \
question sounds like a semantic-search question) before answering.
2. Prefer ask_nl for straightforward questions; fall back to run_sql when you need to combine, \
aggregate, or verify results precisely.
3. Always answer in plain English with the concrete numbers/rows, not just a description of the \
query you ran. If you ran SQL, you may show it briefly.
4. If a query fails, read the error, adjust, and retry — don't give up after one attempt.
"""


def _options() -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers=lakehouse_mcp_config("data_analyst_agent", role="business"),
        tools=[],  # no built-in Claude Code tools — only the lakehouse MCP tools
        allowed_tools=[
            "mcp__lakehouse__get_schema",
            "mcp__lakehouse__list_vector_datasets",
            "mcp__lakehouse__list_metrics",
            "mcp__lakehouse__ask_nl",
            "mcp__lakehouse__run_sql",
            "mcp__lakehouse__vector_search",
        ],
        permission_mode="bypassPermissions",
    )


async def ask(question: str) -> None:
    async for message in query(prompt=question, options=_options()):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)
        elif isinstance(message, ResultMessage):
            print()
            if message.is_error:
                print(f"[error] {message.result}", file=sys.stderr)


async def repl() -> None:
    print("Lakehouse data analyst — ask a question (Ctrl-D to quit).")
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
