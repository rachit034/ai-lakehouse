"""
Ingestion agent — lands raw files into the lakehouse and optionally embeds a
text column, via the `lakehouse` MCP server (see ../mcp_server/server.py).

Usage:
    python ingestion_agent.py "Ingest sample_data/products.csv as table products \
in schema demo, then embed the description column into a dataset called products"
    python ingestion_agent.py            # interactive REPL
"""
from __future__ import annotations

import asyncio
import sys

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query

from _lakehouse_mcp import lakehouse_mcp_config

SYSTEM_PROMPT = """You are an ingestion agent for the AI Lakehouse prototype (DuckDB/Iceberg + \
LanceDB, fronted by a FastAPI gateway exposed to you as MCP tools).

Tools available to you:
- ingest_csv — load a local CSV file into an Iceberg table (creates the table if needed)
- embed_table — embed a text column from an existing Iceberg table into a LanceDB dataset
- get_schema / list_vector_datasets — check what already exists before/after ingesting

Approach:
1. Work out the target schema/table/dataset names from the user's request; default schema to \
"demo" unless told otherwise, and default a dataset name to the table name if not specified.
2. Call ingest_csv with the file path the user gave you (resolve relative paths against the \
project root if needed).
3. If the user also asked to embed a column, call embed_table after the ingest succeeds — do \
not guess the column name if it's ambiguous, ask or inspect the ingest_csv result's `columns` \
list first.
4. Report back concretely: table name, rows ingested, and (if applicable) dataset name and \
records embedded. If a step fails, read the error and retry with a fix rather than giving up.
"""


def _options() -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers=lakehouse_mcp_config("ingestion_agent", role="data_engineer"),
        tools=[],  # no built-in Claude Code tools — only the lakehouse MCP tools
        allowed_tools=[
            "mcp__lakehouse__get_schema",
            "mcp__lakehouse__list_vector_datasets",
            "mcp__lakehouse__ingest_csv",
            "mcp__lakehouse__embed_table",
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
    print("Lakehouse ingestion agent — describe an ingestion task (Ctrl-D to quit).")
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
