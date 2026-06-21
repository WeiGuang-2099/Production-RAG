"""MCP server (stdio) exposing the RAG system to MCP clients like Claude Desktop.

Run:  rag-mcp           (after `pip install -e .`)
or:   python -m app.mcp_server
Qdrant must be running and .env configured.
"""
from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from app.mcp_tools import mcp_ask, mcp_ingest, mcp_list_documents, mcp_search

mcp = FastMCP("production-rag")


@mcp.tool()
def search(query: str, top_k: int = 5) -> dict:
    """Search the ingested corpus; returns cited snippets without generating an answer."""
    return mcp_search(query, top_k)


@mcp.tool()
def ask(question: str, top_k: int = 5) -> dict:
    """Answer a question with the corrective-RAG agent (routes, self-corrects, grounds with
    citations). Returns the answer, sources, route, and attempts."""
    return mcp_ask(question, top_k)


@mcp.tool()
def ingest(source: str) -> dict:
    """Ingest a local file (under DATA_DIR) or a URL into the corpus."""
    return mcp_ingest(source)


@mcp.tool()
def list_documents() -> list[dict]:
    """List the documents currently ingested in the corpus."""
    return mcp_list_documents()


@mcp.resource("rag://documents")
def documents_resource() -> str:
    """The ingested document list as readable JSON."""
    return json.dumps(mcp_list_documents(), indent=2)


@mcp.prompt()
def grounded_research(topic: str) -> str:
    """A prompt template guiding the client to research a topic using the tools."""
    return (
        f"Use the `search` and `ask` tools to research '{topic}' in the corpus. "
        "Answer with inline [n] citations, and if the corpus does not cover it, say so explicitly."
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
