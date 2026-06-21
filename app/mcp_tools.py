"""MCP tool logic (in-process). Thin reuse of the pipeline, agent, and guardrails."""
from __future__ import annotations

from app.agent.graph import run_agent
from app.config import get_settings
from app.core.pipeline import ingest_pipeline, list_documents, retrieve_sources
from app.guardrails.service import apply_output, check_input
from app.ingestion.validation import validate_source


def mcp_search(query: str, top_k: int = 5) -> dict:
    blocked = check_input(query)
    if blocked:
        return {"error": "blocked by input guardrails", "patterns": blocked}
    return {"sources": retrieve_sources(query, top_k)}


def mcp_ask(question: str, top_k: int = 5) -> dict:
    blocked = check_input(question)
    if blocked:
        return {"error": "blocked by input guardrails", "patterns": blocked}
    result = run_agent(question, top_k)
    guarded = apply_output(result.get("answer", ""))
    result["answer"] = guarded["answer"]
    result["guardrails"] = {"pii_redacted": guarded["pii_redacted"], "flags": guarded["flags"]}
    return result


def mcp_ingest(source: str) -> dict:
    if not get_settings().MCP_ALLOW_INGEST:
        return {"error": "ingest is disabled (set MCP_ALLOW_INGEST=true)"}
    try:
        validate_source(source, get_settings())
    except ValueError as exc:
        return {"error": str(exc)}
    return ingest_pipeline(source)


def mcp_list_documents() -> list[dict]:
    return list_documents()
