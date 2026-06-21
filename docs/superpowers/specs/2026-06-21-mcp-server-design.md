# MCP Server — Design

**Status:** approved (design phase)
**Date:** 2026-06-21
**Scope:** Subsystem #4 of 4 (Agentic RAG → model routing → guardrails → **MCP server**).
**Depends on:** #3 (guardrails `service`) — the tools reuse `check_input` / `apply_output`, so
implement #4 after #3.

## Goal

Expose the RAG system as an MCP server so any MCP client (e.g. Claude Desktop) can search and
query the private corpus. Built on the official MCP Python SDK (`FastMCP`) as a thin, in-process
adapter over the existing pipeline/agent functions. The server demonstrates all three MCP
primitives: **tools**, a **resource**, and a **prompt**.

## Architecture (in-process)

- `app/mcp_tools.py` — tool **logic** as plain functions (unit-tested), reusing
  `retrieve_sources`, `run_agent`, `ingest_pipeline`, `list_documents`, the guardrails `service`,
  and the extracted `validate_source`.
- `app/mcp_server.py` — `FastMCP` wiring: register the tool functions, one resource, one prompt;
  `main()` runs over **stdio**.

Separating logic from protocol means the logic layer is fully testable and the protocol layer
stays a thin shell. Because stdio servers do not pass through FastAPI's auth/rate-limit/guardrails,
the tools reuse the guardrail `service` explicitly.

## Tools / resource / prompt

| Kind | Name | Reuses | Behavior |
|---|---|---|---|
| tool | `search(query, top_k=5)` | `retrieve_sources` | cited snippets, no generation; input via `check_input` |
| tool | `ask(question, top_k=5)` | `run_agent` | corrective-RAG agent answer; input `check_input`, output `apply_output` (PII redact + toxicity flag) |
| tool | `ingest(source)` | `ingest_pipeline` | gated by `MCP_ALLOW_INGEST`; `validate_source` first (path-traversal safe) |
| tool | `list_documents()` | `list_documents` | list ingested documents |
| resource | `rag://documents` | `list_documents` | the document list as readable JSON context |
| prompt | `grounded_research(topic)` | — | a reusable prompt template that guides the client to use search+ask with citations |

`ask` is named simply for client UX; its docstring states it is agent-backed (route + self-correct),
and it returns `route` / `attempts` so the agent behavior is visible.

## Reuse refactors (small, necessary)

- Extract the ingest source validation from `app/api/routes_ingest.py` into
  `app/ingestion/validation.py::validate_source(source: str, settings) -> str` (raises `ValueError`).
  `routes_ingest` calls `validate_source(v, get_settings())` (so its existing tests, which patch
  `app.api.routes_ingest.get_settings`, keep working); `mcp_ingest` calls it too. **Without this,
  the MCP ingest tool would bypass path-traversal protection.**
- Add `app/core/pipeline.py::list_documents() -> list[dict]` (reads `ingestions.json` via the
  existing `_load_tracking`), shared by the tool and the resource.

## Config / dependencies

- `MCP_ALLOW_INGEST: bool = True` (gate the write tool; flip to `false` for a read-only server).
- New dependency: `mcp>=1.2.0`.
- Console script: `[project.scripts] rag-mcp = "app.mcp_server:main"`.

## Claude Desktop integration (README)

```json
{
  "mcpServers": {
    "production-rag": { "command": "rag-mcp" }
  }
}
```
(after `pip install -e .`; Qdrant must be running and `.env` configured). Add a screenshot of
Claude Desktop calling the `ask` tool.

## Error handling

Tool functions never raise to the protocol layer: input-blocked / invalid-source / disabled-ingest
return `{"error": ...}` dicts the client can read. `validate_source` raises `ValueError`, which
`mcp_ingest` catches and converts to `{"error": ...}`.

## Testing (TDD)

- **`app/mcp_tools.py`** — unit tests with mocked `retrieve_sources` / `run_agent` /
  `ingest_pipeline` / `list_documents` / `check_input` / `apply_output` / `validate_source`:
  search returns sources; injection → error; `ask` redacts output; ingest disabled → error;
  ingest validates then calls the pipeline; invalid source → error; list returns docs.
- **`validate_source`** — URL passthrough, path-outside-DATA_DIR rejected, missing file, bad
  extension, oversize (mirror the existing route tests).
- **`list_documents`** — reads a temp `ingestions.json`.
- **`app/mcp_server.py`** — import smoke test (`srv.mcp` exists) once `mcp` is installed.

## Trade-offs & limitations (stated honestly)

- **stdio only** — remote SSE/HTTP transport is a follow-up.
- `ingest` is a write operation; mitigated by the `MCP_ALLOW_INGEST` gate + path validation, but a
  read-only deployment should set it `false`.
- The server runs in-process and needs Qdrant + a configured `.env`, same as the API.

## Follow-up (out of scope)

- Remote transport (SSE/HTTP) + auth for non-local clients.
- This is the last of the four planned integrations.
