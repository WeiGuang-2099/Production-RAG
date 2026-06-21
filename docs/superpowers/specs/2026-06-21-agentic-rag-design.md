# Agentic RAG — Design

**Status:** approved (design phase)
**Date:** 2026-06-21
**Scope:** Subsystem #1 of 4 (Agentic RAG → model routing → guardrails → MCP server).
This spec covers Agentic RAG only.

## Goal

Upgrade the fixed `retrieve → rerank → generate` flow into an agent that **routes,
self-grades, and self-corrects**. The core pattern is Corrective-RAG (CRAG): when the
retrieved context is judged insufficient, the agent rewrites the query and retrieves again
instead of generating from weak context. Implemented as a LangGraph `StateGraph`.

The agent is exposed as a **new** `POST /agent` (+ `/agent/stream`) endpoint. The existing
`/chat` is kept as the simple, fast, single-pass path, so the two can be demoed and A/B-compared.

## Control flow

```
START → route ─┬→ answer_directly → END      (greeting / general question, no retrieval)
               ├→ clarify          → END      (too vague — ask a clarifying question)
               └→ retrieve → grade ─┬→ generate → END        (relevant → grounded, cited answer)
                     ↑              └→ rewrite → retrieve     (weak → rewrite query, loop ≤ N)
                                         (after N rewrites still weak → generate; the grounded
                                          prompt refuses if context is genuinely insufficient)
```

## State

`AgentState` (`app/agent/state.py`), a `TypedDict(total=False)`:

| field | type | meaning |
|---|---|---|
| `question` | str | original user question (never mutated) |
| `query` | str | current search query (starts = question; rewritten by `rewrite`) |
| `top_k` | int | retrieval depth |
| `route` | str | `"retrieve"` \| `"answer"` \| `"clarify"` |
| `documents` | list[Document] | reranked docs from the latest retrieve |
| `relevant` | bool | grader verdict on `documents` vs `question` |
| `attempts` | int | rewrite count (loop guard) |
| `answer` | str | final answer |
| `sources` | list[dict] | cited sources (`_docs_to_sources` shape) |
| `usage` | dict | token/cost from `usage_for` |

`run_agent` seeds the initial state as `{question, query=question, top_k, attempts=0}`. `query`
diverges from `question` only after a `rewrite`.

## Nodes

Each node is a function `state -> partial_state` in `app/agent/nodes.py`. All LLM access goes
through the existing `get_llm()` factory (so model routing, subsystem #2, slots in for free) and
is mocked in tests. Parsing helpers (`parse_route`, `parse_grade`) are pure and unit-tested.

- **route** → `{"route": ...}`. LLM classifies the question into `retrieve | answer | clarify`.
  Robust parse; unparseable output defaults to `retrieve`.
- **retrieve** → `{"documents": [...]}`. Reuses `app.core.pipeline._retrieve_and_rerank(state["query"], top_k, settings)` — hybrid + graph + rerank come for free.
- **grade** → `{"relevant": bool}`. LLM judges whether `documents` are sufficient to answer
  `question`. No documents ⇒ `False`. Parse yes/no; unparseable defaults to `True` (fail-open to
  generation, which can still refuse).
- **rewrite** → `{"query": ..., "attempts": attempts + 1}`. LLM rewrites the query for better
  retrieval. On LLM failure, keep the current query but still increment `attempts` (guarantees loop termination).
- **generate** → `{"answer", "sources", "usage"}`. Reuses `select_prompt(grounded)` +
  `format_context` + `get_llm` + `usage_for` + `_docs_to_sources`. Same grounded, cited, refusal-aware behavior as `/chat`.
- **answer_directly** → `{"answer", "sources": [], "usage"}`. Answers from the model without
  retrieval; sources empty and the answer is explicitly marked as not grounded in the corpus.
- **clarify** → `{"answer": <clarifying question>, "sources": [], "usage"}`.

## Edges

- entry → `route`
- conditional from `route`: `retrieve` → retrieve · `answer` → answer_directly · `clarify` → clarify
- `retrieve` → `grade`
- conditional from `grade`:
  - `relevant is True` → generate
  - `relevant is False and attempts < AGENT_MAX_REWRITES` → rewrite
  - `relevant is False and attempts >= AGENT_MAX_REWRITES` → generate
- `rewrite` → `retrieve`
- `generate`, `answer_directly`, `clarify` → `END`

## Public interface (`app/agent/graph.py`)

- `build_agent_graph()` — compiles and returns the `StateGraph`.
- `run_agent(question, top_k=None) -> dict` — sync; invokes the compiled graph and returns
  `{answer, sources, latency_ms, usage, route, attempts}` (superset of the `/chat` response shape).
- `stream_agent(question, top_k=None) -> AsyncIterator[dict]` — emits a `step` event per node
  transition (`{"event":"step","node":"route"|"retrieve"|"grade"|"rewrite"|"generate", ...}`) so the
  UI can show progress, a `sources` event, and a terminal `done` event with answer/usage.
  Token-level streaming of the final answer is a **stretch goal**, not required for v1.

## API

- `POST /agent` → `run_agent` via `asyncio.to_thread`; `AgentResponse {answer, sources, latency_ms, usage, route, attempts}`. Reuses `verify_api_key` + rate limit.
- `POST /agent/stream` → NDJSON stream of the `stream_agent` events.
- Streamlit UI gains an "Agent mode" toggle that targets `/agent/stream` and renders the step
  trace (route → retrieve → grade → rewrite → generate).

## Config

- `AGENT_MAX_REWRITES: int = 2` (validate `>= 0`); documented in `.env.example`.

## Dependencies

- Add `langgraph>=0.2.0` to `pyproject.toml`.

## Error handling

Every node wraps its LLM call with a safe default (route→`retrieve`, grade→`relevant=True`,
rewrite→keep query but `attempts++`). The agent never hard-fails a query; the worst case is a
grounded refusal. The rewrite loop is bounded by `AGENT_MAX_REWRITES`.

## Testing (TDD, matching the existing suite)

- **Unit:** each node with mocked `get_llm` and mocked `_retrieve_and_rerank`; assert the returned
  partial state. Pure parsers (`parse_route`, `parse_grade`) with varied/garbage LLM outputs.
- **Integration:** compiled graph with mocked LLM + retrieval, asserting the path taken and final
  state for: (a) relevant-first → generate; (b) weak → rewrite → relevant → generate (attempts==1);
  (c) weak forever → stops at `AGENT_MAX_REWRITES`, then generate; (d) route=answer → answer_directly,
  no retrieval; (e) route=clarify → clarify.
- **API:** `/agent` and `/agent/stream` with `run_agent`/`stream_agent` mocked.

## Trade-offs & limitations (stated honestly)

- The agent makes extra LLM calls (route + grade + possible rewrites), so it is **slower and
  costlier** than `/chat`. The existing per-query token/cost accounting quantifies exactly this.
- Router and grader are LLM-based; a fine-tuned small classifier would be cheaper (future work).
- `answer_directly` is intentionally ungrounded (marked as such), for non-corpus questions.

## Follow-up (out of scope for this spec)

- Evaluate `/agent` vs `/chat` on the 48-question set to quantify the CRAG gain on multi-hop/hard
  questions (extends the existing eval harness).
- Subsystems #2-#4 (model routing, guardrails, MCP server) build on this.
