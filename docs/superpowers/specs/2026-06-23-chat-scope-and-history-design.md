# Design: Chat persistence, per-file scoping, and clean source names

Date: 2026-06-23
Status: Approved (design); pending spec review

## Problem

Three issues surfaced while testing the React demo UI against the running backend:

1. **Chat history is lost when switching pages.** `useChat` holds `messages` in local
   `useState` inside `ChatPage`. Navigating to `/documents` or `/about` unmounts the page and
   destroys the conversation.
2. **No way to answer from a specific file.** Retrieval always runs across every ingested
   document. With many docs of very different sizes (e.g. `gpt3.pdf` = 178 chunks vs a 4-chunk
   quiz), a small target file gets drowned out, and users cannot say "answer only from this file."
3. **Source labels are confusing.** `SourceCards` renders the raw `metadata.source`
   (`./data/papers/gpt3.pdf`, `data\Quiz_ Practice Exam.pdf`, or `graph`). Users want a readable
   filename, not a path or an internal id.

## Goals

- Chat conversation and chat-tab controls survive page navigation **and** a full browser reload.
- Users can scope a question to one or more ingested documents via a picker in the chat UI; the
  scope filters real retrieval (not just the displayed source list).
- Sources display a clean, human-readable filename, with the raw value still available on hover.

## Non-goals

- No changes to ingestion, auth, rate limiting, or guardrails.
- Not fixing the separate `cs_code.pdf` "0 chunks" extraction issue.
- No new state-management dependency; no per-document Qdrant collections.

## Chosen approaches (with rejected alternatives)

**Feature 1 — persistence:** React Context provider + `localStorage`.
Rejected: in-memory-only context (does not survive reload); a state library such as Zustand
(new dependency for a single screen).

**Feature 2 — file scoping:** a real retrieval filter on `metadata.source`, threaded through the
single shared `_retrieve_and_rerank` function, surfaced as a multi-select picker in chat.
Rejected: display-only filtering of the returned sources (does not change the answer); a separate
Qdrant collection per document (heavy, breaks cross-document search).

**Feature 3 — clean names:** a pure `displaySource()` helper used wherever a source is shown.

---

## Feature 1: Chat history survives navigation and reload

### Components

- `src/context/ChatContext.tsx` — new provider that owns the chat state currently living in
  `useChat` and the chat-tab controls (`agent`, `stream`, `topK`, and the new `scopeSources`).
  Exposes `{ messages, busy, send, clear, agent, stream, topK, scopeSources, setAgent,
  setStream, setTopK, setScopeSources }`.
- `App.tsx` — wrap `<Routes>` in `<ChatProvider>` so the chat state is mounted above the router
  and is not unmounted on navigation.
- `ChatPage.tsx` / `ChatControls.tsx` — consume the context instead of local `useState`.

### Data flow / persistence

- On provider mount, hydrate `messages` and controls from `localStorage` key `rag-chat`.
- On change, write back to `localStorage`. A size guard keeps only the most recent ~50 messages
  to bound storage growth.
- A **"New chat"** control (in `ChatControls`) calls `clear()`, which empties `messages` and the
  `localStorage` entry.

### Error handling

- `localStorage` reads/writes are wrapped in try/catch; on failure the app continues with
  in-memory state (degrade gracefully, never throw on hydrate).

---

## Feature 2: Answer from specific file(s)

### Backend

Single shared retrieval path means one filter threads everywhere.

- `_retrieve_and_rerank(question, top_k, settings, sources: list[str] | None = None)`:
  - **Vector:** `VectorStore.search(query, top_k, sources=None)` builds a `qdrant_client.models.Filter`
    on the source payload field (`MatchValue` for one, `MatchAny` for several) and passes it as the
    `filter` argument to `similarity_search_with_score`. The exact payload key (expected
    `metadata.source`) is verified against `langchain_qdrant` during implementation with a small
    live query before wiring the rest.
  - **BM25:** `BM25Store.search(query, top_k, sources=None)` filters candidate indices to docs whose
    `metadata.source` is in `sources` **before** taking top_k, preserving recall.
  - **Graph:** when `sources` is set, **skip graph expansion** (graph retrieval is cross-document and
    would leak other files into a scoped answer).
- Thread `sources` through `query_pipeline`, `stream_query`, `retrieve_sources`, and the agent
  (`run_agent`, `stream_agent`, agent state, and `retrieve_node`).
- **Cache:** when `sources` is set, **bypass** the semantic query cache (a scoped query must not
  return an unscoped cached answer, and vice versa).

### API

- Add `sources: list[str] | None = None` to `ChatRequest` (`routes_chat.py`) and `AgentRequest`
  (`routes_agent.py`). An empty or omitted list behaves as "all documents." Unknown source strings
  simply match nothing (empty scoped result), no error.

### Frontend

- New `src/components/chat/DocumentScopePicker.tsx` — multi-select chips populated from
  `useDocuments`. Default = nothing selected = "All documents." Selected `DocumentRecord.source`
  strings are stored in context as `scopeSources`.
- `useChat.send` opts gain `sources: string[]`; the request body includes `sources` only when
  non-empty.
- Chip labels use `displaySource()` (Feature 3).

### Rationale

Multi-select is chosen over single-file: backend cost is identical (one `MatchAny` filter), and it
lets a user compare across two documents. Defaulting to "all" preserves current behavior for users
who ignore the picker.

---

## Feature 3: Clean source filenames

- `src/api/sources.ts` → `displaySource(raw: string): string`:
  - `graph` -> `"Knowledge graph"`
  - filesystem path (`./data/papers/gpt3.pdf`, `data\Quiz_ Practice Exam.pdf`) -> basename
    (`gpt3.pdf`, `Quiz_ Practice Exam.pdf`), splitting on both `/` and `\`.
  - URL (`http...`) -> host + last path segment.
  - anything else -> returned unchanged.
- Used in `SourceCards.tsx` (raw value kept as the element `title` tooltip) and in the picker chips.

---

## Testing

### Backend
- `VectorStore.search` builds the expected Qdrant filter for one and several sources.
- `BM25Store.search` restricts results to the selected sources and still returns top_k by score.
- `_retrieve_and_rerank` skips graph when `sources` is set; threads `sources` to vector + BM25.
- `query_pipeline` / `stream_query` bypass the cache when `sources` is set.
- API: `/chat` and `/agent` accept `sources`, reject malformed types, and pass them through.

### Frontend
- `displaySource` unit tests for path, Windows path, URL, `graph`, and passthrough cases.
- `ChatContext` persistence: hydrate from `localStorage`, write on change, `clear()` empties both.
- Picker selection produces the expected `sources` array in the request body.
- Existing `useChat` / `ChatPage` tests updated for the context provider.

## Rollout / risk

- The Qdrant payload key assumption is the main backend risk; verified empirically before building
  the rest of the filter path.
- Persistence is additive and degrades to in-memory on `localStorage` failure.
- Default behavior (no scope selected, cache and graph active) is unchanged, so existing eval and
  tests remain valid.
