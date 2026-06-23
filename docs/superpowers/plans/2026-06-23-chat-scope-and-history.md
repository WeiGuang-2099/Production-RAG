# Chat Persistence, Per-File Scoping, and Clean Source Names — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the demo UI keep its chat across navigation/reload, scope answers to chosen documents, and show readable source filenames.

**Architecture:** Backend threads an optional `sources` filter through the single shared `_retrieve_and_rerank` path (vector + BM25 filtered by `metadata.source`, graph skipped when scoped, cache bypassed when scoped). Frontend lifts chat state into a `ChatProvider` (mounted above the router) with `localStorage` persistence, adds a multi-select document picker, and renders clean filenames via a pure helper.

**Tech Stack:** FastAPI + Pydantic, LangChain `Document`, `langchain_qdrant` / `qdrant_client`, `rank_bm25`; React 18 + TypeScript + Vite, Vitest + Testing Library, Tailwind, framer-motion.

## Global Constraints

- Run all Python tests/linters via the venv interpreter: `./.venv/Scripts/python.exe -m pytest ...` (system Python lacks deps).
- Run frontend tests from `frontend/`: `npx vitest run <path>`.
- Commit messages: plain text, no emojis, **no AI attribution / co-author trailers**. Do not `git push` (the user pushes manually).
- Naming is fixed across the stack: API request field and all Python pipeline params are `sources: list[str] | None`. Inside the agent, the scope filter is carried as `scope_sources` because `AgentState` already uses `sources: list[dict]` for output citations. Frontend context state is `scopeSources: string[]`; the request body field it serializes to is `sources`.
- Default behavior must be unchanged when no scope is selected (no `sources`, or empty list): cache and graph stay active, retrieval spans all documents. Existing tests and eval must remain valid.
- TDD: write the failing test first, watch it fail, implement minimally, watch it pass, commit.

---

### Task 1: Clean source-name helper (`displaySource`) + use in SourceCards

**Files:**
- Create: `frontend/src/api/sources.ts`
- Create: `frontend/src/api/sources.test.ts`
- Modify: `frontend/src/components/chat/SourceCards.tsx`

**Interfaces:**
- Produces: `displaySource(raw: string): string` — maps a raw source value to a human label.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/api/sources.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { displaySource } from "./sources";

describe("displaySource", () => {
  it("maps the graph sentinel to a readable label", () => {
    expect(displaySource("graph")).toBe("Knowledge graph");
  });
  it("returns the basename of a posix path", () => {
    expect(displaySource("./data/papers/gpt3.pdf")).toBe("gpt3.pdf");
  });
  it("returns the basename of a windows path, keeping spaces", () => {
    expect(displaySource("data\\Quiz_ Practice Exam.pdf")).toBe("Quiz_ Practice Exam.pdf");
  });
  it("shows host and last segment for a URL", () => {
    expect(displaySource("https://example.com/docs/intro.html")).toBe("example.com/intro.html");
  });
  it("passes through an unknown bare value", () => {
    expect(displaySource("unknown")).toBe("unknown");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/api/sources.test.ts`
Expected: FAIL — `Failed to resolve import "./sources"`.

- [ ] **Step 3: Write minimal implementation**

Create `frontend/src/api/sources.ts`:

```ts
// Maps a raw retrieval source value (path, URL, or the "graph" sentinel)
// to a short, human-readable label for the UI. Pure and dependency-free.
export function displaySource(raw: string): string {
  if (!raw) return "unknown";
  if (raw === "graph") return "Knowledge graph";
  if (/^https?:\/\//i.test(raw)) {
    try {
      const u = new URL(raw);
      const last = u.pathname.split("/").filter(Boolean).pop();
      return last ? `${u.host}/${last}` : u.host;
    } catch {
      return raw;
    }
  }
  const parts = raw.split(/[/\\]/).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : raw;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/api/sources.test.ts`
Expected: PASS (5 tests).

- [ ] **Step 5: Use the helper in SourceCards**

In `frontend/src/components/chat/SourceCards.tsx`, add the import at the top:

```ts
import { displaySource } from "../../api/sources";
```

Replace the source `<span>` (currently line 28) with a labelled, tooltipped version:

```tsx
                    <span className="truncate" title={s.metadata?.source ?? "unknown"}>
                      {displaySource(s.metadata?.source ?? "unknown")}
                    </span>
```

- [ ] **Step 6: Run the SourceCards tests**

Run: `cd frontend && npx vitest run src/components/chat/SourceCards.test.tsx`
Expected: PASS. If an existing assertion checks for the raw path text, update it to the basename (e.g. `gpt3.pdf`).

- [ ] **Step 7: Commit**

```bash
git add frontend/src/api/sources.ts frontend/src/api/sources.test.ts frontend/src/components/chat/SourceCards.tsx
git commit -m "feat(ui): show clean source filenames in source cards"
```

---

### Task 2: BM25 source filter

**Files:**
- Modify: `app/retrieval/bm25_store.py:41-47`
- Test: `tests/test_bm25_store.py`

**Interfaces:**
- Produces: `BM25Store.search(query: str, top_k: int = 5, sources: list[str] | None = None) -> list[tuple[Document, float]]` — when `sources` is non-empty, only docs whose `metadata["source"]` is in `sources` are eligible, ranked before the top_k cut.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bm25_store.py`:

```python
def test_search_filters_by_source(tmp_path):
    store = BM25Store(data_dir=str(tmp_path))
    docs = [
        Document(page_content="Machine learning is a subset of artificial intelligence.", metadata={"source": "a"}),
        Document(page_content="Machine learning powers modern search.", metadata={"source": "b"}),
        Document(page_content="Neural networks are used in deep learning.", metadata={"source": "c"}),
    ]
    store.add_documents(docs)

    results = store.search("machine learning", top_k=10, sources=["b"])
    assert len(results) == 1
    assert results[0][0].metadata["source"] == "b"


def test_search_empty_sources_means_all(tmp_path):
    store = BM25Store(data_dir=str(tmp_path))
    docs = [
        Document(page_content="Machine learning is a subset of artificial intelligence.", metadata={"source": "a"}),
        Document(page_content="Machine learning powers modern search.", metadata={"source": "b"}),
    ]
    store.add_documents(docs)

    assert len(store.search("machine learning", top_k=10, sources=[])) >= 2
    assert len(store.search("machine learning", top_k=10)) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_bm25_store.py::test_search_filters_by_source -v`
Expected: FAIL — `search() got an unexpected keyword argument 'sources'`.

- [ ] **Step 3: Write minimal implementation**

Replace `BM25Store.search` (lines 41-47) with:

```python
    def search(
        self, query: str, top_k: int = 5, sources: list[str] | None = None
    ) -> list[tuple[Document, float]]:
        if self._bm25 is None or not self._documents:
            return []
        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        indices = range(len(scores))
        if sources:
            allowed = set(sources)
            indices = [i for i in indices if self._documents[i].metadata.get("source") in allowed]
        top_indices = sorted(indices, key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._documents[i], float(scores[i])) for i in top_indices if scores[i] > 0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_bm25_store.py -v`
Expected: PASS (all, including the two new tests).

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/bm25_store.py tests/test_bm25_store.py
git commit -m "feat(retrieval): filter BM25 results by source"
```

---

### Task 3: Vector store source filter

**Files:**
- Modify: `app/retrieval/vector_store.py:54-56`
- Test: `tests/test_vector_store.py`

**Interfaces:**
- Produces: `VectorStore.search(query: str, top_k: int = 5, sources: list[str] | None = None) -> list[tuple[Document, float]]` — when `sources` is non-empty, passes a Qdrant payload filter on `metadata.source` (`MatchAny`) to `similarity_search_with_score`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_vector_store.py`:

```python
def test_search_passes_source_filter(mock_qdrant):
    mock_instance, _ = mock_qdrant
    mock_instance.similarity_search_with_score.return_value = []

    with patch("app.retrieval.vector_store.get_embedder") as mock_emb, \
         patch("app.retrieval.vector_store.get_settings") as mock_s:
        mock_emb.return_value = MagicMock()
        mock_s.return_value.QDRANT_URL = "http://localhost:6333"
        mock_s.return_value.COLLECTION_NAME = "test_col"
        mock_s.return_value.QDRANT_API_KEY = None

        vs = VectorStore()
        vs.search("q", top_k=5, sources=["./data/a.pdf", "./data/b.pdf"])

        _, kwargs = mock_instance.similarity_search_with_score.call_args
        flt = kwargs["filter"]
        cond = flt.must[0]
        assert cond.key == "metadata.source"
        assert set(cond.match.any) == {"./data/a.pdf", "./data/b.pdf"}


def test_search_without_sources_sends_no_filter(mock_qdrant):
    mock_instance, _ = mock_qdrant
    mock_instance.similarity_search_with_score.return_value = []

    with patch("app.retrieval.vector_store.get_embedder") as mock_emb, \
         patch("app.retrieval.vector_store.get_settings") as mock_s:
        mock_emb.return_value = MagicMock()
        mock_s.return_value.QDRANT_URL = "http://localhost:6333"
        mock_s.return_value.COLLECTION_NAME = "test_col"
        mock_s.return_value.QDRANT_API_KEY = None

        vs = VectorStore()
        vs.search("q", top_k=5)

        _, kwargs = mock_instance.similarity_search_with_score.call_args
        assert kwargs.get("filter") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_vector_store.py::test_search_passes_source_filter -v`
Expected: FAIL — `search() got an unexpected keyword argument 'sources'`.

- [ ] **Step 3: Write minimal implementation**

In `app/retrieval/vector_store.py`, add to the imports near the top:

```python
from qdrant_client import models as qmodels
```

Replace `search` (lines 54-56) with:

```python
    def search(
        self, query: str, top_k: int = 5, sources: list[str] | None = None
    ) -> list[tuple[Document, float]]:
        store = self._get_store()
        flt = None
        if sources:
            flt = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="metadata.source",
                        match=qmodels.MatchAny(any=list(sources)),
                    )
                ]
            )
        return store.similarity_search_with_score(query, k=top_k, filter=flt)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_vector_store.py -v`
Expected: PASS (all, including the two new tests).

- [ ] **Step 5: Verify the Qdrant payload key against the live index**

Run (backend + Qdrant must be up):

```bash
./.venv/Scripts/python.exe -c "from app.retrieval.vector_store import VectorStore; vs=VectorStore(); r=vs.search('exam', top_k=3, sources=['data\\\\Quiz_ Practice Exam.pdf']); print(len(r)); [print(d.metadata.get('source')) for d,_ in r]"
```

Expected: prints `1` or more rows, and every printed source equals `data\Quiz_ Practice Exam.pdf`. If it prints `0` while the unscoped query returns hits, the payload key differs from `metadata.source` — inspect one point's payload via `QdrantClient.scroll` and adjust the `key=` string, then re-run Step 4.

- [ ] **Step 6: Commit**

```bash
git add app/retrieval/vector_store.py tests/test_vector_store.py
git commit -m "feat(retrieval): filter vector search by source"
```

---

### Task 4: Thread `sources` through hybrid retriever, retrieval/generation pipeline, cache, and graph-skip

**Files:**
- Modify: `app/retrieval/hybrid_retriever.py:37-41`
- Modify: `app/core/pipeline.py` (`_retrieve_and_rerank`, `retrieve_sources`, `query_pipeline`, `stream_query`)
- Test: `tests/test_pipeline.py`, `tests/test_hybrid_retriever.py`

**Interfaces:**
- Consumes: `VectorStore.search(..., sources=)`, `BM25Store.search(..., sources=)` from Tasks 2-3.
- Produces:
  - `HybridRetriever.retrieve(query, top_k=5, sources: list[str] | None = None)`
  - `_retrieve_and_rerank(question, top_k, settings, sources: list[str] | None = None)`
  - `retrieve_sources(question, top_k=None, sources: list[str] | None = None)`
  - `query_pipeline(question, top_k=None, sources: list[str] | None = None)`
  - `stream_query(question, top_k=None, sources: list[str] | None = None)`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hybrid_retriever.py`:

```python
def test_retrieve_passes_sources_to_both_stores():
    from unittest.mock import MagicMock
    from app.retrieval.hybrid_retriever import HybridRetriever

    vs, bm25 = MagicMock(), MagicMock()
    vs.search.return_value = []
    bm25.search.return_value = []
    HybridRetriever(vector_store=vs, bm25_store=bm25).retrieve("q", top_k=5, sources=["x"])

    assert vs.search.call_args.kwargs["sources"] == ["x"]
    assert bm25.search.call_args.kwargs["sources"] == ["x"]
```

Append to `tests/test_pipeline.py`:

```python
def test_retrieve_and_rerank_skips_graph_when_scoped(monkeypatch):
    import app.core.pipeline as pipe
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "RETRIEVAL_MODE", "dense", raising=False)
    monkeypatch.setattr(settings, "GRAPH_EXTRACTOR", "nlp", raising=False)

    class _VS:
        def search(self, q, top_k=5, sources=None):
            return []
    monkeypatch.setattr(pipe, "VectorStore", lambda: _VS())

    called = {"graph": False}

    class _GR:
        def __init__(self, *a, **k): pass
        def retrieve(self, *a, **k):
            called["graph"] = True
            return []
    monkeypatch.setattr(pipe, "GraphRetriever", _GR)
    monkeypatch.setattr(pipe, "GraphStore", lambda **k: object())
    monkeypatch.setattr(pipe, "RerankerService", lambda reranker: type("R", (), {"rerank": lambda self, q, d, top_k: d})())
    monkeypatch.setattr(pipe, "get_reranker", lambda: object())

    pipe._retrieve_and_rerank("q", 5, settings, sources=["only.pdf"])
    assert called["graph"] is False


def test_query_pipeline_bypasses_cache_when_scoped(monkeypatch):
    import app.core.pipeline as pipe

    monkeypatch.setattr(pipe, "_retrieve_and_rerank", lambda *a, **k: [])
    monkeypatch.setattr(pipe, "format_context", lambda docs: "")
    monkeypatch.setattr(pipe, "complete_with_model", lambda prompt: ("answer", "model"))

    cache = type("C", (), {"get": lambda self, q: {"answer": "CACHED", "sources": []}, "put": lambda self, q, r: None})()
    monkeypatch.setattr(pipe, "_get_query_cache", lambda settings: cache)

    out = pipe.query_pipeline("q", top_k=3, sources=["only.pdf"])
    assert out["answer"] == "answer"  # not "CACHED": scoped query must skip the cache
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
./.venv/Scripts/python.exe -m pytest tests/test_hybrid_retriever.py::test_retrieve_passes_sources_to_both_stores tests/test_pipeline.py::test_retrieve_and_rerank_skips_graph_when_scoped tests/test_pipeline.py::test_query_pipeline_bypasses_cache_when_scoped -v
```
Expected: FAIL — unexpected keyword argument `sources`.

- [ ] **Step 3: Update HybridRetriever**

Replace `HybridRetriever.retrieve` (lines 37-41) in `app/retrieval/hybrid_retriever.py`:

```python
    def retrieve(
        self, query: str, top_k: int = 5, sources: list[str] | None = None
    ) -> list[tuple[Document, float]]:
        vector_results = self.vector_store.search(query, top_k=top_k, sources=sources)
        bm25_results = self.bm25_store.search(query, top_k=top_k, sources=sources)
        fused = rrf_fuse([vector_results, bm25_results])
        return fused[:top_k]
```

- [ ] **Step 4: Thread `sources` through the pipeline**

In `app/core/pipeline.py`, change the signature of `_retrieve_and_rerank` (line 149):

```python
def _retrieve_and_rerank(
    question: str, top_k: int, settings, sources: list[str] | None = None
) -> list[Document]:
```

Inside it, pass `sources` to the dense and hybrid branches (lines 169-172):

```python
            if settings.RETRIEVAL_MODE == "dense":
                result_lists.append(vs.search(q, top_k=top_k, sources=sources))
            else:
                result_lists.append(retriever.retrieve(q, top_k=top_k, sources=sources))
```

Guard the graph block so it is skipped when scoped (line 187):

```python
    if settings.GRAPH_EXTRACTOR != "none" and not sources:
```

Update `retrieve_sources` (line 223) to accept and forward `sources`:

```python
def retrieve_sources(
    question: str, top_k: int | None = None, sources: list[str] | None = None
) -> list[dict]:
```
and its call (line 233):
```python
    reranked = _retrieve_and_rerank(question, top_k, settings, sources=sources)
```

Update `query_pipeline` (line 259) to accept `sources`, bypass cache when scoped, and forward:

```python
def query_pipeline(
    question: str, top_k: int | None = None, sources: list[str] | None = None
) -> dict:
```
Change the cache read (lines 266-271) to:
```python
    cache = _get_query_cache(settings)
    if cache is not None and not sources:
        cached = cache.get(question)
        if cached is not None:
            logger.info("cache_hit: question=%s", question[:100])
            return {**cached, "cached": True}
```
Change the retrieval call (line 273):
```python
    reranked = _retrieve_and_rerank(question, top_k, settings, sources=sources)
```
Change the cache write (lines 310-311):
```python
    if cache is not None and not sources:
        cache.put(question, result)
```

Update `stream_query` (line 315) to accept `sources` and forward:

```python
async def stream_query(
    question: str, top_k: int | None = None, sources: list[str] | None = None
) -> AsyncIterator[dict]:
```
Change the retrieval call (line 328):
```python
    reranked = await asyncio.to_thread(_retrieve_and_rerank, question, top_k, settings, sources)
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```
./.venv/Scripts/python.exe -m pytest tests/test_hybrid_retriever.py tests/test_pipeline.py -v
```
Expected: PASS (existing + 3 new).

- [ ] **Step 6: Commit**

```bash
git add app/retrieval/hybrid_retriever.py app/core/pipeline.py tests/test_hybrid_retriever.py tests/test_pipeline.py
git commit -m "feat(retrieval): thread source scope through pipeline, skip graph and cache when scoped"
```

---

### Task 5: Thread `sources` through the agent

**Files:**
- Modify: `app/agent/state.py:6-17`
- Modify: `app/agent/nodes.py:33-37`
- Modify: `app/agent/graph.py` (`run_agent` line 63-66, `stream_agent` line 78-81)
- Test: `tests/test_agent_nodes.py`

**Interfaces:**
- Consumes: `_retrieve_and_rerank(..., sources=)` from Task 4.
- Produces:
  - `AgentState` gains `scope_sources: list[str]` (distinct from output `sources: list[dict]`).
  - `run_agent(question, top_k=None, sources: list[str] | None = None)`
  - `stream_agent(question, top_k=None, sources: list[str] | None = None)`
  - `retrieve_node` reads `state.get("scope_sources")` and forwards it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent_nodes.py`:

```python
def test_retrieve_node_forwards_scope_sources(monkeypatch):
    import app.agent.nodes as nodes

    captured = {}
    def fake_retrieve(query, top_k, settings, sources=None):
        captured["sources"] = sources
        return []
    monkeypatch.setattr(nodes, "_retrieve_and_rerank", fake_retrieve)

    nodes.retrieve_node({"question": "q", "query": "q", "top_k": 5, "scope_sources": ["only.pdf"]})
    assert captured["sources"] == ["only.pdf"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_agent_nodes.py::test_retrieve_node_forwards_scope_sources -v`
Expected: FAIL — `captured["sources"]` is `None` (the node does not yet read `scope_sources`).

- [ ] **Step 3: Add the state field**

In `app/agent/state.py`, add inside `AgentState` (after `top_k`):

```python
    scope_sources: list[str]   # optional retrieval scope: restrict to these source paths
```

- [ ] **Step 4: Read and forward it in `retrieve_node`**

Replace `retrieve_node` (lines 33-37) in `app/agent/nodes.py`:

```python
def retrieve_node(state: dict) -> dict:
    settings = get_settings()
    query = state.get("query") or state["question"]
    top_k = state.get("top_k") or settings.TOP_K
    sources = state.get("scope_sources") or None
    return {"documents": _retrieve_and_rerank(query, top_k, settings, sources=sources)}
```

- [ ] **Step 5: Seed `scope_sources` into the agent init dicts**

In `app/agent/graph.py`, change `run_agent` (lines 63-66):

```python
def run_agent(question: str, top_k: int | None = None, sources: list[str] | None = None) -> dict:
    settings = get_settings()
    start = time.time()
    init = {"question": question, "query": question, "top_k": top_k or settings.TOP_K,
            "attempts": 0, "scope_sources": sources or []}
```

Change `stream_agent` (lines 78-81):

```python
async def stream_agent(question: str, top_k: int | None = None, sources: list[str] | None = None) -> AsyncIterator[dict]:
    settings = get_settings()
    start = time.time()
    init = {"question": question, "query": question, "top_k": top_k or settings.TOP_K,
            "attempts": 0, "scope_sources": sources or []}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_agent_nodes.py tests/test_agent_graph.py -v`
Expected: PASS (existing + new).

- [ ] **Step 7: Commit**

```bash
git add app/agent/state.py app/agent/nodes.py app/agent/graph.py tests/test_agent_nodes.py
git commit -m "feat(agent): thread source scope through the agent retrieve path"
```

---

### Task 6: API — accept `sources` on `/chat` and `/agent`

**Files:**
- Modify: `app/api/routes_chat.py` (`ChatRequest` line 18-20; `chat` line 43; `chat_stream` line 75)
- Modify: `app/api/routes_agent.py` (`AgentRequest` line 18-20; `agent` line 39; `agent_stream` line 61)
- Test: `tests/test_api.py`, `tests/test_agent_api.py`

**Interfaces:**
- Consumes: `query_pipeline(..., sources=)`, `stream_query(..., sources=)`, `run_agent(..., sources=)`, `stream_agent(..., sources=)`.
- Produces: request bodies accept optional `sources: list[str] | None = None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_api.py` (follow the file's existing client/auth fixture style; reuse its module-level `client` if present):

```python
def test_chat_forwards_sources(monkeypatch):
    import app.api.routes_chat as rc

    captured = {}
    def fake_query(question, top_k, sources=None):
        captured["sources"] = sources
        return {"answer": "ok", "sources": [], "latency_ms": 1.0, "usage": {}}
    monkeypatch.setattr(rc, "query_pipeline", fake_query)

    from fastapi.testclient import TestClient
    from app.main import app
    resp = TestClient(app).post("/chat", json={"question": "q", "sources": ["only.pdf"]})
    assert resp.status_code == 200
    assert captured["sources"] == ["only.pdf"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_api.py::test_chat_forwards_sources -v`
Expected: FAIL — `captured["sources"]` is `None` (route ignores the field).

- [ ] **Step 3: Add `sources` to the request models**

In `app/api/routes_chat.py`, extend `ChatRequest`:

```python
class ChatRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=50)
    sources: list[str] | None = None
```

In `app/api/routes_agent.py`, extend `AgentRequest` the same way:

```python
class AgentRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=50)
    sources: list[str] | None = None
```

- [ ] **Step 4: Forward `sources` from each handler**

`app/api/routes_chat.py` — in `chat`, change the pipeline call (line 43):
```python
    result = await asyncio.to_thread(query_pipeline, body.question, body.top_k, body.sources)
```
In `chat_stream`, change the generator call (line 75):
```python
            async for event in stream_query(body.question, body.top_k, body.sources):
```

`app/api/routes_agent.py` — in `agent`, change the call (line 39):
```python
    result = await asyncio.to_thread(run_agent, body.question, body.top_k, body.sources)
```
In `agent_stream`, change the call (line 61):
```python
            async for event in stream_agent(body.question, body.top_k, body.sources):
```

- [ ] **Step 5: Run the API tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_api.py tests/test_agent_api.py -v`
Expected: PASS (existing + new). Add an equivalent `test_agent_forwards_sources` to `tests/test_agent_api.py` mirroring Step 1 against `run_agent`, and confirm it passes.

- [ ] **Step 6: Full backend suite (no regressions)**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add app/api/routes_chat.py app/api/routes_agent.py tests/test_api.py tests/test_agent_api.py
git commit -m "feat(api): accept optional source scope on chat and agent endpoints"
```

---

### Task 7: Lift chat state into a persistent `ChatProvider` (survives nav + reload) + New chat

**Files:**
- Modify: `frontend/src/hooks/useChat.ts` (add persistence + `clear`)
- Create: `frontend/src/context/ChatContext.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/pages/ChatPage.tsx`
- Modify: `frontend/src/components/chat/ChatControls.tsx` (New chat button)
- Test: `frontend/src/hooks/useChat.test.tsx`

**Interfaces:**
- Produces:
  - `useChat({ persistKey?: string })` returns `{ messages, busy, send, clear }`, hydrating `messages` from and writing them to `localStorage[persistKey]` (capped to the last 50), `clear()` empties messages and removes the key.
  - `ChatProvider` + `useChatContext()` exposing `{ messages, busy, send, clear, agent, stream, topK, setAgent, setStream, setTopK }`.

- [ ] **Step 1: Write the failing test**

Append to `frontend/src/hooks/useChat.test.tsx` (match the file's existing render/provider wrappers for `SettingsContext`/`ToastContext`):

```tsx
it("hydrates messages from localStorage and clear() empties them", async () => {
  localStorage.setItem(
    "test-chat",
    JSON.stringify([{ role: "user", content: "hi" }, { role: "assistant", content: "yo" }]),
  );
  const { result } = renderHook(() => useChat({ persistKey: "test-chat" }), { wrapper });
  expect(result.current.messages).toHaveLength(2);

  act(() => result.current.clear());
  expect(result.current.messages).toHaveLength(0);
  expect(localStorage.getItem("test-chat")).toBeNull();
});
```

(If `renderHook`/`act`/`wrapper` are not already imported in this file, add them: `import { act, renderHook } from "@testing-library/react";` and reuse the existing `wrapper` that supplies `SettingsProvider` + `ToastProvider`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/hooks/useChat.test.tsx`
Expected: FAIL — `useChat` takes no argument / `clear` is undefined.

- [ ] **Step 3: Add persistence + clear to `useChat`**

In `frontend/src/hooks/useChat.ts`, replace the state setup (lines 26-30) with an options arg and persisted state:

```ts
const MAX_PERSISTED = 50;

export function useChat(options: { persistKey?: string } = {}) {
  const { persistKey } = options;
  const { client } = useSettings();
  const { toast } = useToast();
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    if (!persistKey) return [];
    try {
      const raw = localStorage.getItem(persistKey);
      return raw ? (JSON.parse(raw) as ChatMessage[]) : [];
    } catch {
      return [];
    }
  });
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!persistKey) return;
    try {
      localStorage.setItem(persistKey, JSON.stringify(messages.slice(-MAX_PERSISTED)));
    } catch {
      /* ignore quota/serialization errors; keep in-memory state */
    }
  }, [messages, persistKey]);

  const clear = useCallback(() => {
    setMessages([]);
    if (persistKey) {
      try {
        localStorage.removeItem(persistKey);
      } catch {
        /* ignore */
      }
    }
  }, [persistKey]);
```

Add `useEffect` to the React import (line 1):

```ts
import { useCallback, useEffect, useState } from "react";
```

Add `clear` to the returned object (line 93):

```ts
  return { messages, busy, send, clear };
```

- [ ] **Step 4: Run the hook test to verify it passes**

Run: `cd frontend && npx vitest run src/hooks/useChat.test.tsx`
Expected: PASS.

- [ ] **Step 5: Create the provider**

Create `frontend/src/context/ChatContext.tsx`:

```tsx
import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import { useChat } from "../hooks/useChat";

type ChatContextValue = ReturnType<typeof useChat> & {
  agent: boolean;
  stream: boolean;
  topK: number;
  setAgent: (v: boolean) => void;
  setStream: (v: boolean) => void;
  setTopK: (v: number) => void;
};

const ChatContext = createContext<ChatContextValue | null>(null);

export function ChatProvider({ children }: { children: ReactNode }) {
  const chat = useChat({ persistKey: "rag-chat" });
  const [agent, setAgent] = useState(false);
  const [stream, setStream] = useState(true);
  const [topK, setTopK] = useState(5);
  const value = useMemo(
    () => ({ ...chat, agent, stream, topK, setAgent, setStream, setTopK }),
    [chat, agent, stream, topK],
  );
  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChatContext(): ChatContextValue {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error("useChatContext must be used within ChatProvider");
  return ctx;
}
```

- [ ] **Step 6: Mount the provider above the router**

In `frontend/src/App.tsx`, import and wrap:

```tsx
import { ChatProvider } from "./context/ChatContext";
```
Wrap the `<main>...<Routes/>...</main>` (or the whole returned tree) in `<ChatProvider>`:

```tsx
      <ChatProvider>
        <main className="mx-auto max-w-5xl px-4 py-6">
          <Routes>
            <Route path="/" element={<ChatPage />} />
            <Route path="/documents" element={<DocumentsPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>
      </ChatProvider>
```

- [ ] **Step 7: Consume the context in ChatPage**

Replace the body of `frontend/src/pages/ChatPage.tsx` to use the context instead of local state:

```tsx
import { ChatControls } from "../components/chat/ChatControls";
import { MessageThread } from "../components/chat/MessageThread";
import { useChatContext } from "../context/ChatContext";

export function ChatPage() {
  const { messages, busy, send, clear, agent, stream, topK, setAgent, setStream, setTopK } =
    useChatContext();

  return (
    <div className="flex h-[calc(100vh-9rem)] flex-col rounded-lg border border-muted/30 bg-surface">
      <div className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 ? (
          <p className="text-sm text-muted">Ask a question about your ingested documents.</p>
        ) : (
          <MessageThread messages={messages} />
        )}
      </div>
      <ChatControls
        agent={agent}
        stream={stream}
        topK={topK}
        busy={busy}
        onAgent={setAgent}
        onStream={setStream}
        onTopK={setTopK}
        onClear={clear}
        onSend={(q) => send(q, { agent, stream, topK })}
      />
    </div>
  );
}
```

- [ ] **Step 8: Add the New chat button to ChatControls**

In `frontend/src/components/chat/ChatControls.tsx`, add `onClear: () => void;` to `Props`, accept it in the destructure, and add a button in the controls row (after the `top_k` label, line 41):

```tsx
        <button
          type="button"
          className="ml-auto rounded border border-muted/50 px-2 py-1 text-xs hover:bg-muted/10"
          onClick={onClear}
        >
          New chat
        </button>
```

- [ ] **Step 9: Run the affected frontend tests**

Run: `cd frontend && npx vitest run src/hooks/useChat.test.tsx src/pages/ChatPage.test.tsx`
Expected: PASS. Update `ChatPage.test.tsx` to render inside `<ChatProvider>` if it previously relied on `ChatPage` owning its own state.

- [ ] **Step 10: Commit**

```bash
git add frontend/src/hooks/useChat.ts frontend/src/context/ChatContext.tsx frontend/src/App.tsx frontend/src/pages/ChatPage.tsx frontend/src/components/chat/ChatControls.tsx frontend/src/hooks/useChat.test.tsx frontend/src/pages/ChatPage.test.tsx
git commit -m "feat(ui): persist chat across navigation and reload with New chat reset"
```

---

### Task 8: Document scope picker (frontend) wired into the request

**Files:**
- Modify: `frontend/src/hooks/useChat.ts` (`SendOpts` + body)
- Modify: `frontend/src/context/ChatContext.tsx` (add `scopeSources`)
- Create: `frontend/src/components/chat/DocumentScopePicker.tsx`
- Create: `frontend/src/components/chat/DocumentScopePicker.test.tsx`
- Modify: `frontend/src/pages/ChatPage.tsx` (render picker, pass `sources` to `send`)

**Interfaces:**
- Consumes: `useDocuments()` (`docs: DocumentRecord[]`), `displaySource` (Task 1), API `sources` field (Task 6).
- Produces: `SendOpts` gains `sources: string[]`; the request body includes `sources` only when non-empty. `useChatContext()` gains `scopeSources: string[]` and `setScopeSources`.

- [ ] **Step 1: Write the failing test (picker)**

Create `frontend/src/components/chat/DocumentScopePicker.test.tsx`:

```tsx
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DocumentScopePicker } from "./DocumentScopePicker";

const docs = [
  { id: "1", source: "./data/papers/gpt3.pdf", chunks: 178, ingested_at: "" },
  { id: "2", source: "data\\Quiz_ Practice Exam.pdf", chunks: 4, ingested_at: "" },
];

describe("DocumentScopePicker", () => {
  it("renders clean filenames and toggles selection", () => {
    const onChange = vi.fn();
    render(<DocumentScopePicker docs={docs} selected={[]} onChange={onChange} />);

    expect(screen.getByText("gpt3.pdf")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Quiz_ Practice Exam.pdf"));
    expect(onChange).toHaveBeenCalledWith(["data\\Quiz_ Practice Exam.pdf"]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/components/chat/DocumentScopePicker.test.tsx`
Expected: FAIL — cannot resolve `./DocumentScopePicker`.

- [ ] **Step 3: Implement the picker**

Create `frontend/src/components/chat/DocumentScopePicker.tsx`:

```tsx
import type { DocumentRecord } from "../../api/types";
import { displaySource } from "../../api/sources";

interface Props {
  docs: DocumentRecord[];
  selected: string[];
  onChange: (next: string[]) => void;
}

export function DocumentScopePicker({ docs, selected, onChange }: Props) {
  if (docs.length === 0) return null;
  const toggle = (source: string) => {
    onChange(
      selected.includes(source) ? selected.filter((s) => s !== source) : [...selected, source],
    );
  };
  return (
    <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
      <span className="text-muted">Scope:</span>
      <button
        type="button"
        className={`rounded-full border px-2 py-0.5 ${selected.length === 0 ? "border-primary text-primary" : "border-muted/50 text-muted"}`}
        onClick={() => onChange([])}
      >
        All documents
      </button>
      {docs.map((d) => {
        const on = selected.includes(d.source);
        return (
          <button
            key={d.id}
            type="button"
            title={d.source}
            className={`rounded-full border px-2 py-0.5 ${on ? "border-primary text-primary" : "border-muted/50 text-muted"}`}
            onClick={() => toggle(d.source)}
          >
            {displaySource(d.source)}
          </button>
        );
      })}
    </div>
  );
}
```

- [ ] **Step 4: Run the picker test to verify it passes**

Run: `cd frontend && npx vitest run src/components/chat/DocumentScopePicker.test.tsx`
Expected: PASS.

- [ ] **Step 5: Extend `SendOpts` and the request body**

In `frontend/src/hooks/useChat.ts`, extend `SendOpts` (lines 20-24):

```ts
export interface SendOpts {
  agent: boolean;
  stream: boolean;
  topK: number;
  sources?: string[];
}
```

Change the body construction (line 47) to include `sources` only when non-empty:

```ts
      const body: { question: string; top_k: number; sources?: string[] } = {
        question,
        top_k: opts.topK,
      };
      if (opts.sources && opts.sources.length > 0) body.sources = opts.sources;
```

- [ ] **Step 6: Add `scopeSources` to the provider**

In `frontend/src/context/ChatContext.tsx`, add state and expose it:

```tsx
  const [scopeSources, setScopeSources] = useState<string[]>([]);
```
Extend the `ChatContextValue` type and the `useMemo` value to include `scopeSources` and `setScopeSources` (mirror how `agent` is wired), and add both to the `useMemo` dependency array.

- [ ] **Step 7: Render the picker and pass `sources` to send**

In `frontend/src/pages/ChatPage.tsx`, pull docs and scope, render the picker above the message list, and forward `sources`:

```tsx
import { useEffect } from "react";
import { ChatControls } from "../components/chat/ChatControls";
import { DocumentScopePicker } from "../components/chat/DocumentScopePicker";
import { MessageThread } from "../components/chat/MessageThread";
import { useChatContext } from "../context/ChatContext";
import { useDocuments } from "../hooks/useDocuments";

export function ChatPage() {
  const { messages, busy, send, clear, agent, stream, topK, setAgent, setStream, setTopK,
    scopeSources, setScopeSources } = useChatContext();
  const { docs, refresh } = useDocuments();
  useEffect(() => { refresh(); }, [refresh]);

  return (
    <div className="flex h-[calc(100vh-9rem)] flex-col rounded-lg border border-muted/30 bg-surface">
      <div className="border-b border-muted/30 px-4 pt-3">
        <DocumentScopePicker docs={docs} selected={scopeSources} onChange={setScopeSources} />
      </div>
      <div className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 ? (
          <p className="text-sm text-muted">Ask a question about your ingested documents.</p>
        ) : (
          <MessageThread messages={messages} />
        )}
      </div>
      <ChatControls
        agent={agent} stream={stream} topK={topK} busy={busy}
        onAgent={setAgent} onStream={setStream} onTopK={setTopK} onClear={clear}
        onSend={(q) => send(q, { agent, stream, topK, sources: scopeSources })}
      />
    </div>
  );
}
```

- [ ] **Step 8: Run frontend tests + typecheck**

Run: `cd frontend && npx vitest run` then `cd frontend && npx tsc --noEmit`
Expected: all tests PASS, no type errors. Fix any `ChatPage.test.tsx` render to wrap in `<ChatProvider>` and (if it asserts network) mock `useDocuments`.

- [ ] **Step 9: Manual smoke test**

With backend + Qdrant up and `npm run dev`: select the quiz chip, ask "What topics does this quiz cover?", confirm the answer cites only the quiz and the source card shows `Quiz_ Practice Exam.pdf`. Switch to Documents and back; confirm the conversation is still there. Reload the browser; confirm it survives. Click New chat; confirm it clears.

- [ ] **Step 10: Commit**

```bash
git add frontend/src/hooks/useChat.ts frontend/src/context/ChatContext.tsx frontend/src/components/chat/DocumentScopePicker.tsx frontend/src/components/chat/DocumentScopePicker.test.tsx frontend/src/pages/ChatPage.tsx
git commit -m "feat(ui): add document scope picker and send source filter with chat"
```

---

## Self-Review

**Spec coverage:**
- Feature 1 (persistence + nav + reload + New chat) -> Task 7. ✓
- Feature 2 backend (vector/BM25 filter, graph-skip, cache-bypass, pipeline + agent threading, API) -> Tasks 2-6. ✓
- Feature 2 frontend (multi-select picker, default all, send `sources`) -> Task 8. ✓
- Feature 3 (clean filenames, tooltip with raw value, used in cards + picker) -> Task 1, reused in Task 8. ✓
- Testing section (backend filter/threading/cache/api; frontend helper/persistence/picker) -> covered across tasks. ✓

**Placeholder scan:** No TBD/TODO; every code step has concrete code. The one runtime check (Task 3 Step 5, Qdrant payload key) is an explicit verification with a defined fallback, not a placeholder.

**Type/name consistency:** `sources: list[str]` used uniformly in API + pipeline + retriever + stores; agent carries it as `scope_sources` (deliberately distinct from output `sources: list[dict]`) and `run_agent`/`stream_agent` expose it as the `sources` parameter; frontend context uses `scopeSources`, serialized to body field `sources`. `displaySource`, `clear`, `useChatContext`, `DocumentScopePicker` props (`docs`/`selected`/`onChange`) match between definition and use.
