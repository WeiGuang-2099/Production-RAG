# Agentic RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Corrective-RAG agent (route → retrieve → grade → rewrite-loop → generate) built on LangGraph, exposed as `POST /agent` alongside the existing `/chat`.

**Architecture:** A LangGraph `StateGraph` whose nodes reuse the existing retrieval (`_retrieve_and_rerank`) and grounded generation (`select_prompt`/`format_context`/`usage_for`). LLM access goes through the existing `get_llm()` factory. Each node is a pure `state -> partial_state` function, unit-tested with mocks; the compiled graph is integration-tested by patching node functions and asserting the path taken.

**Tech Stack:** Python 3.11+, LangGraph, LangChain, FastAPI, pytest.

## Global Constraints

- Python `>=3.11`; run tests/lint with the project venv: `./.venv/Scripts/python.exe -m pytest` and `./.venv/Scripts/python.exe -m ruff check .`
- TDD: write the failing test first, watch it fail, then implement.
- Commits: conventional style, no Claude attribution / co-author trailers, do not push (user pushes manually).
- Reuse existing helpers; do not duplicate retrieval or generation logic.
- New dependency floor: `langgraph>=0.2.0`.
- Default config value (verbatim): `AGENT_MAX_REWRITES = 2`.

---

### Task 1: Config setting + LangGraph dependency

**Files:**
- Modify: `app/config.py` (add `AGENT_MAX_REWRITES` + validator)
- Modify: `pyproject.toml` (add `langgraph>=0.2.0`)
- Modify: `.env.example` (document the setting)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings.AGENT_MAX_REWRITES: int` (default 2, validated `>= 0`)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_config.py`:

```python
def test_agent_max_rewrites_default():
    settings = Settings(
        LLM_API_KEY="test", EMBEDDING_API_KEY="test", COHERE_API_KEY="test"
    )
    assert settings.AGENT_MAX_REWRITES == 2


def test_invalid_agent_max_rewrites():
    with pytest.raises(ValueError, match="AGENT_MAX_REWRITES must be"):
        Settings(
            AGENT_MAX_REWRITES=-1,
            LLM_API_KEY="test",
            EMBEDDING_API_KEY="test",
            COHERE_API_KEY="test",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_config.py -q`
Expected: FAIL (`AttributeError`/no validator)

- [ ] **Step 3: Add the setting and validator** — in `app/config.py`, add the field in the `# Cache` block area (after `CACHE_SIMILARITY_THRESHOLD`):

```python
    # Agent
    AGENT_MAX_REWRITES: int = 2
```

and add the validator next to the other validators:

```python
    @field_validator("AGENT_MAX_REWRITES")
    @classmethod
    def validate_agent_max_rewrites(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"AGENT_MAX_REWRITES must be >= 0, got {v}")
        return v
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_config.py -q`
Expected: PASS

- [ ] **Step 5: Add the dependency and docs**

In `pyproject.toml` `dependencies`, add the line:

```
    "langgraph>=0.2.0",
```

In `.env.example`, under the `# ── Generation ──` section add:

```
AGENT_MAX_REWRITES=2                    # /agent corrective-RAG max query rewrites
```

- [ ] **Step 6: Install langgraph into the venv**

Run: `./.venv/Scripts/python.exe -m pip install "langgraph>=0.2.0"`
Expected: installs without errors; `./.venv/Scripts/python.exe -c "import langgraph; print('ok')"` prints `ok`

- [ ] **Step 7: Commit**

```bash
git add app/config.py pyproject.toml .env.example tests/test_config.py
git commit -m "feat: AGENT_MAX_REWRITES config and langgraph dependency"
```

---

### Task 2: Agent state, prompts, and output parsers

**Files:**
- Create: `app/agent/__init__.py` (empty)
- Create: `app/agent/state.py`
- Create: `app/agent/prompts.py`
- Create: `app/agent/parsers.py`
- Test: `tests/test_agent_parsers.py`

**Interfaces:**
- Produces: `AgentState` (TypedDict); prompt constants `ROUTER_PROMPT`, `GRADER_PROMPT`, `REWRITE_PROMPT`, `ANSWER_DIRECTLY_PROMPT`, `CLARIFY_PROMPT`; `parse_route(text: str) -> str` (one of `retrieve|answer|clarify`, default `retrieve`); `parse_grade(text: str) -> bool` (default `True`).

- [ ] **Step 1: Write the failing parser tests** — create `tests/test_agent_parsers.py`:

```python
from app.agent.parsers import parse_grade, parse_route


def test_parse_route_each_label():
    assert parse_route("retrieve") == "retrieve"
    assert parse_route("answer") == "answer"
    assert parse_route("clarify") == "clarify"


def test_parse_route_is_case_and_noise_insensitive():
    assert parse_route("RETRIEVE.") == "retrieve"
    assert parse_route("I think we should answer this") == "answer"


def test_parse_route_defaults_to_retrieve():
    assert parse_route("???") == "retrieve"
    assert parse_route("") == "retrieve"


def test_parse_grade_yes_no():
    assert parse_grade("yes") is True
    assert parse_grade("Yes, sufficient") is True
    assert parse_grade("no") is False
    assert parse_grade("No, missing the key fact") is False


def test_parse_grade_defaults_to_true():
    assert parse_grade("???") is True
    assert parse_grade("") is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_agent_parsers.py -q`
Expected: FAIL (module missing)

- [ ] **Step 3: Create the package and parsers**

`app/agent/__init__.py`: empty file.

`app/agent/parsers.py`:

```python
"""Robust parsers for the agent's single-word LLM decisions."""
from __future__ import annotations


def parse_route(text: str) -> str:
    """Map a router LLM response to one of retrieve|answer|clarify.

    Unparseable output fails safe to ``retrieve`` (the most useful default for
    a document-QA system)."""
    t = (text or "").strip().lower()
    for label in ("retrieve", "answer", "clarify"):
        if label in t:
            return label
    return "retrieve"


def parse_grade(text: str) -> bool:
    """Map a grader LLM response to sufficient (True) / insufficient (False).

    Unparseable output fails open to ``True`` so generation still runs; the
    grounded prompt refuses if the context is genuinely insufficient."""
    t = (text or "").strip().lower()
    if t.startswith("no"):
        return False
    if t.startswith("yes"):
        return True
    return "no" not in t.split()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_agent_parsers.py -q`
Expected: PASS

- [ ] **Step 5: Create the state and prompts (no tests needed — pure data)**

`app/agent/state.py`:

```python
from typing import TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    question: str            # original question, never mutated
    query: str               # current search query; rewritten by `rewrite`
    top_k: int
    route: str               # retrieve | answer | clarify
    documents: list[Document]
    relevant: bool
    attempts: int
    answer: str
    sources: list[dict]
    usage: dict
```

`app/agent/prompts.py`:

```python
"""Prompts for the agent's control nodes (router, grader, rewriter, etc.)."""

ROUTER_PROMPT = """Classify how to handle the user's question. Reply with exactly one word:
- retrieve: needs information looked up in the document corpus
- answer: a greeting or general question that needs no document lookup
- clarify: too vague or ambiguous to act on

Question: {question}

One word:"""

GRADER_PROMPT = """Decide whether the retrieved context is sufficient to answer the question.
Reply with exactly one word: yes or no.

Question: {question}

Context:
{context}

Sufficient (yes/no):"""

REWRITE_PROMPT = """The previous search did not retrieve enough relevant context. Rewrite the
question into a better search query using keywords and key entities. Return only the query.

Question: {question}

Rewritten search query:"""

ANSWER_DIRECTLY_PROMPT = """Answer the user's question directly and concisely. This is a general
question not grounded in the document corpus, so do not invent citations.

Question: {question}

Answer:"""

CLARIFY_PROMPT = """The user's question is too vague to answer well. Ask one concise clarifying
question to narrow it down.

Question: {question}

Clarifying question:"""
```

- [ ] **Step 6: Commit**

```bash
git add app/agent/__init__.py app/agent/state.py app/agent/prompts.py app/agent/parsers.py tests/test_agent_parsers.py
git commit -m "feat: agent state, prompts, and output parsers"
```

---

### Task 3: Agent node functions

**Files:**
- Create: `app/agent/nodes.py`
- Test: `tests/test_agent_nodes.py`

**Interfaces:**
- Consumes: `parse_route`, `parse_grade` (Task 2); `app.core.pipeline._retrieve_and_rerank`, `app.core.pipeline._docs_to_sources`; `app.core.prompts.select_prompt`, `format_context`; `app.observability.cost.usage_for`; `app.core.factories.get_llm`; `app.config.get_settings`.
- Produces node functions, each `state -> dict` partial state:
  `route_question` → `{"route": str}`; `retrieve_node` → `{"documents": list}`; `grade_documents` → `{"relevant": bool}`; `rewrite_query` → `{"query": str, "attempts": int}`; `generate_node`/`answer_directly`/`clarify` → `{"answer": str, "sources": list, "usage": dict}`.

- [ ] **Step 1: Write the failing node tests** — create `tests/test_agent_nodes.py`:

```python
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.agent import nodes


def _llm(content):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=content)
    return llm


def test_route_question_uses_parsed_label():
    with patch("app.agent.nodes.get_llm", return_value=_llm("answer")):
        assert nodes.route_question({"question": "hi"}) == {"route": "answer"}


def test_route_question_defaults_to_retrieve_on_error():
    with patch("app.agent.nodes.get_llm", side_effect=RuntimeError("boom")):
        assert nodes.route_question({"question": "hi"}) == {"route": "retrieve"}


def test_retrieve_node_uses_current_query():
    doc = Document(page_content="d", metadata={"source": "a"})
    with patch("app.agent.nodes._retrieve_and_rerank", return_value=[doc]) as mock_rr, \
         patch("app.agent.nodes.get_settings") as mock_s:
        mock_s.return_value.TOP_K = 5
        out = nodes.retrieve_node({"question": "orig", "query": "rewritten", "top_k": 7})
        assert out["documents"] == [doc]
        assert mock_rr.call_args[0][0] == "rewritten"
        assert mock_rr.call_args[0][1] == 7


def test_grade_documents_no_docs_is_irrelevant():
    assert nodes.grade_documents({"question": "q", "documents": []}) == {"relevant": False}


def test_grade_documents_yes():
    doc = Document(page_content="d", metadata={"source": "a"})
    with patch("app.agent.nodes.get_llm", return_value=_llm("yes")):
        assert nodes.grade_documents({"question": "q", "documents": [doc]}) == {"relevant": True}


def test_grade_documents_no():
    doc = Document(page_content="d", metadata={"source": "a"})
    with patch("app.agent.nodes.get_llm", return_value=_llm("no")):
        assert nodes.grade_documents({"question": "q", "documents": [doc]}) == {"relevant": False}


def test_rewrite_query_increments_attempts():
    with patch("app.agent.nodes.get_llm", return_value=_llm("better query")):
        out = nodes.rewrite_query({"question": "q", "attempts": 0})
        assert out["query"] == "better query"
        assert out["attempts"] == 1


def test_generate_node_produces_cited_sources_and_usage():
    doc = Document(page_content="ctx", metadata={"source": "a.pdf"})
    with patch("app.agent.nodes.get_llm", return_value=_llm("the answer")), \
         patch("app.agent.nodes.get_settings") as mock_s:
        mock_s.return_value.PROMPT_MODE = "grounded"
        mock_s.return_value.LLM_MODEL = "gpt-4o"
        out = nodes.generate_node({"question": "q", "documents": [doc]})
        assert out["answer"] == "the answer"
        assert out["sources"][0]["metadata"]["citation"] == 1
        assert "cost_usd" in out["usage"]


def test_answer_directly_has_no_sources():
    with patch("app.agent.nodes.get_llm", return_value=_llm("general answer")), \
         patch("app.agent.nodes.get_settings") as mock_s:
        mock_s.return_value.LLM_MODEL = "gpt-4o"
        out = nodes.answer_directly({"question": "hi"})
        assert out["answer"] == "general answer"
        assert out["sources"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_agent_nodes.py -q`
Expected: FAIL (module missing)

- [ ] **Step 3: Implement the nodes** — create `app/agent/nodes.py`:

```python
"""Agent nodes. Each is `state -> partial_state` and reuses existing pipeline
helpers so the agent and /chat share the exact retrieval and generation code."""
from __future__ import annotations

import logging

from app.agent.parsers import parse_grade, parse_route
from app.agent.prompts import (
    ANSWER_DIRECTLY_PROMPT,
    CLARIFY_PROMPT,
    GRADER_PROMPT,
    REWRITE_PROMPT,
    ROUTER_PROMPT,
)
from app.config import get_settings
from app.core.factories import get_llm
from app.core.pipeline import _docs_to_sources, _retrieve_and_rerank
from app.core.prompts import format_context, select_prompt
from app.observability.cost import usage_for

logger = logging.getLogger(__name__)


def _invoke(llm, prompt: str) -> str:
    return getattr(llm.invoke(prompt), "content", "") or ""


def route_question(state: dict) -> dict:
    try:
        out = _invoke(get_llm(), ROUTER_PROMPT.format(question=state["question"]))
        return {"route": parse_route(out)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("route_failed: %s", exc)
        return {"route": "retrieve"}


def retrieve_node(state: dict) -> dict:
    settings = get_settings()
    query = state.get("query") or state["question"]
    top_k = state.get("top_k") or settings.TOP_K
    return {"documents": _retrieve_and_rerank(query, top_k, settings)}


def grade_documents(state: dict) -> dict:
    docs = state.get("documents") or []
    if not docs:
        return {"relevant": False}
    try:
        out = _invoke(
            get_llm(),
            GRADER_PROMPT.format(question=state["question"], context=format_context(docs)),
        )
        return {"relevant": parse_grade(out)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("grade_failed: %s", exc)
        return {"relevant": True}


def rewrite_query(state: dict) -> dict:
    attempts = state.get("attempts", 0) + 1
    fallback = state.get("query") or state["question"]
    try:
        out = _invoke(get_llm(), REWRITE_PROMPT.format(question=state["question"])).strip()
        return {"query": out or fallback, "attempts": attempts}
    except Exception as exc:  # noqa: BLE001
        logger.warning("rewrite_failed: %s", exc)
        return {"query": fallback, "attempts": attempts}


def _generate_with(prompt_text: str) -> tuple[str, dict]:
    settings = get_settings()
    answer = _invoke(get_llm(), prompt_text)
    usage = usage_for(prompt_text, answer, str(settings.LLM_MODEL))
    return answer, usage


def generate_node(state: dict) -> dict:
    settings = get_settings()
    docs = state.get("documents") or []
    prompt_text = select_prompt(settings.PROMPT_MODE).format(
        context=format_context(docs), question=state["question"]
    )
    answer, usage = _generate_with(prompt_text)
    return {"answer": answer, "sources": _docs_to_sources(docs), "usage": usage}


def answer_directly(state: dict) -> dict:
    answer, usage = _generate_with(ANSWER_DIRECTLY_PROMPT.format(question=state["question"]))
    return {"answer": answer, "sources": [], "usage": usage}


def clarify(state: dict) -> dict:
    answer, usage = _generate_with(CLARIFY_PROMPT.format(question=state["question"]))
    return {"answer": answer, "sources": [], "usage": usage}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_agent_nodes.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent/nodes.py tests/test_agent_nodes.py
git commit -m "feat: agent nodes (route, retrieve, grade, rewrite, generate)"
```

---

### Task 4: Graph wiring + run_agent

**Files:**
- Create: `app/agent/graph.py`
- Test: `tests/test_agent_graph.py`

**Interfaces:**
- Consumes: `app.agent.nodes.*` (Task 3); `app.agent.state.AgentState`; `Settings.AGENT_MAX_REWRITES`.
- Produces: `build_agent_graph() -> CompiledGraph`; `run_agent(question: str, top_k: int | None = None) -> dict` returning `{answer, sources, latency_ms, usage, route, attempts}`.

- [ ] **Step 1: Write the failing graph tests** — create `tests/test_agent_graph.py`:

```python
from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.agent import graph as graph_mod
from app.agent import nodes


def _settings(max_rewrites):
    s = MagicMock()
    s.AGENT_MAX_REWRITES = max_rewrites
    s.TOP_K = 5
    return s


def test_relevant_path_generates(monkeypatch):
    monkeypatch.setattr(nodes, "route_question", lambda s: {"route": "retrieve"})
    monkeypatch.setattr(nodes, "retrieve_node", lambda s: {"documents": [Document(page_content="d")]})
    monkeypatch.setattr(nodes, "grade_documents", lambda s: {"relevant": True})
    monkeypatch.setattr(nodes, "generate_node", lambda s: {"answer": "A", "sources": [], "usage": {}})
    monkeypatch.setattr(graph_mod, "get_settings", lambda: _settings(2))

    final = graph_mod.build_agent_graph().invoke(
        {"question": "q", "query": "q", "top_k": 5, "attempts": 0}
    )
    assert final["answer"] == "A"


def test_weak_then_relevant_rewrites_once(monkeypatch):
    calls = {"retrieve": 0}

    def fake_retrieve(s):
        calls["retrieve"] += 1
        return {"documents": [Document(page_content="d")]}

    grades = iter([{"relevant": False}, {"relevant": True}])
    monkeypatch.setattr(nodes, "route_question", lambda s: {"route": "retrieve"})
    monkeypatch.setattr(nodes, "retrieve_node", fake_retrieve)
    monkeypatch.setattr(nodes, "grade_documents", lambda s: next(grades))
    monkeypatch.setattr(nodes, "rewrite_query", lambda s: {"query": "q2", "attempts": s.get("attempts", 0) + 1})
    monkeypatch.setattr(nodes, "generate_node", lambda s: {"answer": "A", "sources": [], "usage": {}})
    monkeypatch.setattr(graph_mod, "get_settings", lambda: _settings(2))

    final = graph_mod.build_agent_graph().invoke(
        {"question": "q", "query": "q", "top_k": 5, "attempts": 0}
    )
    assert calls["retrieve"] == 2
    assert final["attempts"] == 1
    assert final["answer"] == "A"


def test_rewrite_loop_is_capped(monkeypatch):
    calls = {"retrieve": 0}

    def fake_retrieve(s):
        calls["retrieve"] += 1
        return {"documents": [Document(page_content="d")]}

    monkeypatch.setattr(nodes, "route_question", lambda s: {"route": "retrieve"})
    monkeypatch.setattr(nodes, "retrieve_node", fake_retrieve)
    monkeypatch.setattr(nodes, "grade_documents", lambda s: {"relevant": False})
    monkeypatch.setattr(nodes, "rewrite_query", lambda s: {"query": "q2", "attempts": s.get("attempts", 0) + 1})
    monkeypatch.setattr(nodes, "generate_node", lambda s: {"answer": "A", "sources": [], "usage": {}})
    monkeypatch.setattr(graph_mod, "get_settings", lambda: _settings(1))

    final = graph_mod.build_agent_graph().invoke(
        {"question": "q", "query": "q", "top_k": 5, "attempts": 0}
    )
    assert calls["retrieve"] == 2          # initial + 1 rewrite, then capped
    assert final["answer"] == "A"


def test_answer_route_skips_retrieval(monkeypatch):
    monkeypatch.setattr(nodes, "route_question", lambda s: {"route": "answer"})
    monkeypatch.setattr(nodes, "retrieve_node", lambda s: (_ for _ in ()).throw(AssertionError("should not retrieve")))
    monkeypatch.setattr(nodes, "answer_directly", lambda s: {"answer": "hi there", "sources": [], "usage": {}})
    monkeypatch.setattr(graph_mod, "get_settings", lambda: _settings(2))

    final = graph_mod.build_agent_graph().invoke(
        {"question": "hi", "query": "hi", "top_k": 5, "attempts": 0}
    )
    assert final["answer"] == "hi there"


def test_run_agent_returns_response_shape(monkeypatch):
    monkeypatch.setattr(nodes, "route_question", lambda s: {"route": "retrieve"})
    monkeypatch.setattr(nodes, "retrieve_node", lambda s: {"documents": [Document(page_content="d")]})
    monkeypatch.setattr(nodes, "grade_documents", lambda s: {"relevant": True})
    monkeypatch.setattr(nodes, "generate_node", lambda s: {"answer": "A", "sources": [], "usage": {"cost_usd": 0.0}})
    monkeypatch.setattr(graph_mod, "get_settings", lambda: _settings(2))
    monkeypatch.setattr(graph_mod, "_compiled", None, raising=False)

    result = graph_mod.run_agent("q", top_k=5)
    assert set(result) >= {"answer", "sources", "latency_ms", "usage", "route", "attempts"}
    assert result["answer"] == "A"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_agent_graph.py -q`
Expected: FAIL (module missing)

- [ ] **Step 3: Implement the graph** — create `app/agent/graph.py`:

```python
"""LangGraph wiring for the Corrective-RAG agent."""
from __future__ import annotations

import logging
import time

from langgraph.graph import END, StateGraph

from app.agent import nodes
from app.agent.state import AgentState
from app.config import get_settings

logger = logging.getLogger(__name__)


def _route_edge(state: dict) -> str:
    return state.get("route", "retrieve")


def _grade_edge(state: dict) -> str:
    if state.get("relevant"):
        return "generate"
    if state.get("attempts", 0) < get_settings().AGENT_MAX_REWRITES:
        return "rewrite"
    return "generate"


def build_agent_graph():
    g = StateGraph(AgentState)
    g.add_node("route", nodes.route_question)
    g.add_node("retrieve", nodes.retrieve_node)
    g.add_node("grade", nodes.grade_documents)
    g.add_node("rewrite", nodes.rewrite_query)
    g.add_node("generate", nodes.generate_node)
    g.add_node("answer_directly", nodes.answer_directly)
    g.add_node("clarify", nodes.clarify)

    g.set_entry_point("route")
    g.add_conditional_edges(
        "route", _route_edge,
        {"retrieve": "retrieve", "answer": "answer_directly", "clarify": "clarify"},
    )
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _grade_edge, {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    g.add_edge("answer_directly", END)
    g.add_edge("clarify", END)
    return g.compile()


_compiled = None


def _get_compiled():
    global _compiled
    if _compiled is None:
        _compiled = build_agent_graph()
    return _compiled


def run_agent(question: str, top_k: int | None = None) -> dict:
    settings = get_settings()
    start = time.time()
    init = {"question": question, "query": question, "top_k": top_k or settings.TOP_K, "attempts": 0}
    final = _get_compiled().invoke(init)
    return {
        "answer": final.get("answer", ""),
        "sources": final.get("sources", []),
        "latency_ms": (time.time() - start) * 1000,
        "usage": final.get("usage", {}),
        "route": final.get("route", ""),
        "attempts": final.get("attempts", 0),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_agent_graph.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent/graph.py tests/test_agent_graph.py
git commit -m "feat: agentic RAG graph wiring and run_agent"
```

---

### Task 5: Streaming (stream_agent)

**Files:**
- Modify: `app/agent/graph.py` (add `stream_agent`)
- Test: `tests/test_agent_graph.py`

**Interfaces:**
- Produces: `async stream_agent(question: str, top_k: int | None = None) -> AsyncIterator[dict]`, emitting `{"event":"step","node":...}` per node, then `{"event":"sources","sources":[...]}`, then `{"event":"done","answer","usage","route","attempts","latency_ms"}`.

- [ ] **Step 1: Write the failing streaming test** — append to `tests/test_agent_graph.py`:

```python
async def test_stream_agent_emits_steps_then_done(monkeypatch):
    class FakeGraph:
        async def astream(self, init, stream_mode="updates"):
            yield {"route": {"route": "retrieve"}}
            yield {"retrieve": {"documents": []}}
            yield {"grade": {"relevant": True}}
            yield {"generate": {"answer": "Final", "sources": [{"content": "d", "metadata": {}}], "usage": {"cost_usd": 0.0}}}

    monkeypatch.setattr(graph_mod, "_get_compiled", lambda: FakeGraph())

    events = [e async for e in graph_mod.stream_agent("q")]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "step"
    assert kinds.count("step") == 4
    assert "sources" in kinds
    assert kinds[-1] == "done"
    assert events[-1]["answer"] == "Final"
    assert [e["node"] for e in events if e["event"] == "step"] == ["route", "retrieve", "grade", "generate"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_agent_graph.py::test_stream_agent_emits_steps_then_done -q`
Expected: FAIL (`stream_agent` missing)

- [ ] **Step 3: Implement stream_agent** — append to `app/agent/graph.py`:

```python
from collections.abc import AsyncIterator  # add to the top-of-file imports


async def stream_agent(question: str, top_k: int | None = None) -> AsyncIterator[dict]:
    settings = get_settings()
    start = time.time()
    init = {"question": question, "query": question, "top_k": top_k or settings.TOP_K, "attempts": 0}
    final: dict = {}
    async for update in _get_compiled().astream(init, stream_mode="updates"):
        for node_name, partial in update.items():
            final.update(partial or {})
            yield {"event": "step", "node": node_name}
    yield {"event": "sources", "sources": final.get("sources", [])}
    yield {
        "event": "done",
        "answer": final.get("answer", ""),
        "usage": final.get("usage", {}),
        "route": final.get("route", ""),
        "attempts": final.get("attempts", 0),
        "latency_ms": (time.time() - start) * 1000,
    }
```

(Place the `from collections.abc import AsyncIterator` import with the other imports at the top of the file, not inline.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_agent_graph.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent/graph.py tests/test_agent_graph.py
git commit -m "feat: token/step streaming for the agent (stream_agent)"
```

---

### Task 6: API endpoints (/agent, /agent/stream)

**Files:**
- Create: `app/api/routes_agent.py`
- Modify: `app/main.py` (include the router)
- Test: `tests/test_agent_api.py`

**Interfaces:**
- Consumes: `app.agent.graph.run_agent`, `stream_agent`; `app.api.deps.verify_api_key`, `limiter`.
- Produces: `POST /agent` → `AgentResponse`; `POST /agent/stream` → NDJSON.

- [ ] **Step 1: Write the failing API tests** — create `tests/test_agent_api.py`:

```python
import json as _json
from unittest.mock import patch


def test_agent_endpoint_returns_route_and_usage(client):
    with patch("app.api.routes_agent.run_agent") as mock_run:
        mock_run.return_value = {
            "answer": "A", "sources": [], "latency_ms": 5.0,
            "usage": {"cost_usd": 0.0}, "route": "retrieve", "attempts": 1,
        }
        resp = client.post("/agent", json={"question": "q"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["route"] == "retrieve"
        assert data["attempts"] == 1


def test_agent_stream_endpoint_emits_events(client):
    async def fake_stream(question, top_k=None):
        yield {"event": "step", "node": "route"}
        yield {"event": "done", "answer": "A", "usage": {}}

    with patch("app.api.routes_agent.stream_agent", fake_stream):
        resp = client.post("/agent/stream", json={"question": "q"})
        assert resp.status_code == 200
        events = [_json.loads(ln) for ln in resp.text.strip().split("\n") if ln]
        assert events[0]["event"] == "step"
        assert events[-1]["event"] == "done"


def test_agent_missing_question(client):
    assert client.post("/agent", json={}).status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_agent_api.py -q`
Expected: FAIL (route not registered)

- [ ] **Step 3: Implement the router** — create `app/api/routes_agent.py`:

```python
import asyncio
import json
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.graph import run_agent, stream_agent
from app.api.deps import limiter, verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=50)


class AgentResponse(BaseModel):
    answer: str
    sources: list[dict]
    latency_ms: float
    usage: dict = {}
    route: str = ""
    attempts: int = 0


@router.post("", response_model=AgentResponse)
@limiter.limit("30/minute")
async def agent(request: Request, body: AgentRequest, _key=Depends(verify_api_key)):
    result = await asyncio.to_thread(run_agent, body.question, body.top_k)
    return AgentResponse(**result)


@router.post("/stream")
@limiter.limit("30/minute")
async def agent_stream(request: Request, body: AgentRequest, _key=Depends(verify_api_key)):
    async def event_generator():
        try:
            async for event in stream_agent(body.question, body.top_k):
                yield json.dumps(event) + "\n"
        except Exception as exc:  # noqa: BLE001
            logger.error("agent stream error: %s", exc)
            yield json.dumps({"event": "error", "detail": str(exc)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
```

- [ ] **Step 4: Register the router** — in `app/main.py`, alongside the existing router includes:

```python
from app.api.routes_agent import router as agent_router
# ...
app.include_router(agent_router)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_agent_api.py -q`
Expected: PASS

- [ ] **Step 6: Run the full suite + lint**

Run: `./.venv/Scripts/python.exe -m pytest -q && ./.venv/Scripts/python.exe -m ruff check .`
Expected: all pass, ruff clean

- [ ] **Step 7: Commit**

```bash
git add app/api/routes_agent.py app/main.py tests/test_agent_api.py
git commit -m "feat: /agent and /agent/stream endpoints"
```

---

### Task 7: Streamlit "Agent mode" toggle

**Files:**
- Modify: `ui/streamlit_app.py`

**Interfaces:**
- Consumes: the `/agent/stream` endpoint (Task 6). No new exports.

- [ ] **Step 1: Add an agent-mode toggle and endpoint switch** — in `ui/streamlit_app.py` sidebar, after the `stream` checkbox add:

```python
    agent_mode = st.checkbox("Agent mode (route + self-correct)", value=False)
```

- [ ] **Step 2: Point the stream at the agent endpoint and render step events** — change `stream_events` to accept the path and parse `step` events:

```python
def stream_events(question: str, path: str):
    with httpx.stream(
        "POST",
        f"{api_url}{path}",
        json={"question": question, "top_k": top_k},
        headers=_headers(),
        timeout=120.0,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line.strip():
                yield json.loads(line)
```

In the streaming branch of the chat handler, choose the path and surface `step` events as status text:

```python
                path = "/agent/stream" if agent_mode else "/chat/stream"
                placeholder = st.empty()
                status = st.empty()
                answer, sources, usage, latency = "", [], {}, None
                for event in stream_events(question, path):
                    kind = event.get("event")
                    if kind == "step":
                        status.caption(f"agent: {event.get('node')}")
                    elif kind == "sources":
                        sources = event.get("sources", [])
                    elif kind == "token":
                        answer += event.get("token", "")
                        placeholder.markdown(answer + "█")
                    elif kind == "done":
                        answer = event.get("answer", answer)
                        usage = event.get("usage", {})
                        latency = event.get("latency_ms", latency)
                    elif kind == "error":
                        st.error(event.get("detail", "stream error"))
                status.empty()
                placeholder.markdown(answer)
```

(Keep the existing non-stream branch unchanged; agent mode implies streaming.)

- [ ] **Step 3: Verify the file still compiles**

Run: `./.venv/Scripts/python.exe -m py_compile ui/streamlit_app.py`
Expected: no output (success)

- [ ] **Step 4: Commit**

```bash
git add ui/streamlit_app.py
git commit -m "feat: agent mode toggle in the Streamlit demo UI"
```

---

## Notes for the implementer

- The agent's `/agent` path is intentionally slower/costlier than `/chat` (extra router/grader/rewrite LLM calls); the `usage` field quantifies it.
- Do not change `/chat` — it stays the fast single-pass path.
- After Task 6, the full suite should be green and ruff clean before moving on; the agent adds ~20 tests.
