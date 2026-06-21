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
