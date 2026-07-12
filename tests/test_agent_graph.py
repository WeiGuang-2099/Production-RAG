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


def test_run_agent_condenses_before_the_graph():
    from unittest.mock import patch

    from app.core.condense import CondenseResult

    cond_usage = {"input_tokens": 20, "output_tokens": 6, "cost_usd": 0.00001, "model": "gpt-4o-mini"}
    with patch("app.agent.graph.condense_question",
               return_value=CondenseResult("standalone?", True, cond_usage)), \
         patch("app.agent.graph._get_compiled") as mock_compiled:
        mock_compiled.return_value.invoke.return_value = {
            "answer": "A", "sources": [], "usage": {"cost_usd": 0.001},
            "route": "retrieve", "attempts": 0,
        }
        result = graph_mod.run_agent("f?", top_k=5, history=[{"role": "user", "content": "x"}])

    init = mock_compiled.return_value.invoke.call_args.args[0]
    assert init["question"] == "standalone?"
    assert init["query"] == "standalone?"
    assert result["condensed_question"] == "standalone?"
    assert result["usage"]["condense"] == cond_usage


def test_run_agent_without_history_skips_condense():
    from unittest.mock import patch

    with patch("app.agent.graph.condense_question") as mock_cond, \
         patch("app.agent.graph._get_compiled") as mock_compiled:
        mock_compiled.return_value.invoke.return_value = {"answer": "A", "sources": [], "usage": {}}
        result = graph_mod.run_agent("q")
    mock_cond.assert_not_called()
    assert result["condensed_question"] is None


async def test_stream_agent_emits_condensed_event_first():
    from unittest.mock import patch

    from app.core.condense import CondenseResult

    async def fake_astream(init, stream_mode="updates"):
        yield {"generate": {"answer": "A", "usage": {}}}

    with patch("app.agent.graph.condense_question",
               return_value=CondenseResult("standalone?", True)), \
         patch("app.agent.graph._get_compiled") as mock_compiled:
        mock_compiled.return_value.astream = fake_astream
        events = [e async for e in graph_mod.stream_agent("f?", history=[{"role": "user", "content": "x"}])]

    kinds = [e["event"] for e in events]
    assert kinds[0] == "condensed"
    assert events[0]["condensed_question"] == "standalone?"
    assert events[-1]["event"] == "done"
    assert events[-1]["condensed_question"] == "standalone?"
