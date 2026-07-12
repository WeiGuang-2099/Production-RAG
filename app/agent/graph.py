"""LangGraph wiring for the Corrective-RAG agent."""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator

from langgraph.graph import END, StateGraph

from app.agent import nodes
from app.agent.state import AgentState
from app.config import get_settings
from app.core.condense import CondenseResult, attach_condense, condense_question

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


def run_agent(
    question: str,
    top_k: int | None = None,
    sources: list[str] | None = None,
    history: list[dict] | None = None,
) -> dict:
    settings = get_settings()
    start = time.time()
    cq = (
        condense_question(question, history)
        if history
        else CondenseResult(question=question, applied=False)
    )
    if cq.applied:
        logger.info("condensed: %s -> %s", question[:80], cq.question[:80])
        question = cq.question
    init = {"question": question, "query": question, "top_k": top_k or settings.TOP_K,
            "attempts": 0, "scope_sources": sources or []}
    final = _get_compiled().invoke(init)
    return attach_condense({
        "answer": final.get("answer", ""),
        "sources": final.get("sources", []),
        "latency_ms": (time.time() - start) * 1000,
        "usage": final.get("usage", {}),
        "route": final.get("route", ""),
        "attempts": final.get("attempts", 0),
    }, cq)


async def stream_agent(
    question: str,
    top_k: int | None = None,
    sources: list[str] | None = None,
    history: list[dict] | None = None,
) -> AsyncIterator[dict]:
    settings = get_settings()
    start = time.time()
    if history:
        cq = await asyncio.to_thread(condense_question, question, history)
    else:
        cq = CondenseResult(question=question, applied=False)
    if cq.applied:
        yield {"event": "condensed", "condensed_question": cq.question}
        question = cq.question
    init = {"question": question, "query": question, "top_k": top_k or settings.TOP_K,
            "attempts": 0, "scope_sources": sources or []}
    final: dict = {}
    async for update in _get_compiled().astream(init, stream_mode="updates"):
        for node_name, partial in update.items():
            final.update(partial or {})
            yield {"event": "step", "node": node_name}
    yield {"event": "sources", "sources": final.get("sources", [])}
    done = attach_condense({
        "answer": final.get("answer", ""),
        "usage": final.get("usage", {}),
        "route": final.get("route", ""),
        "attempts": final.get("attempts", 0),
        "latency_ms": (time.time() - start) * 1000,
    }, cq)
    yield {"event": "done", **done}
