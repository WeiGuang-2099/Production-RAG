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
from app.core.factories import complete, complete_with_model
from app.core.pipeline import _docs_to_sources, _retrieve_and_rerank
from app.core.prompts import format_context, select_prompt
from app.observability.cost import usage_for

logger = logging.getLogger(__name__)


def route_question(state: dict) -> dict:
    try:
        out = complete(ROUTER_PROMPT.format(question=state["question"]), fast=True)
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
        out = complete(
            GRADER_PROMPT.format(question=state["question"], context=format_context(docs)),
            fast=True,
        )
        return {"relevant": parse_grade(out)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("grade_failed: %s", exc)
        return {"relevant": True}


def rewrite_query(state: dict) -> dict:
    attempts = state.get("attempts", 0) + 1
    fallback = state.get("query") or state["question"]
    try:
        out = complete(REWRITE_PROMPT.format(question=state["question"]), fast=True).strip()
        return {"query": out or fallback, "attempts": attempts}
    except Exception as exc:  # noqa: BLE001
        logger.warning("rewrite_failed: %s", exc)
        return {"query": fallback, "attempts": attempts}


def _generate_with(prompt_text: str) -> tuple[str, dict]:
    answer, model = complete_with_model(prompt_text)
    usage = usage_for(prompt_text, answer, model)
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
