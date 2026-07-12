"""Condense a follow-up question + chat history into a standalone question.

The standalone rewrite drives the ENTIRE single-turn path (cache, retrieval,
generation) — generation never sees the history, so the grounded
cite-or-refuse contract stays single-turn and auditable. History is
untrusted client input: turns are trimmed to configured caps and screened
by the injection guardrails before entering the condense prompt. A broken
condense step must never take down chat — any failure falls back to the
raw question.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import get_settings
from app.core.factories import complete_with_model
from app.guardrails.service import screen_history
from app.observability.cost import usage_for

logger = logging.getLogger(__name__)

_CONDENSE_PROMPT = """Given the conversation history and a follow-up question, rewrite the \
follow-up into a single self-contained question that can be understood without the history. \
Resolve pronouns and references. If the follow-up is already self-contained, return it \
unchanged. Output ONLY the rewritten question on one line - no explanations, no quotes.

Conversation:
{transcript}

Follow-up question: {question}

Standalone question:"""


@dataclass
class CondenseResult:
    question: str
    applied: bool
    usage: dict | None = None


def trim_history(history: list[dict], max_turns: int, max_chars: int) -> list[dict]:
    """Silently cap history to the last max_turns turns, max_chars per turn."""
    trimmed = history[-max_turns:] if max_turns > 0 else []
    return [
        {"role": t.get("role", ""), "content": (t.get("content") or "")[:max_chars]}
        for t in trimmed
    ]


def condense_question(question: str, history: list[dict]) -> CondenseResult:
    """Rewrite a follow-up into a standalone question using the history.

    Passthrough (applied=False) when the surviving history is empty or the
    kill switch is off; falls back to the raw question on any LLM failure
    or blank output.
    """
    settings = get_settings()
    turns = trim_history(
        history, settings.CHAT_HISTORY_MAX_TURNS, settings.CHAT_HISTORY_MAX_TURN_CHARS
    )
    turns = screen_history(turns)
    if not turns or not settings.HISTORY_CONDENSE_ENABLED:
        return CondenseResult(question=question, applied=False)

    transcript = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
    prompt = _CONDENSE_PROMPT.format(transcript=transcript, question=question)
    try:
        text, model = complete_with_model(prompt, fast=True)
    except Exception as exc:  # noqa: BLE001 — condense must never take down chat
        logger.warning("condense_failed: %s — using the raw question", exc)
        return CondenseResult(question=question, applied=False)

    lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    standalone = lines[0].strip("\"'") if lines else ""
    if not standalone:
        logger.warning("condense_empty_output — using the raw question")
        return CondenseResult(question=question, applied=False)
    return CondenseResult(
        question=standalone, applied=True, usage=usage_for(prompt, standalone, model)
    )


def attach_condense(result: dict, cq: CondenseResult) -> dict:
    """Add the transparency field and merge the condense cost into usage.

    The input result dict is what gets cached (conversation-free); the
    returned dict is what goes back to the caller for THIS request.
    """
    out = {**result, "condensed_question": cq.question if cq.applied else None}
    if cq.applied and cq.usage:
        usage = {**result.get("usage", {}), "condense": cq.usage}
        usage["cost_usd"] = round(usage.get("cost_usd", 0.0) + cq.usage["cost_usd"], 6)
        out["usage"] = usage
    return out
