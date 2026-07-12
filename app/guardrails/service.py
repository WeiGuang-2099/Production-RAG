"""Guardrail orchestration, gated by GUARDRAILS_ENABLED."""
from __future__ import annotations

import logging

from app.config import get_settings
from app.guardrails.injection import detect_injection
from app.guardrails.pii import redact_pii
from app.guardrails.toxicity import detect_toxicity

logger = logging.getLogger(__name__)


def check_input(question: str) -> list[str]:
    """Injection rule labels if the input should be blocked, else []."""
    if not get_settings().GUARDRAILS_ENABLED:
        return []
    return detect_injection(question)


def apply_output(answer: str) -> dict:
    """Redact PII and flag toxicity in an answer."""
    if not get_settings().GUARDRAILS_ENABLED:
        return {"answer": answer, "pii_redacted": [], "flags": []}
    redacted, pii = redact_pii(answer)
    return {"answer": redacted, "pii_redacted": pii, "flags": detect_toxicity(redacted)}


def screen_history(history: list[dict]) -> list[dict]:
    """Drop history turns that match injection heuristics.

    History is untrusted client input that flows into the condense prompt.
    Matching turns are dropped (not blocked): a poisoned old message must
    not permanently brick the conversation.
    """
    if not get_settings().GUARDRAILS_ENABLED:
        return history
    kept: list[dict] = []
    for turn in history:
        patterns = detect_injection(turn.get("content", ""))
        if patterns:
            logger.warning(
                "guardrails_history_dropped: role=%s patterns=%s", turn.get("role"), patterns
            )
            continue
        kept.append(turn)
    return kept
