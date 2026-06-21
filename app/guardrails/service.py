"""Guardrail orchestration, gated by GUARDRAILS_ENABLED."""
from __future__ import annotations

from app.config import get_settings
from app.guardrails.injection import detect_injection
from app.guardrails.pii import redact_pii
from app.guardrails.toxicity import detect_toxicity


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
