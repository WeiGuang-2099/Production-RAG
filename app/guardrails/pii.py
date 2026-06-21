"""Best-effort PII redaction. Order matters: structured/broad first, phone last."""
from __future__ import annotations

import re

_RULES = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[REDACTED_EMAIL]"),
    ("credit_card", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"), "[REDACTED_CC]"),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    ("ipv4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED_IP]"),
    ("phone", re.compile(r"(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"), "[REDACTED_PHONE]"),
]


def redact_pii(text: str) -> tuple[str, list[str]]:
    found: list[str] = []
    for label, pattern, replacement in _RULES:
        if pattern.search(text):
            found.append(label)
            text = pattern.sub(replacement, text)
    return text, sorted(set(found))
