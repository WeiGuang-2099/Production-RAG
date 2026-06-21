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
