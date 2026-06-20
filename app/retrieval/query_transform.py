"""Query transformation before retrieval.

Two opt-in strategies (off by default via QUERY_TRANSFORM=none):

- ``multi_query`` — ask the LLM for a few paraphrases of the question and
  retrieve for all of them, fusing the results. Helps when the user's wording
  doesn't match the corpus wording.
- ``hyde`` — Hypothetical Document Embeddings: ask the LLM to draft a plausible
  answer passage and embed *that* instead of the bare question, since a
  passage is closer in embedding space to real answer passages than a short
  question is.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_MAX_QUERIES = 5  # original + up to 4 paraphrases

MULTI_QUERY_PROMPT = (
    "You are helping improve document retrieval. Generate 3 alternative "
    "phrasings of the question below that a relevant document might use. "
    "Return one phrasing per line, no numbering.\n\nQuestion: {question}"
)

HYDE_PROMPT = (
    "Write a short, factual passage (2-3 sentences) that would directly answer "
    "the question below, as if excerpted from a relevant document. Do not say "
    "you are unsure.\n\nQuestion: {question}"
)


def parse_multi_queries(text: str) -> list[str]:
    """Split an LLM response into clean query strings, dropping list markers."""
    queries: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip()
        if cleaned:
            queries.append(cleaned)
    return queries


def build_queries(question: str, mode: str, llm) -> list[str]:
    """Return the list of query strings to retrieve with for ``mode``.

    ``none`` / unknown modes are a passthrough and never call the LLM.
    """
    if mode == "multi_query":
        response = llm.invoke(MULTI_QUERY_PROMPT.format(question=question))
        content = getattr(response, "content", "") or ""
        queries = [question]
        for q in parse_multi_queries(content):
            if q not in queries:
                queries.append(q)
        return queries[:_MAX_QUERIES]

    if mode == "hyde":
        response = llm.invoke(HYDE_PROMPT.format(question=question))
        content = (getattr(response, "content", "") or "").strip()
        return [content] if content else [question]

    return [question]
