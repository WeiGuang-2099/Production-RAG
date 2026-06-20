"""Deterministic retrieval-quality metrics.

These complement the RAGAS metrics in ``run_eval.py``. RAGAS needs an LLM
judge (slow + paid); these are pure functions over the *source papers* that
each retrieved chunk came from, so they are cheap, deterministic, and need no
generation calls at all. They answer the core retrieval question: "did we
pull the right documents, and how high did we rank them?"

A chunk's ``metadata["source"]`` is a path like ``./data/papers/attention.pdf``;
its slug (``attention``) is compared against the dataset's ``source_papers``.
"""
from __future__ import annotations

import math
import re

__all__ = [
    "slug_from_source",
    "retrieved_slugs",
    "recall_at_k",
    "reciprocal_rank",
    "hit_at_k",
    "aggregate_retrieval_metrics",
    "render_markdown_table",
]


def slug_from_source(source: str) -> str | None:
    """Map a chunk source path to its paper slug.

    ``./data/papers/attention.pdf`` -> ``attention``. Handles both POSIX and
    Windows separators. Returns ``None`` for an empty source.
    """
    if not source:
        return None
    name = re.split(r"[\\/]", source)[-1]
    stem = name.rsplit(".", 1)[0]
    return stem or None


def retrieved_slugs(sources: list[dict]) -> list[str]:
    """Ordered slugs for a query result's ``sources`` list.

    Missing/None slugs become ``""`` so positions (ranks) are preserved.
    """
    slugs: list[str] = []
    for s in sources:
        meta = s.get("metadata") or {}
        slug = slug_from_source(meta.get("source", ""))
        slugs.append(slug if slug is not None else "")
    return slugs


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Fraction of distinct expected papers present in the top-k retrieved.

    Returns NaN when there are no expected papers (e.g. unanswerable
    questions), so callers can exclude those items from averaging.
    """
    want = set(expected)
    if not want:
        return math.nan
    topk = set(retrieved[:k])
    return sum(1 for e in want if e in topk) / len(want)


def reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    """1 / rank of the first relevant retrieved item (0 if none)."""
    want = set(expected)
    for i, slug in enumerate(retrieved, 1):
        if slug in want:
            return 1.0 / i
    return 0.0


def hit_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """1.0 if any expected paper appears in the top-k, else 0.0."""
    want = set(expected)
    return 1.0 if any(s in want for s in retrieved[:k]) else 0.0


def aggregate_retrieval_metrics(items: list[dict], k: int) -> dict:
    """Average recall@k / MRR / hit@k over items that have expected papers.

    Each item is ``{"retrieved": [slug, ...], "expected": [slug, ...]}``.
    Items with no expected papers are skipped.
    """
    scored = [it for it in items if it.get("expected")]
    if not scored:
        return {"count": 0, f"recall@{k}": math.nan, "mrr": math.nan, f"hit@{k}": math.nan}

    recalls = [recall_at_k(it["retrieved"], it["expected"], k) for it in scored]
    rrs = [reciprocal_rank(it["retrieved"], it["expected"]) for it in scored]
    hits = [hit_at_k(it["retrieved"], it["expected"], k) for it in scored]
    n = len(scored)
    return {
        "count": n,
        f"recall@{k}": sum(recalls) / n,
        "mrr": sum(rrs) / n,
        f"hit@{k}": sum(hits) / n,
    }


def _fmt_cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "-"
        return f"{value:.3f}"
    return str(value)


def render_markdown_table(rows: list[dict], columns: list[str]) -> str:
    """Render rows (list of dicts) as a GitHub-flavored markdown table.

    Floats are formatted to 3 decimals; NaN/None render as ``-``.
    """
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for row in rows:
        cells = [_fmt_cell(row.get(c)) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
