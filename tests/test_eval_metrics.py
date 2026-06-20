import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import (
    aggregate_retrieval_metrics,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    render_markdown_table,
    retrieved_slugs,
    slug_from_source,
)

# ── slug_from_source ───────────────────────────────────

def test_slug_from_posix_path():
    assert slug_from_source("./data/papers/attention.pdf") == "attention"


def test_slug_from_windows_path():
    assert slug_from_source(r"C:\data\papers\bert.pdf") == "bert"


def test_slug_from_bare_name():
    assert slug_from_source("gpt3.pdf") == "gpt3"


def test_slug_from_empty_is_none():
    assert slug_from_source("") is None
    assert slug_from_source("graph") == "graph"  # non-paper sources pass through


# ── retrieved_slugs ────────────────────────────────────

def test_retrieved_slugs_preserves_order():
    sources = [
        {"content": "x", "metadata": {"source": "./data/papers/rag.pdf"}},
        {"content": "y", "metadata": {"source": "./data/papers/lora.pdf"}},
    ]
    assert retrieved_slugs(sources) == ["rag", "lora"]


def test_retrieved_slugs_missing_metadata():
    assert retrieved_slugs([{"content": "x"}]) == [""]


# ── recall_at_k ────────────────────────────────────────

def test_recall_full():
    assert recall_at_k(["attention", "bert"], ["attention", "bert"], k=5) == 1.0


def test_recall_partial():
    assert recall_at_k(["attention", "x"], ["attention", "bert"], k=5) == 0.5


def test_recall_respects_k_cutoff():
    # bert is at rank 3 but k=2, so it does not count
    assert recall_at_k(["attention", "x", "bert"], ["attention", "bert"], k=2) == 0.5


def test_recall_empty_expected_is_nan():
    assert math.isnan(recall_at_k(["attention"], [], k=5))


# ── reciprocal_rank ────────────────────────────────────

def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["bert", "x"], ["bert"]) == 1.0


def test_reciprocal_rank_second_position():
    assert reciprocal_rank(["x", "bert"], ["bert"]) == 0.5


def test_reciprocal_rank_not_found():
    assert reciprocal_rank(["x", "y"], ["bert"]) == 0.0


# ── hit_at_k ───────────────────────────────────────────

def test_hit_at_k_true():
    assert hit_at_k(["x", "bert"], ["bert"], k=2) == 1.0


def test_hit_at_k_false_due_to_cutoff():
    assert hit_at_k(["x", "y", "bert"], ["bert"], k=2) == 0.0


# ── aggregation ────────────────────────────────────────

def test_aggregate_skips_items_without_expected():
    items = [
        {"retrieved": ["attention", "bert"], "expected": ["attention"]},
        {"retrieved": ["x"], "expected": []},  # unanswerable: no ground-truth papers
    ]
    agg = aggregate_retrieval_metrics(items, k=5)
    assert agg["count"] == 1
    assert agg["recall@5"] == 1.0
    assert agg["mrr"] == 1.0
    assert agg["hit@5"] == 1.0


def test_aggregate_averages_across_items():
    items = [
        {"retrieved": ["attention"], "expected": ["attention"]},  # recall 1.0, rr 1.0
        {"retrieved": ["x", "bert"], "expected": ["bert"]},        # recall 1.0, rr 0.5
    ]
    agg = aggregate_retrieval_metrics(items, k=5)
    assert agg["count"] == 2
    assert agg["mrr"] == 0.75


# ── markdown rendering ─────────────────────────────────

def test_render_markdown_table_formats_floats_and_nan():
    rows = [{"stage": "baseline", "recall@5": 0.66666, "mrr": float("nan")}]
    out = render_markdown_table(rows, ["stage", "recall@5", "mrr"])
    lines = out.splitlines()
    assert lines[0] == "| stage | recall@5 | mrr |"
    assert set(lines[1]) <= {"|", "-", " "}  # separator row
    assert "0.667" in out
    assert "| baseline |" in out
    assert "nan" not in out.lower()  # NaN must render as a dash, not "nan"


def test_render_markdown_table_keeps_integers_and_strings():
    rows = [{"stage": "x", "count": 42}]
    out = render_markdown_table(rows, ["stage", "count"])
    assert "| x | 42 |" in out
