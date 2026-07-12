"""Multi-turn retrieval eval: does condense-question recover follow-up recall?

Three conditions per item, retrieval-only (no generation LLM):

    raw        the follow-up as typed (today's failure mode)
    condensed  condense_question(follow_up, history) output
    oracle     the hand-written standalone reference (upper bound)

Retrieval config is pinned to the system's best-performing ablation stage
(+rerank: hybrid + cohere, graph off) so the numbers compose with the
published tables. Cost: ~18 fast-model calls + ~54 retrievals (embeddings
+ Cohere) — cents. Requires live Qdrant + .env keys, base6 corpus.

Usage:
    python evaluation/run_multiturn.py --k 5 --label base6
    python evaluation/run_multiturn.py --subset 3 --no-save   # smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.condense import condense_question  # noqa: E402
from app.core.pipeline import retrieve_sources  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    aggregate_retrieval_metrics,
    render_markdown_table,
    retrieved_slugs,
)

CONDITIONS = ("raw", "condensed", "oracle")

OVERRIDES = {"RETRIEVAL_MODE": "hybrid", "RERANKER_PROVIDER": "cohere", "GRAPH_EXTRACTOR": "none"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Three-condition multi-turn retrieval eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=None, help="Path to multiturn_dataset.json")
    p.add_argument("--subset", type=int, default=None, help="Run only the first N items")
    p.add_argument("--k", type=int, default=5, help="Cutoff for recall@k / hit@k (default 5)")
    p.add_argument("--top-k", type=int, default=10, help="Retrieval depth before rerank (default 10)")
    p.add_argument("--no-save", action="store_true", help="Skip writing the report")
    p.add_argument("--label", default=None, help="Filename suffix for saved reports")
    return p.parse_args()


def load_dataset(path: str | None, subset: int | None) -> list[dict]:
    resolved = Path(path) if path else Path(__file__).resolve().parent / "multiturn_dataset.json"
    with open(resolved, encoding="utf-8") as f:
        data = json.load(f)
    if subset:
        data = data[:subset]
    return data


def evaluate(dataset: list[dict], k: int, top_k: int) -> dict:
    """Score all three conditions; returns rows + per-item rewrites + condense latency."""
    per_condition: dict[str, list[dict]] = {c: [] for c in CONDITIONS}
    rewrites: list[dict] = []
    condense_ms: list[float] = []

    for i, item in enumerate(dataset, 1):
        t0 = time.time()
        cq = condense_question(item["follow_up"], item["history"])
        condense_ms.append((time.time() - t0) * 1000)
        rewrites.append({
            "id": item["id"],
            "follow_up": item["follow_up"],
            "condensed": cq.question,
            "applied": cq.applied,
            "oracle": item["standalone_reference"],
        })
        questions = {
            "raw": item["follow_up"],
            "condensed": cq.question,
            "oracle": item["standalone_reference"],
        }
        for cond, q in questions.items():
            try:
                sources = retrieve_sources(q, top_k=top_k)
            except Exception as exc:  # noqa: BLE001 — one bad item must not kill the run
                print(f"  [{i:2d}] {item['id']}/{cond} FAILED: {exc}")
                sources = []
            per_condition[cond].append(
                {"retrieved": retrieved_slugs(sources), "expected": item.get("source_papers", [])}
            )

    rows = []
    for cond in CONDITIONS:
        agg = aggregate_retrieval_metrics(per_condition[cond], k)
        agg["condition"] = cond
        rows.append(agg)

    latencies = sorted(condense_ms)
    return {
        "rows": rows,
        "rewrites": rewrites,
        "condense_p50_ms": round(median(condense_ms), 1) if condense_ms else None,
        "condense_p95_ms": (
            round(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))], 1)
            if latencies else None
        ),
    }


def main() -> int:
    args = parse_args()
    dataset = load_dataset(args.dataset, args.subset)

    from app.config import get_settings
    from app.core.factories import clear_caches

    for key, val in OVERRIDES.items():
        os.environ[key] = val
    # Keep the full retrieval depth through rerank (same reasoning as run_ablation)
    os.environ["RERANK_TOP_K"] = str(args.top_k)
    get_settings.cache_clear()
    clear_caches()

    print(f"Multi-turn eval over {len(dataset)} items (k={args.k}, top_k={args.top_k})")
    out = evaluate(dataset, args.k, args.top_k)

    columns = ["condition", f"recall@{args.k}", "mrr", f"hit@{args.k}", "count"]
    table = render_markdown_table(out["rows"], columns)
    print("\nFollow-up retrieval by condition (higher is better):\n")
    print(table)
    print(f"\ncondense latency: p50 {out['condense_p50_ms']} ms, p95 {out['condense_p95_ms']} ms")

    if not args.no_save:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = f"_{args.label}" if args.label else ""
        out_dir = Path(__file__).resolve().parent / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{ts}_multiturn{suffix}.md").write_text(table + "\n", encoding="utf-8")
        (out_dir / f"{ts}_multiturn{suffix}.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8"
        )
        print(f"\nSaved: evaluation/results/{ts}_multiturn{suffix}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
