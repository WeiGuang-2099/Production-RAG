"""Retrieval ablation: quantify where retrieval quality comes from.

Runs the project's retrieval over the eval dataset under four cumulative
configurations and reports deterministic, LLM-judge-free retrieval metrics
(recall@k / MRR / hit@k). This is the cheap evidence that each component
(BM25, reranker, graph) actually earns its place.

    stage       RETRIEVAL_MODE  RERANKER_PROVIDER  GRAPH_EXTRACTOR
    ---------   --------------  -----------------  ---------------
    baseline    dense           none               none
    +bm25       hybrid          none               none
    +rerank     hybrid          cohere             none
    +graph      hybrid          cohere             llm (uses the graph built at ingest)

Because it never calls the generation LLM, an ablation over the full 48
questions costs only embeddings (+ Cohere rerank for the last two stages),
not 4x48 chat completions.

Usage:
    python evaluation/run_ablation.py                  # all stages, full dataset
    python evaluation/run_ablation.py --subset 10      # quick smoke
    python evaluation/run_ablation.py --k 5            # recall@5 / hit@5

A failed rerank call does not fail the question: the pipeline falls back to the
unranked candidates with only a log warning, which would quietly turn +rerank
into +bm25. Each stage therefore counts failed rerank calls (the rerank_failures
column); any failure makes the script exit 2 after saving, unless
--allow-rerank-fallback is passed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import (  # noqa: E402
    aggregate_retrieval_metrics,
    render_markdown_table,
    retrieved_slugs,
)

# Each stage layers one component on top of the previous one. Because every
# stage mutates os.environ, the +graph stage must EXPLICITLY re-enable graph
# retrieval — prior stages set GRAPH_EXTRACTOR=none, so omitting it here would
# silently make +graph identical to +rerank. "llm" matches how the eval corpus
# is ingested; graph retrieval reads the prebuilt graph (no re-extraction, so
# no LLM cost).
STAGES: list[tuple[str, dict[str, str]]] = [
    ("baseline", {"RETRIEVAL_MODE": "dense", "RERANKER_PROVIDER": "none", "GRAPH_EXTRACTOR": "none"}),
    ("+bm25", {"RETRIEVAL_MODE": "hybrid", "RERANKER_PROVIDER": "none", "GRAPH_EXTRACTOR": "none"}),
    ("+rerank", {"RETRIEVAL_MODE": "hybrid", "RERANKER_PROVIDER": "cohere", "GRAPH_EXTRACTOR": "none"}),
    ("+graph", {"RETRIEVAL_MODE": "hybrid", "RERANKER_PROVIDER": "cohere", "GRAPH_EXTRACTOR": "llm"}),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrieval ablation over the eval dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=None, help="Path to eval_dataset.json")
    p.add_argument("--subset", type=int, default=None, help="Run only the first N questions")
    p.add_argument("--k", type=int, default=5, help="Cutoff for recall@k / hit@k (default 5)")
    p.add_argument("--top-k", type=int, default=10, help="Initial retrieval depth before rerank (default 10)")
    p.add_argument("--no-save", action="store_true", help="Skip writing the report")
    p.add_argument("--label", default=None, help="Filename suffix for saved reports (e.g. base6, scale30)")
    p.add_argument("--allow-rerank-fallback", action="store_true",
                   help="Exit 0 even if rerank calls failed (default: exit 2)")
    return p.parse_args(argv)


_rerank_failures = {"count": 0}


def install_rerank_counter() -> None:
    """Count failed rerank calls; the pipeline swallows them and keeps the unranked order."""
    import app.reranker.reranker as reranker

    original = reranker.RerankerService.rerank

    def counted(self, query, documents, top_k=3):
        try:
            return original(self, query, documents, top_k)
        except Exception:
            _rerank_failures["count"] += 1
            raise  # keep the pipeline's fallback path unchanged

    reranker.RerankerService.rerank = counted


def load_dataset(path: str | None, subset: int | None) -> list[dict]:
    resolved = Path(path) if path else Path(__file__).resolve().parent / "eval_dataset.json"
    with open(resolved) as f:
        data = json.load(f)
    if subset:
        data = data[:subset]
    return data


def run_stage(label: str, overrides: dict, dataset: list[dict], k: int, top_k: int) -> dict:
    from app.config import get_settings
    from app.core.factories import clear_caches
    from app.core.pipeline import retrieve_sources

    for key, val in overrides.items():
        os.environ[key] = val
    # Measure recall@k over the full retrieval depth: keep all top_k candidates
    # through rerank so the reranker reorders them without truncating below k
    # (the app keeps RERANK_TOP_K=5 passages, which would cap recall@k for k > 5).
    os.environ["RERANK_TOP_K"] = str(top_k)
    get_settings.cache_clear()
    clear_caches()

    items: list[dict] = []
    latencies: list[float] = []
    _rerank_failures["count"] = 0
    print(f"\n=== stage: {label} ({overrides}) ===")
    for i, item in enumerate(dataset, 1):
        t0 = time.time()
        try:
            sources = retrieve_sources(item["question"], top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i:3d}] {item['id']} FAILED: {exc}")
            sources = []
        latencies.append((time.time() - t0) * 1000)
        items.append(
            {"retrieved": retrieved_slugs(sources), "expected": item.get("source_papers", [])}
        )
    agg = aggregate_retrieval_metrics(items, k)
    agg["stage"] = label
    agg["rerank_failures"] = _rerank_failures["count"]
    if latencies:
        agg["p50_ms"] = round(median(latencies), 1)
        agg["p95_ms"] = round(sorted(latencies)[min(len(latencies) - 1, int(len(latencies) * 0.95))], 1)
        agg["mean_ms"] = round(mean(latencies), 1)
    return agg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = load_dataset(args.dataset, args.subset)
    print(f"Ablation over {len(dataset)} questions (k={args.k}, top_k={args.top_k})")

    install_rerank_counter()
    results = [run_stage(label, ov, dataset, args.k, args.top_k) for label, ov in STAGES]

    columns = ["stage", f"recall@{args.k}", "mrr", f"hit@{args.k}", "p50_ms", "p95_ms", "count",
               "rerank_failures"]
    table = render_markdown_table(results, columns)
    failed = {r["stage"]: r["rerank_failures"] for r in results if r["rerank_failures"]}
    warning = ("**Rerank failed in " + ", ".join(f"{s} ({n} of {len(dataset)} questions)" for s, n in failed.items())
               + "; those questions kept the unranked order.**\n\n") if failed else ""
    print("\nRetrieval ablation (higher is better; latency is retrieval-only):\n")
    print(warning + table)

    if not args.no_save:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = f"_{args.label}" if args.label else ""
        out_dir = Path(__file__).resolve().parent / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{ts}_ablation{suffix}.md").write_text(warning + table + "\n", encoding="utf-8")
        (out_dir / f"{ts}_ablation{suffix}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved: evaluation/results/{ts}_ablation{suffix}.md")

    if failed and not args.allow_rerank_fallback:
        print("Rerank calls failed, so the rerank stages did not measure the reranker. Fix it and rerun, "
              "or pass --allow-rerank-fallback.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
