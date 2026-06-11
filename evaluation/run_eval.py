"""Run RAGAS evaluation metrics against the RAG system.

Usage:
    python evaluation/run_eval.py                          # full 48-question eval
    python evaluation/run_eval.py --subset 5               # smoke test on 5
    python evaluation/run_eval.py --types factual,multi_hop
    python evaluation/run_eval.py --label baseline         # tag the saved report
    python evaluation/run_eval.py --no-save                # skip writing report file
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NON_METRIC_COLS = {"user_input", "response", "reference", "retrieved_contexts"}


def _load_ragas() -> dict:
    """Lazy import of ragas so --help and arg parsing work without it installed."""
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    return {
        "evaluate": evaluate,
        "EvaluationDataset": EvaluationDataset,
        "SingleTurnSample": SingleTurnSample,
        "metrics": [faithfulness, answer_relevancy, context_recall, context_precision],
    }


def _load_query_pipeline():
    """Lazy import of the app so --help works without app deps installed."""
    from app.core.pipeline import query_pipeline
    return query_pipeline


# ── CLI ────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run RAGAS evaluation on the project's eval dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=None, help="Path to eval_dataset.json (default: evaluation/eval_dataset.json)")
    p.add_argument("--subset", type=int, default=None, help="Run only the first N questions (smoke test)")
    p.add_argument("--types", default=None, help="Comma-separated question types to include, e.g. factual,multi_hop")
    p.add_argument("--label", default="baseline", help="Label for this run, recorded in the report (e.g. baseline, no-rerank)")
    p.add_argument("--output", default=None, help="Path to save JSON report (default: results/<ts>_<label>.json)")
    p.add_argument("--no-save", action="store_true", help="Skip saving the JSON report")
    return p.parse_args()


# ── Data loading / filtering ───────────────────────────

def load_dataset(path: str | None) -> list[dict]:
    resolved = Path(path) if path else Path(__file__).resolve().parent / "eval_dataset.json"
    with open(resolved) as f:
        return json.load(f)


def filter_dataset(dataset: list[dict], subset: int | None, types: str | None) -> list[dict]:
    if types:
        wanted = {t.strip() for t in types.split(",") if t.strip()}
        dataset = [d for d in dataset if d.get("type") in wanted]
    if subset is not None and subset > 0:
        dataset = dataset[:subset]
    return dataset


# ── Pipeline execution ─────────────────────────────────

def run_pipeline_for_items(items: list[dict], query_fn) -> list[dict]:
    """Invoke query_fn per item, recording latency and capturing failures."""
    records: list[dict] = []
    total = len(items)
    for i, item in enumerate(items, 1):
        print(f"[{i:3d}/{total}] {item['id']} ({item.get('type','?')}) ", end="", flush=True)
        t0 = time.time()
        try:
            result = query_fn(item["question"])
            latency_ms = (time.time() - t0) * 1000
            records.append({"item": item, "result": result, "latency_ms": latency_ms, "error": None})
            print(f"... {latency_ms:.0f} ms")
        except Exception as exc:
            latency_ms = (time.time() - t0) * 1000
            records.append({"item": item, "result": None, "latency_ms": latency_ms, "error": str(exc)})
            print(f"... FAILED ({exc})")
    return records


def build_samples(records: list[dict], SampleCls) -> list:
    samples = []
    for rec in records:
        if rec["error"] or rec["result"] is None:
            continue
        item, result = rec["item"], rec["result"]
        samples.append(
            SampleCls(
                user_input=item["question"],
                response=result["answer"],
                reference=item["ground_truth"],
                retrieved_contexts=[s["content"] for s in result.get("sources", [])],
            )
        )
    return samples


# ── Aggregation ────────────────────────────────────────

def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))


def aggregate_scores(df, records: list[dict]) -> dict:
    """Build per-item score map + aggregate by type and difficulty."""
    metric_cols = [c for c in df.columns if c not in NON_METRIC_COLS]
    successful = [r for r in records if r["error"] is None]

    per_item: dict[str, dict[str, float]] = {}
    for i, rec in enumerate(successful):
        scores: dict[str, float] = {}
        for col in metric_cols:
            v = df.iloc[i][col]
            if _is_number(v):
                scores[col] = float(v)
        per_item[rec["item"]["id"]] = scores

    def bucket(key_fn) -> dict[str, dict[str, float]]:
        b: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for rec in successful:
            key = key_fn(rec["item"])
            for col, v in per_item.get(rec["item"]["id"], {}).items():
                b[key][col].append(v)
        return {k: {col: mean(vs) for col, vs in cols.items() if vs} for k, cols in b.items()}

    overall = {}
    for col in metric_cols:
        vals = [s[col] for s in per_item.values() if col in s]
        if vals:
            overall[col] = mean(vals)

    return {
        "metrics": metric_cols,
        "overall": overall,
        "by_type": bucket(lambda it: it.get("type", "unknown")),
        "by_difficulty": bucket(lambda it: it.get("difficulty", "unknown")),
        "per_item": per_item,
    }


def latency_stats(records: list[dict]) -> dict:
    lats = [r["latency_ms"] for r in records if r["error"] is None]
    if not lats:
        return {}
    lats_sorted = sorted(lats)
    p95_idx = max(0, int(len(lats) * 0.95) - 1)
    return {
        "count": len(lats),
        "mean_ms": mean(lats),
        "p50_ms": median(lats),
        "p95_ms": lats_sorted[p95_idx],
        "max_ms": max(lats),
    }


# ── Reporting ──────────────────────────────────────────

def _fmt_value(v: float | None, width: int = 14) -> str:
    if v is None or not _is_number(v):
        return f"{'-':>{width}}"
    return f"{v:>{width}.3f}"


def print_report(agg: dict, lat: dict, records: list[dict], label: str) -> None:
    total = len(records)
    failed = sum(1 for r in records if r["error"])
    metrics = agg["metrics"]

    print()
    print("=" * 72)
    print(f"Evaluation report: {label}")
    print("=" * 72)
    print(f"Questions: {total} total | {total - failed} succeeded | {failed} failed")
    if lat:
        print(
            f"Latency:   mean={lat['mean_ms']:.0f} ms  p50={lat['p50_ms']:.0f} ms  "
            f"p95={lat['p95_ms']:.0f} ms  max={lat['max_ms']:.0f} ms"
        )
    print()

    if not metrics:
        print("No metric scores produced.")
        return

    print("Overall:")
    for m in metrics:
        v = agg["overall"].get(m)
        print(f"  {m:<22s} {_fmt_value(v, 8).strip():>8s}")
    print()

    succ = [r for r in records if r["error"] is None]

    def _print_bucket(title: str, bucket_field: str, ordered_keys: list[str], data: dict) -> None:
        if not data:
            return
        counts: dict[str, int] = defaultdict(int)
        for r in succ:
            counts[r["item"].get(bucket_field, "unknown")] += 1

        print(f"{title}:")
        print(f"  {'bucket':<15s} " + " ".join(f"{m[:14]:>14s}" for m in metrics) + f" {'n':>5s}")
        print(f"  {'-'*15} " + " ".join("-" * 14 for _ in metrics) + f" {'-'*5}")
        for k in ordered_keys:
            if k not in data:
                continue
            row = data[k]
            vals = " ".join(_fmt_value(row.get(m), 14) for m in metrics)
            print(f"  {k:<15s} {vals} {counts.get(k, 0):>5d}")
        print()

    type_order = ["factual", "multi_hop", "comparative", "numerical", "unanswerable", "long_tail"]
    diff_order = ["easy", "medium", "hard"]
    type_keys = type_order + [k for k in agg["by_type"] if k not in type_order]
    diff_keys = diff_order + [k for k in agg["by_difficulty"] if k not in diff_order]

    _print_bucket("By question type", "type", type_keys, agg["by_type"])
    _print_bucket("By difficulty", "difficulty", diff_keys, agg["by_difficulty"])


def save_report(path: Path, label: str, args: argparse.Namespace, agg: dict, lat: dict, records: list[dict]) -> None:
    report = {
        "label": label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": {"subset": args.subset, "types": args.types, "dataset": args.dataset},
        "questions": {
            "total": len(records),
            "succeeded": sum(1 for r in records if r["error"] is None),
            "failed": sum(1 for r in records if r["error"]),
        },
        "latency": lat,
        "overall": agg["overall"],
        "by_type": agg["by_type"],
        "by_difficulty": agg["by_difficulty"],
        "per_item": [
            {
                "id": rec["item"]["id"],
                "type": rec["item"].get("type"),
                "difficulty": rec["item"].get("difficulty"),
                "source_papers": rec["item"].get("source_papers"),
                "latency_ms": rec["latency_ms"],
                "error": rec["error"],
                "scores": agg["per_item"].get(rec["item"]["id"], {}),
            }
            for rec in records
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {path}")


# ── Entry point ────────────────────────────────────────

def main() -> int:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    dataset = filter_dataset(dataset, args.subset, args.types)
    if not dataset:
        print("No questions selected.", file=sys.stderr)
        return 1

    ragas = _load_ragas()
    query_fn = _load_query_pipeline()

    print(f"Running {len(dataset)} questions [label={args.label}]")
    print()

    records = run_pipeline_for_items(dataset, query_fn)
    samples = build_samples(records, ragas["SingleTurnSample"])
    if not samples:
        print("All queries failed; nothing to evaluate.", file=sys.stderr)
        return 1

    print()
    print(f"Evaluating {len(samples)} successful samples with RAGAS ({len(ragas['metrics'])} metrics)...")
    eval_dataset = ragas["EvaluationDataset"](samples=samples)
    results = ragas["evaluate"](eval_dataset, metrics=ragas["metrics"])
    df = results.to_pandas()

    agg = aggregate_scores(df, records)
    lat = latency_stats(records)
    print_report(agg, lat, records, args.label)

    if not args.no_save:
        if args.output:
            out = Path(args.output)
        else:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out = Path(__file__).resolve().parent / "results" / f"{ts}_{args.label}.json"
        save_report(out, args.label, args, agg, lat, records)

    return 0


if __name__ == "__main__":
    sys.exit(main())
