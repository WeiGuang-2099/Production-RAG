"""Answer-contract evaluation: does the system answer what it can and refuse what it cannot?

RAGAS scores a refusal as zero whether it is right or wrong, so it cannot tell a correct
"not in the documents" from a missed answer. This harness measures the contract directly.
For every question (the 48-question set plus extra near-miss unanswerable items) it runs the
real pipeline several times and saves, per run, the answer text, the cited [n] numbers, the
retrieved passages, latency and the cost of every model and rerank call. An LLM judge then
labels each response:

- stance: refusal / partial (answers part, says the rest is not covered) / answer;
- verdict for answerable items: correct / partially_correct / incorrect against the reference;
- fabricated for unanswerable items: did it assert an answer the documents do not support?

and, for answerable items that were refused or only partly answered, whether the retrieved
passages held the evidence (to split policy refusals from retrieval misses).

Usage:
    python evaluation/run_contract.py --label baseline --runs 3
    python evaluation/run_contract.py --mode agent --label agent --runs 3
    python evaluation/run_contract.py --subset 5 --runs 1 --label smoke
    python evaluation/run_contract.py --rejudge results/<file>.json --judge-model gpt-4o --label judge-check

Config comes from .env like the app; override per run with env vars, e.g.
    LLM_TEMPERATURE=default python evaluation/run_contract.py --label temperature-default

A failed rerank call does not fail the question: the pipeline logs a warning and answers from
the unranked passages. Such a run no longer measures the configured pipeline, so the harness
counts every failed call per answer and, by default, stops starting new questions at the first
one, skips judging, saves the raw records as <stamp>_contract_<label>_aborted.json and exits 2.
--allow-rerank-fallback keeps going instead; the report then states how many answers fell back.
"""
from __future__ import annotations

import argparse
import contextvars
import hashlib
import json
import math
import re
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HERE = Path(__file__).resolve().parent
REFUSAL = "I cannot answer this from the provided documents."
# Cohere list price for rerank-v3.5: $2.00 per 1,000 searches (a search reranks up to 100 docs).
RERANK_USD_PER_SEARCH = 0.002
# text-embedding-3-small: $0.02 per 1M tokens (one query embedding per retrieval).
EMBED_USD_PER_TOKEN = 0.02 / 1_000_000


# ── Pure helpers (unit-tested) ─────────────────────────

def is_exact_refusal(answer: str) -> bool:
    """True when the answer is the system's fixed refusal sentence (quotes/case/space ignored)."""
    norm = (answer or "").strip().strip("\"'").strip().rstrip(".").lower()
    return norm == REFUSAL.rstrip(".").lower()


def cited_numbers(answer: str) -> list[int]:
    """[n] citation numbers in order of appearance, e.g. "[1] and [2][3]" -> [1, 2, 3]."""
    return [int(n) for n in re.findall(r"\[(\d+)\]", answer or "")]


def citations_in_range(cited: list[int], n_sources: int) -> bool:
    """Every cited number points at a returned source (vacuously true with no citations)."""
    return all(1 <= n <= n_sources for n in cited)


def parse_judge_json(text: str) -> dict | None:
    """Extract the first JSON object from a judge reply (tolerates code fences and prose)."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100); 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(q / 100 * len(ordered))))
    return ordered[rank - 1]


def answered(judge: dict) -> bool:
    return judge.get("stance") in ("answer", "partial")


def summarize_run(records: list[dict]) -> dict:
    """Contract metrics for one run's judged records (errors excluded)."""
    ok = [r for r in records if not r.get("error") and r.get("judge")]
    una = [r for r in ok if r["type"] == "unanswerable"]
    ans = [r for r in ok if r["type"] != "unanswerable"]

    def rate(rows, pred):
        return round(mean(1.0 if pred(r) else 0.0 for r in rows), 4) if rows else None

    answered_rows = [r for r in ans if answered(r["judge"])]
    return {
        "n_unanswerable": len(una),
        "n_answerable": len(ans),
        "unanswerable_declined": rate(una, lambda r: not r["judge"].get("fabricated")),
        "unanswerable_exact_refusal": rate(una, lambda r: r["exact_refusal"]),
        "answerable_refused": rate(ans, lambda r: r["judge"].get("stance") == "refusal"),
        "answerable_partial": rate(ans, lambda r: r["judge"].get("stance") == "partial"),
        "answerable_correct": rate(ans, lambda r: answered(r["judge"]) and r["judge"].get("verdict") == "correct"),
        "answerable_partially_correct": rate(
            ans, lambda r: answered(r["judge"]) and r["judge"].get("verdict") == "partially_correct"),
        "answerable_incorrect": rate(ans, lambda r: answered(r["judge"]) and r["judge"].get("verdict") == "incorrect"),
        "answers_with_citation": rate(answered_rows, lambda r: len(r["cited"]) > 0),
        "citations_in_range": rate(answered_rows, lambda r: r["citations_in_range"]),
        "latency_p50_ms": round(percentile([r["latency_ms"] for r in ok], 50), 1),
        "latency_p95_ms": round(percentile([r["latency_ms"] for r in ok], 95), 1),
        "cost_per_1k_questions_usd": round(mean(r["cost"]["total_usd"] for r in ok) * 1000, 3) if ok else None,
    }


def flip_rate(records: list[dict]) -> float | None:
    """Share of questions whose answered-vs-refused outcome differs across runs."""
    outcomes: dict[str, set] = defaultdict(set)
    for r in records:
        if not r.get("error") and r.get("judge"):
            outcomes[r["id"]].add(answered(r["judge"]))
    multi = [s for qid, s in outcomes.items()
             if sum(1 for r in records if r["id"] == qid and r.get("judge")) > 1]
    if not multi:
        return None
    return round(mean(1.0 if len(s) > 1 else 0.0 for s in multi), 4)


def summarize(records: list[dict]) -> dict:
    runs = sorted({r["run"] for r in records})
    per_run = {run: summarize_run([r for r in records if r["run"] == run]) for run in runs}
    keys = [k for k in next(iter(per_run.values())).keys() if not k.startswith("n_")] if per_run else []
    pooled = {}
    for k in keys:
        vals = [m[k] for m in per_run.values() if m[k] is not None]
        if vals:
            pooled[k] = {"mean": round(mean(vals), 4), "min": min(vals), "max": max(vals)}
    judged = [r for r in records if not r.get("error") and r.get("judge")]
    ans = [r for r in judged if r["type"] != "unanswerable"]
    by_type = {}
    for t in sorted({r["type"] for r in ans}):
        rows = [r for r in ans if r["type"] == t]
        by_type[t] = {
            "n": len(rows),
            "refused": round(mean(1.0 if r["judge"].get("stance") == "refusal" else 0.0 for r in rows), 4),
            "correct": round(mean(1.0 if answered(r["judge"]) and r["judge"].get("verdict") == "correct" else 0.0
                                  for r in rows), 4),
        }
    evidence = Counter(r["evidence"]["evidence"] for r in ans
                       if r.get("evidence") and r["judge"].get("stance") == "refusal")
    routes = Counter(r.get("route") for r in judged if r.get("route"))
    return {
        "runs": len(runs),
        "per_run": per_run,
        "pooled": pooled,
        "flip_rate": flip_rate(records),
        "answerable_by_type": by_type,
        "refusal_evidence": dict(evidence),
        "routes": dict(routes),
        "errors": sum(1 for r in records if r.get("error")),
        "rerank_fallback": rerank_fallback(records),
    }


def rerank_fallback(records: list[dict]) -> dict | None:
    """Failed rerank calls and the answers built from unranked passages because of them.

    None when no record carries the count (reports saved before it existed). Records without it
    (from such a report merged with a newer one) are counted as not_recorded, never as clean.
    """
    counted = [r for r in records if "rerank_failures" in r]
    if not counted:
        return None
    return {
        "failed_calls": sum(r["rerank_failures"] for r in counted),
        "answers": sum(1 for r in counted if r["rerank_failures"] > 0),
        "of": len(counted),
        "not_recorded": len(records) - len(counted),
    }


def describe_error(exc: BaseException) -> str:
    """Status code and body first: the Cohere SDK's own str() starts with the response headers."""
    status, body = getattr(exc, "status_code", None), getattr(exc, "body", None)
    text = f"status_code: {status}, body: {body}" if status is not None else str(exc)
    return f"{type(exc).__name__}: {text}"[:500]


# ── Cost meter ─────────────────────────────────────────

_meter: contextvars.ContextVar[dict | None] = contextvars.ContextVar("contract_meter", default=None)
_fallback_meter: dict | None = None  # used when a call runs outside the question's context
# Set by the first failed rerank call; run_one then skips questions it has not started yet.
_rerank_failed = threading.Event()
_unattributed_rerank_failures = 0  # failed calls made outside any question's meter
_first_rerank_error: str | None = None


def _current_meter() -> dict | None:
    return _meter.get() or _fallback_meter


def new_meter() -> dict:
    return {"llm_calls": [], "rerank_searches": 0, "rerank_failures": 0, "embedded_tokens": 0}


def install_meter() -> None:
    """Wrap every LLM call site and the reranker so each question's full cost is recorded."""
    import app.agent.nodes as nodes
    import app.core.condense as condense
    import app.core.factories as factories
    import app.core.pipeline as pipeline
    import app.reranker.reranker as reranker
    from app.config import get_settings
    from app.observability.cost import count_tokens, usage_for

    original = factories.complete_with_model

    def metered(prompt: str, *, fast: bool = False):
        text, model = original(prompt, fast=fast)
        m = _current_meter()
        if m is not None:
            m["llm_calls"].append({"fast": fast, **usage_for(prompt, text, model)})
        return text, model

    for module in (factories, pipeline, condense, nodes):
        module.complete_with_model = metered

    original_rerank = reranker.RerankerService.rerank

    def metered_rerank(self, query, documents, top_k=3):
        global _unattributed_rerank_failures, _first_rerank_error
        m = _current_meter()
        cohere = self.reranker is not None and get_settings().RERANKER_PROVIDER == "cohere"
        if m is not None and cohere:
            m["embedded_tokens"] += count_tokens(query)  # one query embedding per retrieval
        try:
            result = original_rerank(self, query, documents, top_k)
        except Exception as exc:
            # The pipeline catches this and answers from unranked passages with only a log
            # warning, so record it here and re-raise to keep that fallback path unchanged.
            if m is not None:
                m["rerank_failures"] += 1
            else:
                _unattributed_rerank_failures += 1
            if _first_rerank_error is None:
                _first_rerank_error = describe_error(exc)
            _rerank_failed.set()
            raise
        if m is not None and cohere:
            m["rerank_searches"] += 1  # a failed call is not billed
        return result

    reranker.RerankerService.rerank = metered_rerank


def cost_of(meter: dict) -> dict:
    llm = round(sum(c["cost_usd"] for c in meter["llm_calls"]), 6)
    rerank = round(meter["rerank_searches"] * RERANK_USD_PER_SEARCH, 6)
    embed = round(meter["embedded_tokens"] * EMBED_USD_PER_TOKEN, 8)
    return {
        "llm_usd": llm,
        "rerank_usd": rerank,
        "embedding_usd": embed,
        "total_usd": round(llm + rerank + embed, 6),
        "llm_calls": len(meter["llm_calls"]),
        "fast_calls": sum(1 for c in meter["llm_calls"] if c["fast"]),
        "rerank_searches": meter["rerank_searches"],
    }


# ── Running the system ─────────────────────────────────

def load_items(dataset: str | None, extra: str | None, subset: int | None) -> list[dict]:
    items = json.loads(Path(dataset or HERE / "eval_dataset.json").read_text(encoding="utf-8"))
    if extra:
        items += json.loads(Path(extra).read_text(encoding="utf-8"))
    if subset:
        items = items[:subset]
    return items


def run_one(item: dict, run: int, mode: str, top_k: int | None, stop_on_rerank_failure: bool = False) -> dict:
    global _fallback_meter
    meter = new_meter()
    if stop_on_rerank_failure and _rerank_failed.is_set():
        return {"id": item["id"], "type": item["type"], "run": run, "question": item["question"],
                "answer": "", "exact_refusal": False, "cited": [], "citations_in_range": True,
                "sources": [], "latency_ms": 0.0, "error": "skipped: a rerank call failed earlier in this run",
                "rerank_failures": 0, "cost": cost_of(meter)}
    token = _meter.set(meter)
    if mode == "agent":
        _fallback_meter = meter  # agent mode runs one question at a time (see main)
    t0 = time.time()
    record = {"id": item["id"], "type": item["type"], "run": run, "question": item["question"]}
    try:
        if mode == "agent":
            from app.agent.graph import run_agent
            result = run_agent(item["question"], top_k)
        else:
            from app.core.pipeline import query_pipeline
            result = query_pipeline(item["question"], top_k)
        latency = (time.time() - t0) * 1000
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        cited = cited_numbers(answer)
        record.update({
            "answer": answer,
            "exact_refusal": is_exact_refusal(answer),
            "cited": cited,
            "citations_in_range": citations_in_range(cited, len(sources)),
            "sources": [{
                "citation": s.get("metadata", {}).get("citation"),
                "source": s.get("metadata", {}).get("source"),
                "sha1": hashlib.sha1((s.get("content") or "").encode("utf-8")).hexdigest()[:12],
                "content": s.get("content", ""),
            } for s in sources],
            "latency_ms": round(latency, 1),
            "error": None,
        })
        if mode == "agent":
            record.update({"route": result.get("route"), "attempts": result.get("attempts"),
                           "final_query": result.get("query")})
    except Exception as exc:  # noqa: BLE001 - one failed question must not stop the run
        record.update({"answer": "", "exact_refusal": False, "cited": [], "citations_in_range": True,
                       "sources": [], "latency_ms": round((time.time() - t0) * 1000, 1), "error": str(exc)})
    finally:
        _meter.reset(token)
        if mode == "agent":
            _fallback_meter = None
    record["cost"] = cost_of(meter)
    record["rerank_failures"] = meter["rerank_failures"]
    return record


# ── Judging ────────────────────────────────────────────

JUDGE_PROMPT = """You are grading one response from a document question-answering system.

Question: {question}
Reference: {reference}
This question is {kind}.

Response:
<<<
{response}
>>>

Label the response:
- "stance": "refusal" if it declines to answer or only says the documents do not contain the answer; "partial" if it answers part of the question and explicitly says the rest is not covered; "answer" otherwise.
- "verdict" (answerable questions with stance "answer" or "partial"; otherwise "n/a"): "correct" if it states the reference's key facts without contradicting them; "partially_correct" if it gets some key facts right but misses or garbles others; "incorrect" if its key facts are wrong or unrelated.
- "fabricated" (unanswerable questions; otherwise null): true if it asserts a specific answer (a value, name or fact) to the question as if the documents supported it; false if it declines or only states what the documents do not contain.

Reply with JSON only: {{"stance": "...", "verdict": "...", "fabricated": true|false|null, "reason": "<one short sentence>"}}"""

EVIDENCE_PROMPT = """Do the retrieved passages below contain the information needed to give the reference answer?

Question: {question}
Reference answer: {reference}

Passages:
{passages}

Answer "sufficient" if all key facts of the reference are stated in the passages, "partial" if only some are, "absent" if none are.
Reply with JSON only: {{"evidence": "sufficient|partial|absent", "reason": "<one short sentence>"}}"""


def _ask_json(llm, prompt: str) -> dict | None:
    for _ in range(2):
        reply = llm.invoke(prompt)
        parsed = parse_judge_json(getattr(reply, "content", "") or "")
        if parsed is not None:
            return parsed
    return None


def judge_record(llm, record: dict, item: dict) -> dict:
    kind = ("NOT answerable from the documents (the reference explains what is missing)"
            if item["type"] == "unanswerable" else "answerable from the documents")
    verdict = _ask_json(llm, JUDGE_PROMPT.format(
        question=item["question"], reference=item["ground_truth"], kind=kind, response=record["answer"]))
    out = {"judge": verdict or {"stance": "error", "verdict": "n/a", "fabricated": None, "reason": "unparseable"}}
    if item["type"] != "unanswerable" and verdict and verdict.get("stance") in ("refusal", "partial"):
        passages = "\n\n".join(f"[{s['citation']}] {s['content']}" for s in record["sources"]) or "(none)"
        out["evidence"] = _ask_json(llm, EVIDENCE_PROMPT.format(
            question=item["question"], reference=item["ground_truth"], passages=passages)) or {
            "evidence": "error", "reason": "unparseable"}
    return out


def judge_all(records: list[dict], items: dict[str, dict], judge_model: str | None, workers: int) -> None:
    from app.config import get_settings
    from app.core.factories import get_llm

    llm = get_llm(judge_model or get_settings().LLM_MODEL)
    todo = [r for r in records if not r.get("error")]

    def work(r):
        r.update(judge_record(llm, r, items[r["id"]]))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, todo))


# ── Reporting ──────────────────────────────────────────

def config_snapshot(args: argparse.Namespace) -> dict:
    from app.config import get_settings
    from app.core.prompts import select_prompt
    s = get_settings()
    prompt_sha = hashlib.sha1(select_prompt(s.PROMPT_MODE).encode("utf-8")).hexdigest()[:10]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=HERE.parent).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "app", "evaluation"],
                                    capture_output=True, text=True, cwd=HERE.parent).stdout.strip())
    except OSError:
        commit, dirty = "unknown", None
    return {
        "mode": args.mode, "runs": args.runs, "top_k": args.top_k or s.TOP_K,
        "rerank_top_k": s.RERANK_TOP_K, "llm_model": s.LLM_MODEL, "llm_temperature": s.LLM_TEMPERATURE,
        "prompt_mode": s.PROMPT_MODE, "prompt_sha": prompt_sha, "retrieval_mode": s.RETRIEVAL_MODE,
        "reranker": s.RERANKER_PROVIDER,
        "graph_extractor": s.GRAPH_EXTRACTOR, "keyword_backend": s.KEYWORD_BACKEND,
        "collection": s.COLLECTION_NAME, "data_dir": s.DATA_DIR,
        "judge_model": args.judge_model or s.LLM_MODEL, "git_commit": commit, "git_dirty": dirty,
    }


def _fmt(cell: dict | None, pct: bool = True) -> str:
    if not cell:
        return "-"
    f = (lambda v: f"{v * 100:.1f}%") if pct else (lambda v: f"{v:g}")
    if cell["min"] == cell["max"]:
        return f(cell["mean"])
    return f"{f(cell['mean'])} ({f(cell['min'])}-{f(cell['max'])})"


def render_markdown(report: dict) -> str:
    s, c, p = report["summary"], report["config"], report["summary"]["pooled"]
    flips = "-" if s["flip_rate"] is None else f"{s['flip_rate'] * 100:.1f}%"
    fb = s.get("rerank_fallback")
    fallback = "-" if fb is None else f"{fb['answers']} of {fb['of']}" + (
        f" ({fb['not_recorded']} more not recorded)" if fb.get("not_recorded") else "")
    warning = ([f"**Rerank failed for {fb['answers']} of {fb['of']} answers ({fb['failed_calls']} calls). "
                "Those answers used unranked passages, so this run does not measure the configured "
                "pipeline.**", ""] if fb and fb["answers"] else [])
    lines = [
        f"# Answer contract: {report['label']}",
        "",
        f"{report['timestamp_utc']} | mode `{c['mode']}` | {s['runs']} run(s) | "
        f"{report['n_items']} questions ({s['per_run'][next(iter(s['per_run']))]['n_unanswerable']} unanswerable)"
        if s["per_run"] else "",
        "",
        *warning,
        "Config: " + ", ".join(f"{k}={v}" for k, v in c.items()),
        "",
        "Rates are mean over runs (min-max in parentheses).",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| unanswerable: declined (no fabricated answer) | {_fmt(p.get('unanswerable_declined'))} |",
        f"| unanswerable: exact refusal sentence | {_fmt(p.get('unanswerable_exact_refusal'))} |",
        f"| answerable: refused | {_fmt(p.get('answerable_refused'))} |",
        f"| answerable: partial answer | {_fmt(p.get('answerable_partial'))} |",
        f"| answerable: correct | {_fmt(p.get('answerable_correct'))} |",
        f"| answerable: partially correct | {_fmt(p.get('answerable_partially_correct'))} |",
        f"| answerable: incorrect | {_fmt(p.get('answerable_incorrect'))} |",
        f"| answers with a citation | {_fmt(p.get('answers_with_citation'))} |",
        f"| citations in range | {_fmt(p.get('citations_in_range'))} |",
        f"| answered/refused flips across runs | {flips} |",
        f"| latency p50 / p95 (ms) | {_fmt(p.get('latency_p50_ms'), False)} / {_fmt(p.get('latency_p95_ms'), False)} |",
        f"| cost per 1,000 questions (USD) | {_fmt(p.get('cost_per_1k_questions_usd'), False)} |",
        f"| answers from unranked passages (rerank failed) | {fallback} |",
        "",
        "Answerable questions by type (pooled over runs):",
        "",
        "| type | n | refused | correct |",
        "| --- | --- | --- | --- |",
    ]
    for t, v in s["answerable_by_type"].items():
        lines.append(f"| {t} | {v['n']} | {v['refused'] * 100:.1f}% | {v['correct'] * 100:.1f}% |")
    if s["refusal_evidence"]:
        lines += ["", "Refused answerable questions, by whether the retrieved passages held the evidence: "
                  + ", ".join(f"{k} {v}" for k, v in sorted(s["refusal_evidence"].items()))]
    if s["routes"]:
        lines += ["", "Agent routes: " + ", ".join(f"{k} {v}" for k, v in sorted(s["routes"].items()))]
    if s["errors"]:
        lines += ["", f"Errors: {s['errors']}"]
    return "\n".join(lines) + "\n"


def save_aborted(args: argparse.Namespace, now: datetime, config: dict, item_list: list[dict],
                 records: list[dict], failed_calls: int, stopped_in_run: int) -> int:
    """Keep the raw records of a run stopped by a failed rerank call (nothing is judged); exit 2."""
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    skipped = sum(1 for r in records if (r.get("error") or "").startswith("skipped:"))
    report = {"label": args.label, "timestamp_utc": now.isoformat(timespec="seconds"),
              "timestamp_compact": stamp, "config": config, "n_items": len(item_list), "items": item_list,
              "summary": None,
              "aborted": {"reason": "rerank failed", "failed_calls": failed_calls,
                          "first_error": _first_rerank_error, "stopped_in_run": stopped_in_run,
                          "runs_not_started": args.runs - stopped_in_run,
                          "skipped_in_stopped_run": skipped},
              "records": records}
    out = Path(args.output_dir) / f"{stamp}_contract_{args.label}_aborted.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"ABORTED in run {stopped_in_run}/{args.runs}: {failed_calls} rerank call(s) failed, so answers "
          "fell back to unranked passages.\n"
          f"  first error: {_first_rerank_error}\n"
          f"  raw records (not judged): {out}\n"
          "  Fix the reranker and rerun, or pass --allow-rerank-fallback to measure the fallback on purpose.",
          file=sys.stderr, flush=True)
    return 2


def save(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report['timestamp_compact']}_contract_{report['label']}"
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(render_markdown(report), encoding="utf-8")
    return out_dir / f"{stem}.md"


# ── CLI ────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Answer-contract evaluation (answer vs refuse)")
    p.add_argument("--mode", choices=["standard", "agent"], default="standard")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--label", default="contract")
    p.add_argument("--dataset", default=None, help="Base question set (default: eval_dataset.json)")
    p.add_argument("--extra", default=str(HERE / "unanswerable_extra.json"),
                   help="Extra items appended to the dataset (default: unanswerable_extra.json)")
    p.add_argument("--no-extra", action="store_true", help="Use only the base question set")
    p.add_argument("--subset", type=int, default=None, help="First N items only (smoke test)")
    p.add_argument("--top-k", type=int, default=None, help="Retrieval candidates per question (default: TOP_K)")
    p.add_argument("--workers", type=int, default=4, help="Questions in flight (agent mode always uses 1)")
    p.add_argument("--judge-model", default=None,
                   help="Judge model (default: LLM_MODEL; the fast model over-credited thin evidence in a smoke test)")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--rejudge", default=None, help="Re-judge the records of a saved report instead of running")
    p.add_argument("--merge", nargs="+", default=None,
                   help="Combine judged reports of the same config (e.g. base set + extra items) into one")
    p.add_argument("--output-dir", default=str(HERE / "results"))
    p.add_argument("--allow-rerank-fallback", action="store_true",
                   help="Keep running when a rerank call fails (default: stop, save raw records, exit 2)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global _unattributed_rerank_failures, _first_rerank_error
    args = parse_args(argv)
    now = datetime.now(timezone.utc)
    if args.merge:
        sources = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.merge]
        records = [r for s in sources for r in s["records"]]
        item_list = [it for s in sources for it in s["items"]]
        config = {**sources[0]["config"], "merged_from": [Path(p).name for p in args.merge]}
        report = {"label": args.label, "timestamp_utc": now.isoformat(timespec="seconds"),
                  "timestamp_compact": now.strftime("%Y%m%dT%H%M%SZ"), "config": config,
                  "n_items": len(item_list), "items": item_list, "summary": summarize(records),
                  "records": records}
        path = save(report, Path(args.output_dir))
        print(render_markdown(report))
        print(f"saved {path}")
        return 0
    if args.rejudge:
        source = json.loads(Path(args.rejudge).read_text(encoding="utf-8"))
        records = source["records"]
        items = {r["id"]: r for r in source["items"]}
        for r in records:
            r.pop("judge", None)
            r.pop("evidence", None)
        config = {**source["config"], "judge_model": args.judge_model or source["config"]["judge_model"],
                  "rejudged_from": Path(args.rejudge).name}
        item_list = source["items"]
    else:
        item_list = load_items(args.dataset, None if args.no_extra else args.extra, args.subset)
        items = {it["id"]: it for it in item_list}
        install_meter()
        _rerank_failed.clear()
        _unattributed_rerank_failures, _first_rerank_error = 0, None
        stop = not args.allow_rerank_fallback
        workers = 1 if args.mode == "agent" else max(1, args.workers)
        records = []
        last_run = 0
        for run in range(1, args.runs + 1):
            last_run = run
            print(f"run {run}/{args.runs}: {len(item_list)} questions ({args.mode})", flush=True)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                records += list(pool.map(
                    lambda it, run=run: run_one(it, run, args.mode, args.top_k, stop), item_list))
            if stop and _rerank_failed.is_set():
                break
        config = config_snapshot(args)
        failed = (rerank_fallback(records) or {}).get("failed_calls", 0) + _unattributed_rerank_failures
        if stop and failed:
            return save_aborted(args, now, config, item_list, records, failed, last_run)
        if _unattributed_rerank_failures:
            print(f"warning: {_unattributed_rerank_failures} failed rerank call(s) ran outside any question "
                  "and are not in the per-answer counts", file=sys.stderr, flush=True)
    if not args.no_judge:
        print(f"judging {len(records)} responses", flush=True)
        judge_all(records, items, args.judge_model, workers=8)
    report = {
        "label": args.label,
        "timestamp_utc": now.isoformat(timespec="seconds"),
        "timestamp_compact": now.strftime("%Y%m%dT%H%M%SZ"),
        "config": config,
        "n_items": len(item_list),
        "items": item_list,
        "summary": summarize(records) if not args.no_judge else None,
        "records": records,
    }
    if args.no_judge:
        out = Path(args.output_dir) / f"{report['timestamp_compact']}_contract_{args.label}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"saved {out}")
        return 0
    path = save(report, Path(args.output_dir))
    print(render_markdown(report))
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
