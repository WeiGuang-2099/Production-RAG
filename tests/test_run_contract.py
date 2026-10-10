import json

import pytest
from langchain_core.documents import Document

import app.reranker.reranker as reranker_mod
import evaluation.run_contract as rc
from app.config import get_settings
from app.reranker.reranker import RerankerService
from evaluation.run_contract import (
    REFUSAL,
    citations_in_range,
    cited_numbers,
    flip_rate,
    is_exact_refusal,
    parse_judge_json,
    percentile,
    portable_path,
    render_markdown,
    rerank_fallback,
    summarize,
    summarize_run,
)


@pytest.mark.parametrize("text", [
    REFUSAL,
    f'"{REFUSAL}"',
    "  i cannot answer this from the provided documents  ",
    "I cannot answer this from the provided documents",
])
def test_exact_refusal_ignores_quotes_case_and_period(text):
    assert is_exact_refusal(text)


def test_partial_answer_is_not_an_exact_refusal():
    assert not is_exact_refusal("The paper uses 8 heads [1]. " + REFUSAL)


def test_cited_numbers_in_order():
    assert cited_numbers("A [1] and B [2][3], again [1].") == [1, 2, 3, 1]
    assert cited_numbers("no citations") == []


def test_citations_in_range():
    assert citations_in_range([1, 5], 5)
    assert not citations_in_range([6], 5)
    assert citations_in_range([], 0)


def test_parse_judge_json_tolerates_fences_and_prose():
    assert parse_judge_json('```json\n{"stance": "answer"}\n```') == {"stance": "answer"}
    assert parse_judge_json("Here: {\"stance\": \"refusal\"} done") == {"stance": "refusal"}
    assert parse_judge_json("no json") is None
    assert parse_judge_json("{broken") is None


def test_percentile_nearest_rank():
    assert percentile([], 50) == 0.0
    assert percentile([10, 20, 30, 40], 50) == 20
    assert percentile(list(range(1, 21)), 95) == 19


def _rec(qid, qtype, run, stance, verdict="n/a", fabricated=None, cited=(1,), cost=0.01, latency=100.0):
    return {
        "id": qid, "type": qtype, "run": run, "error": None,
        "exact_refusal": stance == "refusal", "cited": list(cited), "citations_in_range": True,
        "latency_ms": latency, "cost": {"total_usd": cost},
        "judge": {"stance": stance, "verdict": verdict, "fabricated": fabricated},
    }


def test_summarize_run_contract_rates():
    records = [
        _rec("u1", "unanswerable", 1, "refusal", fabricated=False, cited=()),
        _rec("u2", "unanswerable", 1, "answer", fabricated=True),
        _rec("a1", "factual", 1, "answer", verdict="correct"),
        _rec("a2", "factual", 1, "refusal", cited=()),
        _rec("a3", "multi_hop", 1, "partial", verdict="partially_correct"),
        _rec("a4", "factual", 1, "answer", verdict="incorrect", cited=()),
    ]
    m = summarize_run(records)
    assert m["unanswerable_declined"] == 0.5
    assert m["answerable_refused"] == 0.25
    assert m["answerable_partial"] == 0.25
    assert m["answerable_correct"] == 0.25
    assert m["answerable_incorrect"] == 0.25
    # 3 answered answerable rows, 2 of them cite something
    assert m["answers_with_citation"] == round(2 / 3, 4)
    assert m["cost_per_1k_questions_usd"] == 10.0


def test_flip_rate_counts_questions_whose_outcome_changes():
    records = [
        _rec("a1", "factual", 1, "answer", verdict="correct"),
        _rec("a1", "factual", 2, "refusal"),
        _rec("a2", "factual", 1, "answer", verdict="correct"),
        _rec("a2", "factual", 2, "partial", verdict="correct"),  # partial counts as answered
    ]
    assert flip_rate(records) == 0.5


def test_summarize_pools_runs_and_breaks_down_by_type():
    records = [
        _rec("a1", "factual", 1, "answer", verdict="correct"),
        _rec("a1", "factual", 2, "refusal"),
        _rec("u1", "unanswerable", 1, "refusal", fabricated=False, cited=()),
        _rec("u1", "unanswerable", 2, "refusal", fabricated=False, cited=()),
    ]
    records[1]["evidence"] = {"evidence": "sufficient"}
    s = summarize(records)
    assert s["runs"] == 2
    assert s["pooled"]["answerable_refused"] == {"mean": 0.5, "min": 0.0, "max": 1.0}
    assert s["answerable_by_type"]["factual"] == {"n": 2, "refused": 0.5, "correct": 0.5}
    assert s["refusal_evidence"] == {"sufficient": 1}
    assert s["errors"] == 0


# ── Rerank-failure guard ───────────────────────────────

class _BillingCapCohere:
    """Stands in for the Cohere client after its billing cap: every call is a 402."""

    def rerank(self, documents, query, top_n):
        raise RuntimeError("status_code: 402, Maximum billing reached for this API key")


class _WorkingCohere:
    def rerank(self, documents, query, top_n):
        return [{"index": len(documents) - 1, "relevance_score": 0.9}][:top_n]


@pytest.fixture
def restore_patches(monkeypatch):
    """install_meter() patches module attributes in place; register them so they are restored."""
    import app.agent.nodes as nodes
    import app.core.condense as condense
    import app.core.factories as factories
    import app.core.pipeline as pipeline

    for module in (factories, pipeline, condense, nodes):
        monkeypatch.setattr(module, "complete_with_model", module.complete_with_model)
    monkeypatch.setattr(reranker_mod.RerankerService, "rerank", reranker_mod.RerankerService.rerank)
    monkeypatch.setattr(rc, "_unattributed_rerank_failures", 0)
    monkeypatch.setattr(rc, "_first_rerank_error", None)
    monkeypatch.setenv("RERANKER_PROVIDER", "cohere")
    get_settings.cache_clear()
    rc._rerank_failed.clear()
    yield
    rc._rerank_failed.clear()
    get_settings.cache_clear()


@pytest.fixture
def metered(restore_patches):
    rc.install_meter()
    meter = rc.new_meter()
    token = rc._meter.set(meter)
    yield meter
    rc._meter.reset(token)


def test_failed_rerank_is_counted_reraised_and_not_billed(metered):
    with pytest.raises(RuntimeError, match="402"):
        RerankerService(reranker=_BillingCapCohere()).rerank("q", [Document(page_content="a")], top_k=1)
    assert metered["rerank_failures"] == 1
    assert metered["rerank_searches"] == 0
    assert rc._rerank_failed.is_set()
    assert "402" in rc._first_rerank_error


def test_successful_rerank_is_billed_and_not_a_failure(metered):
    docs = [Document(page_content="a"), Document(page_content="b")]
    out = RerankerService(reranker=_WorkingCohere()).rerank("q", docs, top_k=1)
    assert [d.page_content for d in out] == ["b"]
    assert metered["rerank_searches"] == 1
    assert metered["rerank_failures"] == 0
    assert not rc._rerank_failed.is_set()


def test_pipeline_still_falls_back_to_unranked_passages(metered, monkeypatch):
    """The wrapper re-raises, so the pipeline's own fallback runs exactly as before."""
    import app.core.pipeline as pipeline

    docs = [(Document(page_content=f"p{i}"), 1.0 - i / 10) for i in range(4)]
    vs = type("VS", (), {"search": lambda self, q, top_k, sources=None: docs})()
    monkeypatch.setattr(pipeline, "get_vector_store", lambda: vs)
    monkeypatch.setattr(pipeline, "get_reranker", lambda: _BillingCapCohere())
    for key, val in {"RETRIEVAL_MODE": "dense", "GRAPH_EXTRACTOR": "none", "QUERY_TRANSFORM": "none",
                     "RERANK_TOP_K": "2"}.items():
        monkeypatch.setenv(key, val)
    get_settings.cache_clear()

    out = pipeline._retrieve_and_rerank("q", 4, get_settings())
    assert [d.page_content for d in out] == ["p0", "p1"]
    assert metered["rerank_failures"] == 1


def _rec_with_fallback(qid, run, failures):
    r = _rec(qid, "factual", run, "answer", verdict="correct")
    r["rerank_failures"] = failures
    return r


def test_rerank_fallback_counts_calls_and_answers():
    records = [_rec_with_fallback("a1", 1, 0), _rec_with_fallback("a2", 1, 2), _rec_with_fallback("a3", 1, 1)]
    assert rerank_fallback(records) == {"failed_calls": 3, "answers": 2, "of": 3, "not_recorded": 0}
    assert summarize(records)["rerank_fallback"]["answers"] == 2


def test_rerank_fallback_is_unknown_for_reports_saved_before_the_count():
    assert rerank_fallback([_rec("a1", "factual", 1, "answer", verdict="correct")]) is None
    assert rerank_fallback([]) is None


def test_merged_records_without_the_count_are_not_reported_as_clean():
    legacy = [_rec("a1", "factual", 1, "answer", verdict="correct") for _ in range(4)]
    records = legacy + [_rec_with_fallback("u1", 1, 0)]
    assert rerank_fallback(records) == {"failed_calls": 0, "answers": 0, "of": 1, "not_recorded": 4}
    md = render_markdown(_report(records))
    assert "| answers from unranked passages (rerank failed) | 0 of 1 (4 more not recorded) |" in md


def test_portable_path_keeps_local_paths_out_of_reports(tmp_path):
    assert portable_path(str(rc.HERE.parent / "data")) == "data"
    assert portable_path(str(rc.HERE.parent / "data_scale")) == "data_scale"
    assert portable_path(str(tmp_path / "scratch" / "data_v1")) == "<outside the repo>/data_v1"


def test_describe_error_puts_status_and_body_before_the_headers():
    class ApiError(Exception):
        status_code = 402
        body = {"message": "Maximum billing reached for this API key"}

        def __str__(self):
            return "headers: {" + "'x-long-header': 'v', " * 60 + "}, status_code: 402, body: ..."

    text = rc.describe_error(ApiError())
    assert text.startswith("ApiError: status_code: 402, body: {'message': 'Maximum billing reached")
    assert rc.describe_error(RuntimeError("timeout")) == "RuntimeError: timeout"


def _report(records):
    return {"label": "t", "timestamp_utc": "2026-10-08T00:00:00+00:00", "n_items": 1,
            "config": {"mode": "standard"}, "summary": summarize(records)}


def test_markdown_warns_when_answers_fell_back():
    md = render_markdown(_report([_rec_with_fallback("a1", 1, 1), _rec_with_fallback("a2", 1, 0)]))
    assert "**Rerank failed for 1 of 2 answers (1 calls)." in md
    assert "| answers from unranked passages (rerank failed) | 1 of 2 |" in md


def test_markdown_shows_zero_fallback_without_a_warning():
    md = render_markdown(_report([_rec_with_fallback("a1", 1, 0)]))
    assert "Rerank failed" not in md
    assert "| answers from unranked passages (rerank failed) | 0 of 1 |" in md


def test_markdown_marks_old_reports_as_not_recorded():
    md = render_markdown(_report([_rec("a1", "factual", 1, "answer", verdict="correct")]))
    assert "| answers from unranked passages (rerank failed) | - |" in md


def _fake_pipeline(question, top_k):
    """Mimics query_pipeline: a failed rerank is swallowed and the answer still comes back."""
    try:
        RerankerService(reranker=_BillingCapCohere()).rerank(question, [Document(page_content="p")], top_k=1)
    except Exception:  # noqa: BLE001 - the pipeline logs and uses the unranked passages
        pass
    return {"answer": "It is p [1].", "sources": [{"content": "p", "metadata": {"citation": 1}}]}


def _contract_args(tmp_path, *extra):
    items = [{"id": f"q{i}", "type": "factual", "question": f"question {i}", "ground_truth": "p"}
             for i in range(3)]
    dataset = tmp_path / "items.json"
    dataset.write_text(json.dumps(items), encoding="utf-8")
    return ["--dataset", str(dataset), "--no-extra", "--runs", "2", "--workers", "1", "--label", "t",
            "--output-dir", str(tmp_path / "out"), *extra]


def test_main_stops_at_the_first_rerank_failure_and_exits_2(restore_patches, monkeypatch, tmp_path):
    import app.core.pipeline as pipeline

    monkeypatch.setattr(pipeline, "query_pipeline", _fake_pipeline)
    monkeypatch.setattr(rc, "judge_all", lambda *a, **k: pytest.fail("an aborted run must not be judged"))

    assert rc.main(_contract_args(tmp_path)) == 2

    [saved] = (tmp_path / "out").glob("*_contract_t_aborted.json")
    report = json.loads(saved.read_text(encoding="utf-8"))
    records = report["records"]
    assert len(records) == 3  # run 2 never started
    assert records[0]["rerank_failures"] == 1 and records[0]["error"] is None
    assert all(r["error"].startswith("skipped:") for r in records[1:])
    assert report["aborted"]["failed_calls"] == 1
    assert report["aborted"]["stopped_in_run"] == 1
    assert report["aborted"]["runs_not_started"] == 1
    assert report["aborted"]["skipped_in_stopped_run"] == 2
    assert "402" in report["aborted"]["first_error"]


def test_main_attributes_every_failure_with_parallel_workers(restore_patches, monkeypatch, tmp_path):
    """The paid runs use 4 threads: a cap reached mid-run must still be counted per answer."""
    import threading

    import app.core.pipeline as pipeline

    calls, lock = {"n": 0}, threading.Lock()

    def capped_after_two(question, top_k):
        with lock:
            calls["n"] += 1
            capped = calls["n"] > 2
        reranker = _BillingCapCohere() if capped else _WorkingCohere()
        try:
            RerankerService(reranker=reranker).rerank(question, [Document(page_content="p")], top_k=1)
        except Exception:  # noqa: BLE001 - the pipeline's fallback
            pass
        return {"answer": "It is p [1].", "sources": [{"content": "p", "metadata": {"citation": 1}}]}

    monkeypatch.setattr(pipeline, "query_pipeline", capped_after_two)
    args = _contract_args(tmp_path)
    args[args.index("--workers") + 1] = "3"
    items = [{"id": f"q{i}", "type": "factual", "question": f"question {i}", "ground_truth": "p"}
             for i in range(12)]
    (tmp_path / "items.json").write_text(json.dumps(items), encoding="utf-8")

    assert rc.main(args) == 2

    [saved] = (tmp_path / "out").glob("*_contract_t_aborted.json")
    report = json.loads(saved.read_text(encoding="utf-8"))
    records = report["records"]
    assert {r["run"] for r in records} == {1}
    assert report["aborted"]["failed_calls"] == calls["n"] - 2 >= 1
    assert sum(r["rerank_failures"] for r in records) == report["aborted"]["failed_calls"]
    assert rc._unattributed_rerank_failures == 0


def test_main_with_no_records_does_not_crash(restore_patches, tmp_path):
    assert rc.main(_contract_args(tmp_path, "--no-judge") + ["--runs", "0"]) == 0


def test_main_keeps_going_when_fallback_is_allowed(restore_patches, monkeypatch, tmp_path):
    import app.core.pipeline as pipeline

    monkeypatch.setattr(pipeline, "query_pipeline", _fake_pipeline)

    assert rc.main(_contract_args(tmp_path, "--allow-rerank-fallback", "--no-judge")) == 0

    [saved] = (tmp_path / "out").glob("*_contract_t.json")
    records = json.loads(saved.read_text(encoding="utf-8"))["records"]
    assert len(records) == 6
    assert all(r["rerank_failures"] == 1 and r["error"] is None for r in records)
