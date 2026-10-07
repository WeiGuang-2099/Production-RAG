import pytest

from evaluation.run_contract import (
    REFUSAL,
    citations_in_range,
    cited_numbers,
    flip_rate,
    is_exact_refusal,
    parse_judge_json,
    percentile,
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
