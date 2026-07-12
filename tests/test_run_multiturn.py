from unittest.mock import patch

from app.core.condense import CondenseResult
from evaluation.run_multiturn import CONDITIONS, evaluate

_DATASET = [
    {
        "id": "mt001",
        "history": [{"role": "user", "content": "What is LoRA?"}],
        "follow_up": "its cost?",
        "standalone_reference": "What is LoRA's cost?",
        "source_papers": ["lora"],
    },
    {
        "id": "mt002",
        "history": [{"role": "user", "content": "What is BERT?"}],
        "follow_up": "how many parameters?",
        "standalone_reference": "How many parameters does BERT-Base have?",
        "source_papers": ["bert"],
    },
]


def _sources(slug):
    return [{"content": "c", "metadata": {"source": f"./data/papers/{slug}.pdf"}}]


def test_evaluate_scores_three_conditions():
    def fake_retrieve(q, top_k=None):
        if q in ("its cost?", "how many parameters?"):
            return _sources("wrong")  # raw follow-ups miss
        if "LoRA" in q:
            return _sources("lora")
        return _sources("bert")

    def fake_condense(follow_up, history):
        standalone = (
            "What is LoRA's cost?" if "cost" in follow_up
            else "How many parameters does BERT-Base have?"
        )
        return CondenseResult(question=standalone, applied=True)

    with patch("evaluation.run_multiturn.retrieve_sources", side_effect=fake_retrieve), \
         patch("evaluation.run_multiturn.condense_question", side_effect=fake_condense):
        out = evaluate(_DATASET, k=5, top_k=10)

    rows = {r["condition"]: r for r in out["rows"]}
    assert list(rows) == list(CONDITIONS)
    assert rows["raw"]["recall@5"] == 0.0
    assert rows["condensed"]["recall@5"] == 1.0
    assert rows["oracle"]["recall@5"] == 1.0
    assert len(out["rewrites"]) == 2
    assert out["rewrites"][0]["condensed"] == "What is LoRA's cost?"
    assert out["condense_p50_ms"] is not None


def test_evaluate_survives_retrieval_failure():
    with patch("evaluation.run_multiturn.retrieve_sources", side_effect=RuntimeError("down")), \
         patch("evaluation.run_multiturn.condense_question",
               return_value=CondenseResult("s?", False)):
        out = evaluate(_DATASET[:1], k=5, top_k=10)
    rows = {r["condition"]: r for r in out["rows"]}
    assert rows["raw"]["recall@5"] == 0.0
    assert rows["oracle"]["recall@5"] == 0.0
