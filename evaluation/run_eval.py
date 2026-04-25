"""Run RAGAS evaluation metrics against the RAG system."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

from app.core.pipeline import query_pipeline


def load_dataset(path: str = None) -> list[dict]:
    if path is None:
        path = str(Path(__file__).resolve().parent / "eval_dataset.json")
    with open(path) as f:
        return json.load(f)


def run_evaluation():
    dataset = load_dataset()
    samples = []

    for item in dataset:
        result = query_pipeline(item["question"])
        sample = SingleTurnSample(
            user_input=item["question"],
            response=result["answer"],
            reference=item["ground_truth"],
            retrieved_contexts=[s["content"] for s in result["sources"]],
        )
        samples.append(sample)

    eval_dataset = EvaluationDataset(samples=samples)
    results = evaluate(eval_dataset, metrics=[faithfulness, answer_relevancy, context_recall])
    print(results)
    return results


if __name__ == "__main__":
    run_evaluation()
