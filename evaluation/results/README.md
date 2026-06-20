# Evaluation results

This directory holds committed evaluation reports — the evidence that the
system works and that each component earns its place. Two kinds of report
land here:

- `*_ablation.{md,json}` — retrieval ablation from `run_ablation.py`
  (deterministic recall@k / MRR / hit@k; no LLM judge, cheap to run).
- `*_<label>.json` — full RAGAS reports from `run_eval.py`
  (faithfulness / answer_relevancy / context_recall / context_precision).

## Retrieval ablation (fill in after running)

Regenerate:

```bash
python evaluation/corpus/download_papers.py        # once
# ingest the 6 papers (see ../README.md), then:
python evaluation/run_ablation.py --k 5
```

Paste the printed table here. It should look like:

| stage | recall@5 | mrr | hit@5 | p50_ms | p95_ms | count |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | 43 |
| +bm25 | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | 43 |
| +rerank | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | 43 |
| +graph | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | 43 |

(`count` is 43 = 48 questions minus the 5 `unanswerable` ones, which have no
ground-truth source papers and so are excluded from retrieval recall.)

## End-to-end RAGAS (fill in after running)

Regenerate the generation-quality numbers, including the grounded-vs-basic
prompt comparison that the `unanswerable` bucket is designed to expose:

```bash
PROMPT_MODE=basic    python evaluation/run_eval.py --label basic
PROMPT_MODE=grounded python evaluation/run_eval.py --label grounded
```

| prompt | faithfulness | answer_relevancy | context_recall | context_precision |
| --- | --- | --- | --- | --- |
| basic | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| grounded | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

The expected story: `grounded` improves faithfulness (and refuses on the
`unanswerable` questions instead of hallucinating) at little or no cost to
relevancy on answerable questions.
