# Evaluation

End-to-end quality evaluation of the RAG pipeline using a fixed corpus of
classic ML/AI papers and 48 hand-crafted questions across 6 question types.

## Corpus

The evaluation corpus is 6 classic AI papers downloaded from arXiv:

| slug | Paper | arXiv |
|---|---|---|
| `attention` | Attention Is All You Need (Vaswani et al., 2017) | 1706.03762 |
| `bert` | BERT (Devlin et al., 2018) | 1810.04805 |
| `gpt3` | Language Models are Few-Shot Learners (Brown et al., 2020) | 2005.14165 |
| `rag` | Retrieval-Augmented Generation (Lewis et al., 2020) | 2005.11401 |
| `lora` | LoRA (Hu et al., 2021) | 2106.09685 |
| `cot` | Chain-of-Thought Prompting (Wei et al., 2022) | 2201.11903 |

These were chosen because (a) they are well-known reference points, (b) they
naturally create cross-paper relationships ideal for multi-hop questions, and
(c) PDFs are stable and downloadable.

## Question set

`eval_dataset.json` contains 48 questions. Each item has:

| field | meaning |
|---|---|
| `id` | stable identifier (`q001` ... `q048`) |
| `question` | natural-language question |
| `ground_truth` | canonical reference answer |
| `source_papers` | list of paper slugs the answer is grounded in |
| `type` | `factual` / `multi_hop` / `comparative` / `numerical` / `unanswerable` / `long_tail` |
| `difficulty` | `easy` / `medium` / `hard` |

Breakdown:

| type | count | purpose |
|---|---|---|
| factual | 15 | single-paper lookup; baseline retrieval quality |
| multi_hop | 10 | requires linking 2+ papers; tests graph / cross-doc reasoning |
| comparative | 8 | side-by-side analysis; tests synthesis |
| numerical | 5 | specific numbers; tests precise retrieval |
| unanswerable | 5 | answer NOT in corpus; tests refusal vs. hallucination |
| long_tail | 5 | obscure details; tests retrieval recall on rare facts |

The `unanswerable` bucket is important: a good RAG system should say "the
paper does not state X" rather than hallucinate. Most public RAG benchmarks
under-test this, and it is what differentiates a production-ready system.

## Running the evaluation

```bash
# 1. Download the corpus (writes PDFs to <DATA_DIR>/papers/)
python evaluation/corpus/download_papers.py

# 2. Ingest the papers through the running API
for slug in attention bert gpt3 rag lora cot; do
  curl -X POST http://localhost:8000/ingest \
    -H "Content-Type: application/json" \
    -d "{\"source\": \"./data/papers/$slug.pdf\"}"
done

# 3. Smoke test on 5 questions first (cheaper)
python evaluation/run_eval.py --subset 5 --label smoke

# 4. Full 48-question baseline
python evaluation/run_eval.py --label baseline
```

Note: ingest may take several minutes depending on `GRAPH_EXTRACTOR` setting
(LLM extraction is the dominant cost). Use `GRAPH_EXTRACTOR=none` for a
faster first pass.

### `run_eval.py` flags

| flag | purpose |
|---|---|
| `--subset N` | run only the first N questions (quick smoke test) |
| `--types factual,multi_hop` | filter by question type |
| `--label baseline` | tag this run; used in the saved report filename |
| `--output path.json` | explicit report path (default: `results/<UTC>_<label>.json`) |
| `--no-save` | skip writing the report file |

### Output

Each run prints:

- overall scores for the 4 RAGAS metrics
- a breakdown by question **type** (factual / multi_hop / ...)
- a breakdown by **difficulty** (easy / medium / hard)
- latency stats (mean / p50 / p95 / max)
- per-bucket question count `n`

And saves a structured JSON report under `evaluation/results/` containing
the per-item scores, latency, and any failures. This is what you commit /
share to demonstrate measurable RAG quality.

## Metrics

`run_eval.py` reports four RAGAS metrics:

- **faithfulness** — does the answer make claims supported by retrieved context?
- **answer_relevancy** — does the answer address the question?
- **context_recall** — does the retrieved context contain the ground-truth answer?
- **context_precision** — are the top-ranked contexts the relevant ones?

Faithfulness + context_recall together catch most "looks-correct-but-isn't"
failure modes; context_precision catches "we retrieved noise that worked
out anyway".
