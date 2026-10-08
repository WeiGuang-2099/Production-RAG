# Evaluation

End-to-end quality evaluation of the RAG pipeline using a fixed corpus of
classic ML/AI papers, 48 hand-crafted questions across 6 question types, and
22 near-miss unanswerable questions used by the answer-contract eval.

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

### Scale-robustness corpus (30 papers)

The ablation can also run against a 4.6x corpus (2,194 chunks) that adds 24
distractor papers deliberately confusable with the 6 above — see `DISTRACTORS`
in [`corpus/download_papers.py`](corpus/download_papers.py). None of their
slugs appear in `source_papers`, so retrieving a distractor is always a scored
miss. The scale corpus lives in its own data dir and Qdrant collection so the
6-paper corpus stays intact:

```bash
DATA_DIR=./data_scale python evaluation/corpus/download_papers.py --with-distractors
DATA_DIR=./data_scale COLLECTION_NAME=rag_docs_scale30 GRAPH_EXTRACTOR=llm \
    python evaluation/ingest_corpus.py --with-distractors
DATA_DIR=./data_scale COLLECTION_NAME=rag_docs_scale30 \
    python evaluation/run_ablation.py --k 5 --label scale30
```

Note: LLM graph extraction is one call per chunk (~1,700 new chunks, roughly
$6-10 with gpt-4o, 1-2 hours; interruptible — per-paper tracking resumes). The
retrieval ablation itself never calls the generation model.

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
| multi_hop | 10 | multi-step questions, 6 of which span 2+ papers; tests cross-doc reasoning |
| comparative | 8 | side-by-side analysis; tests synthesis |
| numerical | 5 | specific numbers; tests precise retrieval |
| unanswerable | 5 | answer NOT in corpus; tests refusal vs. hallucination |
| long_tail | 5 | obscure details; tests retrieval recall on rare facts |

The `unanswerable` bucket is important: a good RAG system should say "the
paper does not state X" rather than hallucinate. Five items are too few to
measure that, so `unanswerable_extra.json` adds 22 near-miss questions (`q049`
... `q070`, same fields): topics the papers cover, facts they never state, such
as how many V100 GPUs trained GPT-3 or which Adam epsilon pre-trained BERT. Each
was drafted with model help and kept only after two independent checks of the
479 chunks found no answer (2 of 24 candidates were dropped as ambiguous). The
answer-contract eval below uses them; the retrieval ablation and RAGAS runs use
the original 48.

## Running the evaluation

Steps 3-4 below run the RAGAS eval, kept for history: it scores every refusal
as zero, so the answer contract (below) superseded it for current numbers. For
those, run `run_contract.py` and `run_ablation.py` after step 2.

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

Note: the ablation's `+graph` stage needs a graph built at ingest, and the
shipped default is `GRAPH_EXTRACTOR=none`. Set `GRAPH_EXTRACTOR=llm` in the
API's `.env` for this step to reproduce it (one LLM call per chunk, the
dominant ingest cost); leave it at `none` for a faster first pass.

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
the per-item scores, latency, and any failures.

## Metrics

`run_eval.py` reports four RAGAS metrics:

- **faithfulness** — does the answer make claims supported by retrieved context?
- **answer_relevancy** — does the answer address the question?
- **context_recall** — does the retrieved context contain the ground-truth answer?
- **context_precision** — are the top-ranked contexts the relevant ones?

Faithfulness + context_recall together catch most "looks-correct-but-isn't"
failure modes; context_precision catches "we retrieved noise that worked
out anyway".

## Retrieval ablation (cheap, deterministic)

`run_eval.py` uses RAGAS, which needs an LLM judge (slow + paid). To answer
the narrower question "is each retrieval component pulling its weight?" use
`run_ablation.py`, which reports deterministic retrieval metrics computed
directly from the `source_papers` ground truth — no generation calls:

```bash
python evaluation/run_ablation.py --k 5
```

It sweeps four cumulative configurations and prints a comparison table:

| stage | RETRIEVAL_MODE | RERANKER_PROVIDER | GRAPH_EXTRACTOR |
|---|---|---|---|
| baseline | dense | none | none |
| +bm25 | hybrid | none | none |
| +rerank | hybrid | cohere | none |
| +graph | hybrid | cohere | llm (needs the graph built at ingest) |

Metrics:

- **recall@k** — fraction of the question's ground-truth papers present in the top-k retrieved contexts.
- **MRR** — 1 / rank of the first relevant context (rewards ranking the right doc high).
- **hit@k** — did any relevant paper make the top-k at all.

The 5 `unanswerable` questions each list the paper they ask about in
`source_papers`, so retrieval is scored on all 48 (the reports show `count` 48).
Reports are written to `results/`; see [`results/README.md`](results/README.md)
for the published tables.

A failed rerank call does not fail the question: the pipeline logs a warning
and keeps the unranked hybrid order, which quietly turns `+rerank` into
`+bm25`. Each stage counts failed calls in a `rerank_failures` column, a
warning above the table names the stages that had any, and the script exits 2
after writing the report unless `--allow-rerank-fallback` is passed.

## Answer contract: answer what it can, refuse what it cannot

RAGAS scores any refusal as zero, right or wrong, so it cannot separate a
correct "not in the documents" from a missed answer. `run_contract.py` measures
that contract directly:

```bash
python evaluation/run_contract.py --label final                 # 70 questions x 3 passes, Standard mode
python evaluation/run_contract.py --mode agent --label agent    # the same questions through the agent
python evaluation/run_contract.py --subset 5 --runs 1 --label smoke
# the shipped setup before the fix (run A), except that BM25 now uses the fixed tokenizer
PROMPT_MODE=strict LLM_TEMPERATURE=default python evaluation/run_contract.py --top-k 5 --label before
```

It runs the 48 questions plus the 22 near-miss unanswerable items in several
passes (`--runs`, default 3; the reports call each pass a run), and saves
every answer with its `[n]` citations, the retrieved passages, latency and the
cost of every model and rerank call. A judge model (default `LLM_MODEL`)
labels each response:

- **stance**: refusal, partial (answers part and says what is missing) or answer;
- **verdict** on answerable items: correct, partially correct or incorrect against the reference;
- **fabricated** on unanswerable items: did it assert an answer the documents do not support;
- **evidence** on answerable items it refused or answered only in part: did the retrieved passages hold the answer (the report counts this for the refused ones).

The report gives each rate per pass and pooled. Rates: unanswerable declined,
answerable refused, correct / partially correct / incorrect, citation validity,
and answered-vs-refused flips across passes. It also gives p50/p95 latency
(4 questions in flight by default, `--workers`; Agent mode runs one at a time)
and cost per 1,000 questions over all calls (Cohere rerank priced at $2 per
1,000 searches; a failed rerank call is not priced).
`--rejudge <report.json> --judge-model <model>` re-labels saved answers with
another judge, and `--merge` combines reports from separate invocations.

A failed rerank call does not fail the question here either: the pipeline
answers from the first `RERANK_TOP_K` candidates in unranked hybrid order, so
the run no longer measures the configured pipeline. `run_contract.py` counts
failed calls per answer, and by default the first one stops the run:

- questions not yet started are skipped (those in flight finish) and later
  passes do not start;
- nothing is judged; the raw records are saved as
  `results/<stamp>_contract_<label>_aborted.json` with the first error's status
  code and body;
- the script exits 2.

Rate-limit errors (429) are retried with backoff before they count; any other
error, such as a 402 billing cap or a 401 bad key, counts at once.
`--allow-rerank-fallback` keeps going instead. Every report has the row
"answers from unranked passages (rerank failed)", which shows how many answers
used unranked passages; when any did, a warning heads the report.

Published runs:
[`results/README.md`](results/README.md#answer-contract-measured-and-fixed).
