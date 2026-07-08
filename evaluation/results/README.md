# Evaluation results

This directory holds committed evaluation reports — the evidence that the
system works and that each component earns its place. Two kinds of report
land here:

- `*_ablation.{md,json}` — retrieval ablation from `run_ablation.py`
  (deterministic recall@k / MRR / hit@k; no LLM judge, cheap to run).
- `*_<label>.json` — full RAGAS reports from `run_eval.py`
  (faithfulness / answer_relevancy / context_recall / context_precision).

## Retrieval ablation

Regenerate (corpus must be ingested first — see ../README.md):

```bash
python evaluation/run_ablation.py --k 5      # baseline -> +bm25 -> +rerank -> +graph
```

Real run, 2026-06-22 · 6-paper corpus (479 chunks) · `top_k=10`, recall@5 over the
reranked top-5 · production Cohere key · graph built by the gpt-4o extractor.
Latency is retrieval-only but includes the per-query embedding API round-trip.

| stage | recall@5 | mrr | hit@5 | p50_ms | p95_ms | count |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (dense)    | 0.934 | 0.979 | 1.000 | 1133 | 1783 | 48 |
| +bm25 (hybrid RRF)  | 0.972 | 0.958 | 1.000 | 1039 | 1640 | 48 |
| +rerank (Cohere)    | 0.962 | 0.979 | 1.000 | 1747 | 2067 | 48 |
| +graph              | 0.903 | 0.927 | 0.958 | 1773 | 2185 | 48 |

(`count` = 48: every question, including the 5 `unanswerable` ones, carries a
related `source_papers` entry, so retrieval is scored on all 48.)

What this honestly shows on this corpus:

- **Dense retrieval is already near-ceiling.** Six topically distinct papers
  separate cleanly in embedding space, so baseline already has hit@5 1.000 and
  MRR 0.979 — the ablation is really measuring *which knob moves what*.
- **+BM25 maximizes recall@5** (0.934 -> 0.972) at no latency cost; keyword signal
  recovers the few queries dense missed, though the RRF reshuffle nudges MRR down
  to 0.958.
- **+rerank maximizes MRR** (0.958 -> 0.979): it trades a hair of recall (0.962)
  to put the single best chunk first, for ~0.7s of added p95 — exactly what you
  want feeding a top-3 context window.
- **+graph hurts here** (recall 0.903, hit@5 0.958): cross-paper expansion adds
  noise on a small, well-separated corpus. Flagged honestly rather than hidden —
  it matches the "graph is intentionally lightweight" caveat in the root README;
  its payoff would be on larger, noisier corpora.

(An earlier run of this table was silently degraded by a Cohere *trial* key's
10 req/min limit — most rerank calls 429'd and fell back to unranked. That silent
failure is now handled with retry + backoff in `app/reranker/reranker.py`; the
numbers above are from a production key.)

## Scale robustness: 6 papers vs 30 papers

The 6-paper ablation has a known blind spot: six topically distinct papers
separate so cleanly in embedding space that dense retrieval starts near the
ceiling (hit@5 = 1.000), so the table above measures *which knob moves what*,
not where quality comes from under pressure. To create that pressure without
touching the dataset, a second corpus adds 24 **adversarial distractor**
papers — for every ground-truth paper, several that a retriever plausibly
confuses with it (RoBERTa/ALBERT/ELECTRA vs BERT; DPR/FiD/REALM/ColBERT vs
RAG; adapters/prefix-tuning/QLoRA vs LoRA; self-consistency/ReAct vs CoT...).
The 48 questions and metrics are unchanged; no distractor slug appears in any
`source_papers`, so retrieving one is always a scored miss.

Both runs 2026-07-08 (UTC), same session, machine, and keys; `top_k=10`,
recall@5 over the reranked top-5. 6-paper corpus: 479 chunks (`rag_docs`);
30-paper corpus: 2,194 chunks (`rag_docs_scale30`); both graphs built by the
gpt-4o extractor. Regenerate with:

```bash
python evaluation/run_ablation.py --k 5 --label base6
DATA_DIR=./data_scale COLLECTION_NAME=rag_docs_scale30 \
    python evaluation/run_ablation.py --k 5 --label scale30
```

| stage | recall@5 6p | recall@5 30p | mrr 6p | mrr 30p | hit@5 6p | hit@5 30p | p50_ms 6p | p50_ms 30p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (dense)   | 0.934 | 0.934 | 0.979 | 0.844 | 1.000 | 1.000 | 1015 | 1000 |
| +bm25 (hybrid RRF) | 0.972 | 0.941 | 0.958 | 0.839 | 1.000 | 0.979 | 1010 | 1041 |
| +rerank (Cohere)   | 0.962 | 0.934 | 0.979 | 0.877 | 1.000 | 1.000 | 1351 | 1399 |
| +graph             | 0.903 | 0.872 | 0.927 | 0.815 | 0.958 | 0.938 | 1400 | 1395 |

What 4.6x adversarial scale actually changed:

- **The failure mode is ranking, not recall.** Dense recall@5 did not move
  (0.934 -> 0.934) and a correct paper still always makes the top-5
  (hit@5 = 1.000) — but MRR collapsed 0.979 -> 0.844. Distractors do not push
  the right paper out of the window; they crowd the top of it. A system
  serving a tight top-k context feels this directly.
- **BM25's recall edge nearly vanishes** (+0.038 at 6p -> +0.007 at 30p) and
  its RRF reshuffle now costs a hit (hit@5 0.979): keyword overlap is exactly
  what confusable papers share (every BERT-variant paper is dense with
  "BERT"), so at scale the keyword channel pulls distractors in. A
  component's value is corpus-dependent — the 6p and 30p columns genuinely
  disagree about BM25.
- **The reranker's contribution roughly doubles, making it the component
  that earns its keep at scale.** At 6p it added +0.021 MRR; at 30p it adds
  +0.038 (0.839 -> 0.877) and repairs the hit@5 that BM25 lost (0.979 ->
  1.000). It mitigates but does not cancel the crowding: best-achievable MRR
  still fell 0.979 -> 0.877.
- **Graph's verdict is unchanged and amplified** (recall 0.872, MRR 0.815,
  hit@5 0.938): lexical entity expansion pulls in near-topic chunks, which is
  precisely what the distractor set is made of.
- **Latency is flat at 4.6x** (p50 ~1.0s dense, ~1.4s reranked at both
  scales): retrieval time is still dominated by the per-query embedding API
  round-trip, not local index work. The known per-query BM25 unpickle /
  Qdrant session rebuild is a scaling liability, but at 2.2k chunks it is not
  yet the bottleneck.

## End-to-end RAGAS

```bash
PROMPT_MODE=basic    python evaluation/run_eval.py --label basic
PROMPT_MODE=grounded python evaluation/run_eval.py --label grounded
```

Answers generated by gpt-4o; judge = gpt-4o-mini (`LLM_MODEL_FAST`, for cost).
48/48 succeeded for both prompts, production keys, `RERANK_TOP_K=5`.

| prompt | faithfulness | answer_relevancy | context_recall | context_precision |
| --- | --- | --- | --- | --- |
| basic | **0.878** | 0.838 | 0.896 | 0.794 |
| grounded | 0.534 | 0.521 | 0.844 | 0.790 |

**Read this carefully — the naive expectation is wrong, and *why* is the point.**
Grounded scores *lower* on faithfulness and answer_relevancy. That is not grounded
being worse; it is RAGAS penalizing the right behavior:

**RAGAS penalizes correct refusals.** The grounded prompt is cite-or-refuse; on the
5 `unanswerable` questions it correctly says "I cannot answer this from the
provided documents." RAGAS scores that refusal `answer_relevancy = 0.000` (a
refusal is not "relevant" to the question), so doing the right thing tanks the
metric. Measured directly:

| refusal rate on the 5 unanswerable | basic | grounded |
| --- | --- | --- |
| refuses (correct) | **0/5** | **5/5** |

Basic answers all five questions that have no answer in the corpus; grounded
refuses all five. Refusal accuracy is the metric that matters here, and grounded
wins it 5–0 — while RAGAS faithfulness/relevancy say the opposite.

`context_recall` / `context_precision` are ~identical across prompts (0.896/0.794
vs 0.844/0.790) — retrieval is the same in both runs, so the gap is generation
behavior, not retrieval.

**Per-type faithfulness (basic → grounded):**

| type | basic | grounded | what's happening |
| --- | --- | --- | --- |
| factual | 0.893 | 0.467 | grounded refuses when context lacks the exact fact |
| multi_hop | 0.791 | 0.500 | improved by the RERANK_TOP_K fix below (was 0.150) |
| comparative | 0.955 | 0.609 | grounded more conservative |
| numerical | 1.000 | 1.000 | both solid |
| unanswerable | 0.787 | 0.000 | grounded refuses all 5 correctly; RAGAS scores it 0 |
| long_tail | 0.853 | 0.750 | both solid |

### Tuning: RERANK_TOP_K and multi-hop over-refusal

The first run (then-default `RERANK_TOP_K=3`) exposed a *real* second effect,
separate from the metric artifact: grounded was over-refusing answerable
multi-hop questions, because the top-3 context didn't co-locate the cross-paper
evidence. Refusal rate on the 10 (answerable) multi_hop questions vs the final
context size:

| RERANK_TOP_K | 3 | 5 | 8 |
| --- | --- | --- | --- |
| multi_hop refused (lower is better) | 6/10 | 4/10 | 3/10 |

Raising the final context 3 → 5 chunks (**now the default**) cut the over-refusal
and lifted grounded multi_hop faithfulness **0.150 → 0.500** (relevancy 0.141 →
0.463), with correct, cited answers (e.g. q017 "BERT bidirectional vs GPT
left-only [2]"; q019 "BART [3]"). The trade-off is honest: a little more context
slightly lowers faithfulness on single-fact buckets, so grounded *overall* is ~flat
while multi-hop jumps. True cross-paper *synthesis* (q016) still refuses even at
k=8 — strict grounding won't assert a fact no single chunk states.

**Takeaway (the senior point):** standard RAGAS faithfulness/answer_relevancy
reward confident answering and punish honest refusal, so they are the *wrong*
yardstick for a cite-or-refuse system — you need refusal-segmented metrics
(refusal accuracy on unanswerable, answer accuracy on answerable). And the eval
didn't just measure: it surfaced a concrete over-refusal issue, and the fix
(`RERANK_TOP_K` 3 → 5) is already in the numbers above.
