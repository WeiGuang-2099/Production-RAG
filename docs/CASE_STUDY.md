# Case study: measuring and hardening a production RAG system

This is the short version of what the project is actually about: not "I wired up
RAG," but "I measured it against real keys, found where it breaks, and fixed
that." The numbers below are real runs (2026-06-22), not illustrations.

## The system under test

Hybrid retrieval (dense vectors + BM25 with RRF fusion + a lightweight knowledge
graph) -> Cohere rerank -> grounded, cited, *cite-or-refuse* generation, fronted
by a corrective-RAG agent (route / retrieve / grade / rewrite) and edge
guardrails (prompt-injection block, PII redaction, toxicity flag). Provider-
agnostic factories, per-query token/cost accounting, a semantic cache, Dockerized,
215 mocked unit tests.

## How it is measured

- **Corpus:** 6 classic ML papers (Attention, BERT, GPT-3, RAG, LoRA, CoT) — 479
  chunks. Chosen so multi-hop questions can legitimately span papers.
- **Dataset:** 48 hand-written questions across 6 types (factual, multi_hop,
  comparative, numerical, unanswerable, long_tail), each tagged with its
  ground-truth source paper(s).
- **Two harnesses:** `run_ablation.py` (deterministic recall@k / MRR / hit@k — no
  LLM judge, cheap) and `run_eval.py` (RAGAS faithfulness / answer_relevancy /
  context_recall / context_precision; judge = gpt-4o-mini for cost, answers from
  gpt-4o).

## Finding 1 — retrieval: dense is already near-ceiling; BM25 earns its place

| stage | recall@5 | mrr | hit@5 |
| --- | --- | --- | --- |
| baseline (dense)   | 0.934 | 0.979 | 1.000 |
| +bm25 (hybrid RRF) | **0.972** | 0.958 | 1.000 |
| +rerank (Cohere)   | 0.962 | **0.979** | 1.000 |
| +graph             | 0.903 | 0.927 | 0.958 |

Six topically distinct papers separate cleanly in embedding space, so dense
retrieval already lands a correct paper in the top-5 every time (hit@5 = 1.000).
With almost no recall headroom, the ablation measures *which knob moves what*:
**+BM25 maximizes recall@5** (0.934 -> 0.972, no latency cost); **+rerank
maximizes MRR** (0.958 -> 0.979 — it puts the single best chunk first, ~0.7s added
p95), which is what actually matters feeding a top-3 context window; and **+graph
hurts here** (recall 0.903), because cross-paper expansion adds noise on a corpus
this small.

I would rather report "graph doesn't pay off at this scale" than manufacture a
monotonic "every component helps" table. (These rerank/graph numbers are from a
production Cohere key — an earlier run was silently degraded by a *trial* key's
rate limit, which is itself Finding 3.)

## Finding 2 — generation: the eval inverted "grounded is more faithful," and the inversion is the lesson

RAGAS, both prompts, full 48 questions (identical retrieval; gpt-4o answers,
gpt-4o-mini judge):

| prompt | faithfulness | answer_relevancy | context_recall | context_precision |
| --- | --- | --- | --- | --- |
| basic | 0.789 | 0.887 | 0.842 | 0.802 |
| grounded | 0.532 | 0.501 | 0.837 | 0.825 |

I expected the grounded (cite-or-refuse) prompt to *win* on faithfulness; it scored
markedly lower. Why is the whole point — and it is two different things, which I
verified by reading the actual answers:

1. **RAGAS punishes correct refusals.** On the 5 `unanswerable` questions the
   grounded prompt correctly replies "I cannot answer this from the provided
   documents," which RAGAS scores `answer_relevancy = 0.000`. The metric that
   actually matters — does it refuse when it should? — says the opposite:

   | refusal rate on the 5 unanswerable | basic | grounded |
   | --- | --- | --- |
   | refuses (correct) | 0/5 | **5/5** |

   Basic answers all five questions that have no answer in the corpus; grounded
   refuses all five. That is the behavior you want, and standard faithfulness /
   answer_relevancy penalize it because they reward confident answering.

2. **Grounded over-refuses on hard *answerable* questions** (`multi_hop`
   faithfulness 0.150). Error analysis, q016 — "Which architecture underlies both
   BERT and GPT-3?" (answer: the Transformer): basic answers correctly; grounded
   refuses, because the top-3 retrieved chunks don't co-locate the cross-paper
   evidence. A real retrieval-depth limit, not a metric artifact — and it points at
   a concrete fix (raise `RERANK_TOP_K` for multi-hop).

`context_recall` / `context_precision` are nearly identical across prompts, which
confirms retrieval was the same and the whole gap is generation behavior.

The takeaway is something you only learn by running the eval and reading the
outputs: **standard RAGAS faithfulness/answer_relevancy are the wrong yardstick for
a refusal-capable RAG system.** Taken naively they say "turn off grounding" — the
exact opposite of what a system that must not hallucinate should do. The right
measure is refusal-segmented: refusal accuracy on unanswerable, answer accuracy on
answerable.

## Finding 3 — three production bugs you only find by actually running it

The eval did more than produce numbers; running it end-to-end against real APIs
surfaced three bugs that 215 *mocked* unit tests never could:

1. **Graph extraction crashed on null triples.** The LLM occasionally returns
   `{"head": null, ...}`; the parser only checked that keys existed, so
   `networkx.add_edge(None, ...)` raised "None cannot be a node" and dropped the
   entire paper's graph. Fixed test-first (reject empty head/relation/tail, plus
   a defensive guard in the store).
2. **RAGAS would not import.** The environment had drifted to the langchain v1
   line (langchain 1.3.x / community 0.4.x), and ragas 0.4.3 hard-imports a
   VertexAI wrapper that v1 removed. Rather than downgrade the whole stack (which
   the app no longer targets), I added a tiny, documented compatibility shim that
   stubs the two unused symbols.
3. **The reranker failed silently under rate limits.** A Cohere `429` raised
   straight through and the pipeline quietly fell back to *unranked* results — and
   this silently contaminated my first ablation run (most rerank calls were
   429ing). The fix: retry with bounded backoff on rate-limit errors, then degrade
   gracefully. The lesson is the real deliverable here — *measure, and trust no
   result until you have checked its failure path.*

## Cost and latency reality

- **Retrieval latency** (includes the per-query embedding call): p50 ~1.0s, p95
  ~1.7s for dense/hybrid; reranking adds ~0.7s (p50 1.7s, p95 2.1s).
- **End-to-end answer latency** (gpt-4o, basic prompt): mean 3.3s, p50 2.9s, p95
  5.6s.
- **Ingestion cost lesson:** graph extraction fires *one gpt-4o call per chunk* —
  479 sequential calls, ~32 minutes, and it was the dominant spend (enough to
  exhaust the OpenAI budget mid-eval). The next iteration moves extraction to
  gpt-4o-mini with bounded concurrency: roughly 15x cheaper and 10x faster, for a
  graph this lightweight. Knowing exactly where the money went is the point.

## Honest limitations / what is next

- **Data-driven next steps this eval surfaced:** (1) grounded over-refuses on
  multi-hop because the top-3 context is too shallow — raise `RERANK_TOP_K` or add
  multi-hop-aware retrieval; (2) add refusal-segmented eval metrics (refusal
  accuracy on unanswerable, answer accuracy on answerable), since RAGAS
  faithfulness/relevancy mislead on a cite-or-refuse system; (3) move graph
  extraction to gpt-4o-mini + bounded concurrency (~15x cheaper, ~10x faster).
- **Known architectural limits, called out not hidden:** GraphRAG is intentionally
  lightweight (LLM/NER triples + lexical matching) and did not pay off at this
  scale; the semantic cache is process-local; BM25 rebuilds per ingest.
- **Still to build:** a live hosted demo, and a load test reporting p95 latency and
  $/1k queries.

## Reproduce

```bash
docker-compose up -d qdrant
python evaluation/corpus/download_papers.py
python evaluation/ingest_corpus.py --force
python evaluation/run_ablation.py --k 5
PROMPT_MODE=basic    python evaluation/run_eval.py --label basic
PROMPT_MODE=grounded python evaluation/run_eval.py --label grounded
```

Full per-type tables and the saved reports live in
[`evaluation/results/`](../evaluation/results/README.md).
