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

RAGAS, both prompts, full 48 questions (`RERANK_TOP_K=5`; gpt-4o answers,
gpt-4o-mini judge):

| prompt | faithfulness | answer_relevancy | context_recall | context_precision |
| --- | --- | --- | --- | --- |
| basic | 0.878 | 0.838 | 0.896 | 0.794 |
| grounded | 0.534 | 0.521 | 0.844 | 0.790 |

I expected the grounded (cite-or-refuse) prompt to *win* on faithfulness; it scored
markedly lower. Why is the whole point — and reading the actual answers split it
into a metric artifact and a real bug I then fixed:

1. **RAGAS punishes correct refusals (artifact).** On the 5 `unanswerable`
   questions the grounded prompt correctly replies "I cannot answer this from the
   provided documents," which RAGAS scores `answer_relevancy = 0.000`. The metric
   that actually matters — does it refuse when it should? — says the opposite:

   | refusal rate on the 5 unanswerable | basic | grounded |
   | --- | --- | --- |
   | refuses (correct) | 0/5 | **5/5** |

   Basic answers all five questions that have no answer in the corpus; grounded
   refuses all five. Standard faithfulness/answer_relevancy penalize that because
   they reward confident answering.

2. **Grounded over-refused answerable multi-hop (real bug, now fixed).** At the
   first run's `RERANK_TOP_K=3`, grounded refused 6/10 answerable multi-hop
   questions — the top-3 context didn't co-locate the cross-paper evidence.
   Measuring refusal rate against the final context size made the fix obvious:

   | RERANK_TOP_K | 3 | 5 | 8 |
   | --- | --- | --- | --- |
   | multi_hop refused | 6/10 | 4/10 | 3/10 |

   Raising the context 3 → 5 chunks (now the default) lifted grounded multi-hop
   faithfulness **0.150 → 0.500**, with correct cited answers (q017 "BERT
   bidirectional vs GPT left-only [2]"; q019 "BART [3]"). Honest limit: true
   *synthesis* questions (q016) still refuse even at k=8 — strict grounding won't
   assert a fact no single chunk states.

`context_recall` / `context_precision` are nearly identical across prompts, which
confirms retrieval was the same and the gap is generation behavior.

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

## Finding 4 — at 5x adversarial scale, ranking degrades, recall does not — and the reranker becomes the component that earns its keep

Finding 1's caveat was that six well-separated papers leave dense retrieval
near the ceiling. So I grew the corpus 4.6x (479 -> 2,194 chunks) with 24
*adversarial* distractors — papers a retriever plausibly confuses with the
ground truth (RoBERTa/ALBERT vs BERT, DPR/FiD/REALM vs RAG, adapters/QLoRA vs
LoRA, self-consistency/ReAct vs CoT) — and reran the identical 48 questions.
Same-day paired runs, same machine and keys (2026-07-08 UTC):

| stage | recall@5 (6p) | recall@5 (30p) | mrr (6p) | mrr (30p) |
| --- | --- | --- | --- | --- |
| baseline (dense)   | 0.934 | 0.934 | 0.979 | 0.844 |
| +bm25 (hybrid RRF) | 0.972 | 0.941 | 0.958 | 0.839 |
| +rerank (Cohere)   | 0.962 | 0.934 | 0.979 | 0.877 |
| +graph             | 0.903 | 0.872 | 0.927 | 0.815 |

Three things the paired table says that neither corpus says alone:

1. **Adversarial density attacks ranking, not recall.** Dense recall@5 is
   unchanged and hit@5 stays 1.000 — the right paper still makes the top-5 —
   but MRR collapses 0.979 -> 0.844 because confusable neighbors crowd the
   top ranks. For a system feeding a tight context window, MRR is the metric
   that predicts answer quality, and it is the one that broke.
2. **Component value is corpus-dependent.** BM25, the recall hero at 6 papers
   (+0.038), is nearly worthless at 30 (+0.007) and now costs a hit@5 —
   keyword overlap is exactly what BERT-variants share with BERT. The
   reranker moves the other way: its MRR contribution roughly doubles
   (+0.021 -> +0.038) and it repairs the hit BM25 lost. Tuning the stack on
   the small corpus alone would have overvalued BM25 and undervalued the
   reranker.
3. **Graph keeps not paying off, more so** (recall 0.872, MRR 0.815, hit@5
   0.938) — lexical entity expansion is the wrong tool when the corpus is
   full of near-topic neighbors by design.

The scale run also killed a latency hypothesis: p50 stayed ~1.0s (dense) /
~1.4s (reranked) at 4.6x the chunks, so retrieval latency is still dominated
by the embedding API round-trip, not local index work — which reprioritizes
the optimization backlog toward parallelizing the retrieval legs over caching
the local indexes.

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

- **Already fixed from this eval's findings:** raised `RERANK_TOP_K` 3 → 5 (now the
  default), which cut grounded's multi-hop over-refusal 6/10 → 4/10 and lifted
  multi-hop faithfulness 0.150 → 0.500.
- **Surfaced, still open:** add refusal-segmented eval metrics (refusal accuracy on
  unanswerable, answer accuracy on answerable), since RAGAS faithfulness/relevancy
  mislead on a cite-or-refuse system; multi-hop-aware retrieval for the true-
  synthesis questions that still refuse even at k=8; and move graph extraction to
  gpt-4o-mini + bounded concurrency (~15x cheaper, ~10x faster).
- **Known architectural limits, called out not hidden:** GraphRAG is intentionally
  lightweight (LLM/NER triples + lexical matching) and did not pay off at either
  corpus scale (6 or 30 papers — Finding 4); the semantic cache is process-local;
  BM25 rebuilds per ingest.
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
