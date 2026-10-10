# Evaluation results

Committed evaluation reports. Kinds of report:

- `*_contract_<label>.{md,json}`: answer-contract runs from `run_contract.py` (answer vs refuse,
  judged; every answer saved). Labels ending in `-norerank` mark runs whose rerank calls failed.
- `*_ablation*.{md,json}`: retrieval ablations from `run_ablation.py` (deterministic recall@k /
  MRR / hit@k; no LLM judge).
- `*_<label>.json` (June-July 2026): RAGAS reports from `run_eval.py` (faithfulness /
  answer_relevancy / context_recall / context_precision).
- `*_latency.md`, `*_keyword_backend.md`, `*_multiturn.md`: paired studies (see below).

## Answer contract: measured and fixed

The question the product depends on is whether it answers what the documents support and refuses
what they do not. RAGAS cannot answer it, because it scores every refusal as zero, right or wrong.
`run_contract.py` measures it directly.

**Setup (2026-10-07 and 2026-10-08).**

- **Questions:** 70 in total: the 48 hand-written ones plus the 22 near-miss unanswerable ones
  (`../unanswerable_extra.json`), so 43 answerable and 27 unanswerable. A and B each merge two
  invocations, the 48 questions and then the 22 extra items. B's 48-question part ran on
  2026-10-07 and its 22 extra items on 2026-10-08, with the same strict prompt, temperature 0,
  5 candidates and old tokenizer.
- **Runs:** each run is three passes over the questions, in Standard mode unless noted, on the
  6-paper corpus (`rag_docs`, rebuilt on 2026-10-07). A, B's 48-question part, D, P and E ran on
  2026-10-07 before 06:01 UTC; S5, C, P5, E2, F and B's 22 extra items ran on 2026-10-08. C and
  B's 22 extra items ran from a worktree of 230d254 (the reports record it as 4375550, its hash
  before a message-only reword) with the pre-fix `bm25_store.py` (from f2036a8) and the BM25 index
  saved before the fix.
- **System:** gpt-4o answers, Cohere rerank keeping 5 passages, local BM25, graph expansion off.
- **Judge:** a gpt-4o judge labeled each response: refusal, partial or answer; correct, partially
  correct or incorrect against the reference; and, for unanswerable questions, whether it asserted
  an unsupported answer.
- **Prompt hash:** every report except A records a hash of the prompt text, which is how a
  mislabeled run was caught (below). A and B's 48-question part predate the hash; they ran the
  shipped `grounded` prompt at fe5a638, byte-identical to today's strict prompt.

Every run below declined all 27 unanswerable questions in every pass, and every run had the
reranker working (0 of 210 answers from unranked passages in each report).

| run | prompt | temperature | candidates | BM25 tokenizer | answerable refused | correct | partially correct | wrong | flips | p50 / p95 | cost per 1,000 q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [A](20261007T054354Z_contract_A-shipped.md) (as shipped) | strict | provider default | 5 | old | 31.8% | 55.8% | 10.1% | 2.3% | 15.7% | 1.27s / 2.77s | \$7.98 |
| [B](20261008T052918Z_contract_B-strict-temp0-cand5.md) | strict | 0 | 5 | old | 32.6% | 53.5% | 14.0% | 0.0% | 0.0% | 1.25s / 4.63s | \$7.97 |
| [S5](20261008T050920Z_contract_S-strict-cand5.md) | strict | 0 | 5 | fixed | 31.0% | 58.1% | 10.1% | 0.8% | 7.1% | 1.21s / 3.20s | \$8.00 |
| [C](20261008T052719Z_contract_C-strict-cand10-oldtok.md) | strict | 0 | 10 | old | 20.9% | 62.0% | 15.5% | 1.6% | 4.3% | 1.26s / 4.24s | \$8.23 |
| [D](20261007T054347Z_contract_D-strict-cand10-tokfix.md) | strict | 0 | 10 | fixed | 20.9% | 62.0% | 17.1% | 0.0% | 8.6% | 1.29s / 2.19s | \$8.25 |
| [P](20261007T053458Z_contract_P-partial-cand10-oldtok.md) | partial-answer | 0 | 10 | old | 2.3% | 66.7% | 27.9% | 3.1% | 0.0% | 1.37s / 2.72s | \$8.46 |
| [P5](20261008T050731Z_contract_P-partial-cand5.md) | partial-answer | 0 | 5 | fixed | 3.9% | 67.4% | 24.8% | 3.1% | 1.4% | 1.25s / 3.46s | \$8.28 |
| **[E](20261007T054934Z_contract_E-partial-cand10.md) (the default now)** | partial-answer | 0 | 10 | fixed | 4.7% | 69.8% | 25.6% | 0.0% | 0.0% | 1.30s / 2.72s | \$8.47 |
| [E2](20261008T050538Z_contract_E-partial-cand10-replicate.md) (E, next day) | partial-answer | 0 | 10 | fixed | 4.7% | 71.3% | 23.3% | 0.8% | 0.0% | 1.31s / 3.58s | \$8.47 |
| [F](20261008T051108Z_contract_F-agent-mode.md) | E in Agent mode | 0 | 10 | fixed | 4.7% | 71.3% | 24.0% | 0.0% | 0.0% | 3.39s / 7.58s | \$10.77 |

How to read the table:

- **Rates** are means over the three passes, over the 43 answerable questions. They do not always
  sum to 100%: in P5 the judge labeled one answer (q028, pass 2) "fabricated", which has no column
  here.
- **Exact refusal sentence:** on the 27 unanswerable questions the Standard-mode runs always used
  it; F, in Agent mode, did on 96.3%, wording q053 differently in all three passes (pass 1: "the
  specific snapshot date is not mentioned in the provided documents").
- **Flips** is the share of all 70 questions whose answered-vs-refused outcome changed across the
  three passes. No unanswerable question flipped in any run, so A's 15.7% is 11 of 70.
- **Cost** covers every model call, rerank (Cohere list price, \$2 per 1,000 searches) and query
  embedding.
- **Latency** is per full answer. The 2026-10-07 runs answered one question at a time
  (`--workers 1`); the 2026-10-08 reruns used the default of 4 in flight (`--workers 4`), except
  Agent mode (F), which always runs one question at a time.
- **Run and pass:** here a run is one configuration, three passes over the questions. The reports
  and the `--runs` flag call each pass a run ("3 run(s)", "mean over runs").

What the runs show:

- **Temperature 0 removes the provider-default randomness, not the refusals** (A → B): flips
  15.7% → 0%, refusals 31.8% → 32.6%. At temperature 0 the strict prompt still flipped 0-8.6%
  (B 0%, C 4.3%, S5 7.1%, D 8.6%). The partial-answer prompt flipped 0% in four runs and 1.4% in
  P5 (one question, q004, refused in 2 of 3 passes).
- **Strict prompt: retrieval depth helped, the tokenizer fix did not.** A 2x2 at temperature 0,
  one run (3 passes) per cell: with 5 candidates it refused 32.6% (old tokenizer, B) and 31.0%
  (fixed, S5); with 10, 20.9% (old, C) and 20.9% (fixed, D). Going from 5 to 10 candidates cut
  refusals by about 10-12 points; the tokenizer fix moved them by 0-1.6 points. Why depth helps
  is not settled: the judge's evidence check found the kept passages sufficient for most strict
  refusals at either depth (over three passes: B 29 of 42, S5 22 of 40, C 16 of 27, D 19 of 27).
- **The prompt is the main fix** (D → E, identical retrieval: 10 candidates, fixed tokenizer,
  reranker on). The grounded prompt now answers when the stated facts answer the question in other
  words, answers part of a question while naming what is missing, and refuses only when nothing in
  the context answers. Refused 20.9% → 4.7% (9 → 2 of 43 per pass), correct 62.0% → 69.8%, wrong
  0% → 0%, flips 8.6% → 0%; both declined all 27 unanswerable questions. The two do not add: depth
  moved only the strict prompt (32.6% to 20.9%), while the partial-answer prompt refused about the
  same at either depth (3.9% at 5 candidates, P5; 4.7% at 10, E). At 5 candidates the prompt alone
  took refusals from 31.0% (S5) to 3.9% (P5), so the default's 4.7% does not depend on the depth.
- **The partial-answer prompt is insensitive to depth and tokenizer.** It refused 2.3-4.7% (1-2 of
  43) across P (10 candidates, old tokenizer), P5 (5, fixed), E and E2 (10, fixed). Wrong answers
  were 0-3.1% (E 0%, E2 0.8%, P 3.1%, P5 3.1%).
- **With the reranker working, the default (E) replicated across sessions.** E (2026-10-07) and E2
  (2026-10-08) got the same 5 passages in the same order for 210 of 210 answers and refused the
  same two questions (q024, q047) in every pass. E2 had 71.3% correct and one wrong answer (q028,
  once). The higher refusal rates first recorded for repeats of D (29.5%) and E (9.3%) came from
  runs with the reranker failing (below), not from run-to-run noise.
- **The reranker-off runs are an unplanned answer-level reranker ablation** (one run each, 3
  passes; without the reranker the pipeline keeps the hybrid RRF top 5 of the candidates; table
  below).
  - With 10 candidates, refusals rose without the reranker in all four pairs: strict with the old
    tokenizer (C) 20.9% → 36.4%, strict fixed (D) 20.9% → 29.5%, partial-answer (E) 4.7% → 9.3%,
    Agent (F) 4.7% → 7.0%. Correct fell (62.0% → 48.8%, 62.0% → 53.5%, 69.8% → 62.8%,
    71.3% → 67.4%) and wrong answers rose to 2.3-3.9%. Without the reranker, E's 5 passages
    differed from the reranked ones in 201 of 210 answers.
  - With 5 candidates (P5 vs P5'), both kept the same 5 passages in 210 of 210 answers, in a
    different order in 204: refusals 3.9% vs 4.7%, correct 67.4% vs 69.0%. With as many candidates
    as kept passages the reranker can only reorder, and order alone made no measurable difference.
    The default retrieves 10 candidates and keeps 5 (4cdee63), so the reranker also chooses which
    passages to keep; it is the configuration the retrieval ablation measures.
- **Agent mode matches Standard mode on quality and costs more** (F vs E2, same session, both
  with the reranker working). Refused 4.7% vs 4.7% (q024 and q047 in every pass of both), correct
  71.3% vs 71.3%, wrong 0% vs 0.8%, unanswerable 27/27 declined in both. By type the two are equal
  or close: multi-hop 66.7% correct in both, comparative 45.8% vs 37.5%, factual 75.6% vs 80.0%.
  Latency p50 3.39s vs 1.31s (2.6x), p95 7.58s vs 3.58s; cost \$10.77 vs \$8.47 per 1,000
  questions (1.27x). It routed all 210 answers (70 questions, three passes) to retrieval. All 81 unanswerable passes used
  both query rewrites (7 model calls and 3 reranks each; \$13.33 per 1,000 unanswerable questions
  vs \$9.16 per 1,000 answerable). 126 of 129 answerable passes needed no rewrite; the other 3
  were q024, still refused. The rewrites restate the question as keywords ("How much did it cost
  in USD to train GPT-3?" became "Cost of training GPT-3 in USD"), because the rewriter sees only
  the question, not why the grader rejected the passages. Agent mode stays opt-in.
- **Per question at the default (E, E2, F):** q024 (Transformer vs BERT positional encoding, a
  cross-paper comparison) and q047 (RAG's K, which the paper gives as "k ∈ {5, 10} for training
  and set k for test time using dev data" rather than one value; its reference answer was wrong
  until dataset v1.1) were refused in every pass. q020
  (it names DeBERTa XXL as the largest model LoRA was evaluated on; the answer is GPT-3 175B) was
  wrong in A (1 pass), P (2 passes), P5 (3 passes) and the reranker-off E' and F' (3 passes each),
  never in E, E2 or F. So there is no recurring wrong answer at the default. P refused only q024;
  its q047 answers were judged against the old, wrong reference, once as incorrect (see the
  [dataset changelog](../DATASET_CHANGELOG.md)).
- **Judges:** gpt-4o and gpt-4o-mini agree on every refusal and fabrication label, but on only 81%
  of "correct" vs "partially correct" calls (judge check below).

**Reranker-off runs (2026-10-07, after the billing cap).** Not comparable with the table above.
Each has the configuration of the valid run with the same letter: D' and E' repeated D and E,
while C, P5 and F were run after C', P5' and F', on 2026-10-08.

| run | prompt, candidates, tokenizer | refused | correct | wrong | flips | p50 | answers from unranked passages |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [C'](20261007T062313Z_contract_C-strict-cand10-norerank.md) | strict, 10, old | 36.4% | 48.8% | 3.9% | 2.9% | 1.14s | 210 of 210 |
| [D'](20261007T062831Z_contract_D-strict-cand10-tokfix-replicate-norerank.md) | strict, 10, fixed | 29.5% | 53.5% | 2.3% | 4.3% | 1.16s | 210 of 210 |
| [E'](20261007T063358Z_contract_E-partial-cand10-replicate-norerank.md) | partial-answer, 10, fixed | 9.3% | 62.8% | 2.3% | 0.0% | 1.24s | 210 of 210 |
| [P5'](20261007T063959Z_contract_P-partial-cand5-norerank.md) | partial-answer, 5, fixed | 4.7% | 69.0% | 2.3% | 0.0% | 1.19s | 210 of 210 |
| [F'](20261007T060113Z_contract_F-agent-mode-norerank.md) | Agent, partial-answer, 10, fixed | 7.0% | 67.4% | 2.3% | 0.0% | 4.09s | 198 of 210 |

Their reports price every failed rerank call as if it had succeeded, so their costs are left out.
They record commit 05fd595, the hash of 4cdee63 before two message-only rewords (05fd595 became
4dd89f3, then 4cdee63; the tree is unchanged).

**How the reranker went off, and what changed.** At 06:01:54 UTC on 2026-10-07 the Cohere key hit
its dashboard billing cap (HTTP 402). A 402 is not retried (only 429s are), and the pipeline
catches any reranker error, logs a warning and answers from the unranked hybrid top 5. Five runs
went on that way and their reports looked valid. It was caught only by grepping the run logs for
`rerank_failed`. This was the second silent reranker fallback in the project; the first was the
Cohere trial-key 429s in June (retrieval ablation, below). Commit 230d254 changed the harnesses:

- `run_contract.py` counts failed rerank calls per answer. By default the first failure stops the
  run: unstarted questions are skipped, nothing is judged, the raw records are saved as
  `<stamp>_contract_<label>_aborted.json` with the first error's status code and body, and the
  process exits 2.
- `--allow-rerank-fallback` keeps the run going, and the report states how many answers used
  unranked passages.
- Every report now has the row "answers from unranked passages (rerank failed)". A failed call is
  no longer priced as a rerank search.
- `run_ablation.py` counts failures per stage (a `rerank_failures` column) and, if any call
  failed, exits 2 after saving the report unless `--allow-rerank-fallback` is passed.
- Checked live with an invalid Cohere key (401): Standard and Agent mode both stop in pass 1 and
  exit 2.

The five reranker-off reports are kept under `-norerank` labels, each with a config note and a
failure count reconstructed from its run log: 210 of 210 answers for C', D', E' and P5', 198 of
210 for F'. The valid 2026-10-07 reports got a count of 0 from their logs, which have no
`rerank_failed` line. On 2026-10-08 C, P5, E (as E2) and F were run again with the reranker
working, along with S5 and B's 22 extra items; all six had 0 failed rerank calls.

**A run that was mislabeled, and how it was caught.** P was launched as the strict-prompt step
with 10 candidates. The prompt file was edited while it was queued, so it ran the partial-answer
prompt. Its 2.3% first read as "10 candidates fix the refusals". The prompt hash in its report
(`cfcd8b5d21`, the partial-answer prompt) showed the mismatch, and the strict-prompt controls C
and D (10 candidates) refused 20.9%. P is kept under its true label. B's first invocation on the
22 extra items had the same problem and was dropped; those items were run again with the strict
prompt on 2026-10-08 and merged into B's report.

**Judge check.** gpt-4o-mini re-labeled E's 210 answers
([`20261007T055629Z_contract_E-judge-check-gpt4o-mini.md`](20261007T055629Z_contract_E-judge-check-gpt4o-mini.md)).

- The two judges agreed on every stance (210/210) and every fabrication label (81/81).
- On answers both judged as answered, they agreed on "correct" vs "partially correct" 81% of the
  time (100/123). gpt-4o-mini credited more answers as correct: 76.7% vs 69.8%.
- Every disagreement was between those two labels; no answer one judge called wrong was called
  right by the other. So read "correct" and "partially correct" together.
- Many "partial" labels are right answers phrased differently from the reference. For example, RAG
  described as a retriever plus a generator rather than as parametric plus non-parametric memory.

**Reports** (answers saved; passage text dropped, sources and sha1 kept; each also as `.json`):
[`A`](20261007T054354Z_contract_A-shipped.md),
[`B`](20261008T052918Z_contract_B-strict-temp0-cand5.md),
[`S5`](20261008T050920Z_contract_S-strict-cand5.md),
[`C`](20261008T052719Z_contract_C-strict-cand10-oldtok.md),
[`D`](20261007T054347Z_contract_D-strict-cand10-tokfix.md),
[`P`](20261007T053458Z_contract_P-partial-cand10-oldtok.md),
[`P5`](20261008T050731Z_contract_P-partial-cand5.md),
[`E`](20261007T054934Z_contract_E-partial-cand10.md),
[`E2`](20261008T050538Z_contract_E-partial-cand10-replicate.md),
[`F`](20261008T051108Z_contract_F-agent-mode.md),
[`E judge check`](20261007T055629Z_contract_E-judge-check-gpt4o-mini.md).
Reranker off:
[`C'`](20261007T062313Z_contract_C-strict-cand10-norerank.md),
[`D'`](20261007T062831Z_contract_D-strict-cand10-tokfix-replicate-norerank.md),
[`E'`](20261007T063358Z_contract_E-partial-cand10-replicate-norerank.md),
[`P5'`](20261007T063959Z_contract_P-partial-cand5-norerank.md),
[`F'`](20261007T060113Z_contract_F-agent-mode-norerank.md).
Prompt hashes: strict `8c265b584e` (now `PROMPT_MODE=strict`), partial-answer `cfcd8b5d21`
(`PROMPT_MODE=grounded`, the default).

## Retrieval ablation

```bash
python evaluation/run_ablation.py --k 5      # baseline -> +bm25 -> +rerank -> +graph
```

Current code (2026-10-07, after the tokenizer fix), 6-paper corpus (479 chunks), `top_k=10`,
recall@5 over the reranked top 5, production Cohere key, graph built by the gpt-4o extractor.
Latency is retrieval-only and includes the per-query embedding call. Report:
[`20261007T055713Z_ablation_tokfix-base6.md`](20261007T055713Z_ablation_tokfix-base6.md).

| stage | recall@5 | mrr | hit@5 | p50_ms | p95_ms | count |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (dense)    | 0.934 | 0.979 | 1.000 | 144 | 172 | 48 |
| +bm25 (hybrid RRF)  | 0.972 | 0.965 | 1.000 | 155 | 178 | 48 |
| +rerank (Cohere)    | 0.962 | 0.990 | 1.000 | 452 | 755 | 48 |
| +graph              | 0.910 | 0.938 | 0.958 | 457 | 599 | 48 |

Every question, including the 5 unanswerable ones, carries a related `source_papers` entry, so
retrieval is scored on all 48.

- **Dense retrieval is already near the ceiling.** Six topically distinct papers separate cleanly
  in embedding space: hit@5 1.000 and MRR 0.979 before any other stage. The ablation shows which
  knob moves what.
- **BM25 lifts recall@5** (0.934 → 0.972) with no latency cost; the RRF reshuffle lowers MRR to
  0.965.
- **Reranking puts the best chunk first** (MRR 0.990) for about 0.3s, at a hair of recall (0.962).
- **Graph expansion lowers every metric.** Its hits are extracted triples that cite no document;
  each one in the top 5 pushes out a citable chunk. It is off by default.

Before the tokenizer fix, the same stages measured MRR 0.958 (+bm25), 0.979 (+rerank) and 0.927
(+graph), and +graph recall 0.903; the other recall and hit rates were the same
([`20260622T030819Z_ablation.md`](20260622T030819Z_ablation.md)). The vector store was rebuilt on
2026-10-07, and the pre-fix code first reproduced those numbers exactly
([`20261007T051315Z_ablation_repro-base6.md`](20261007T051315Z_ablation_repro-base6.md)). An
earlier run of the June table was silently degraded by a Cohere trial key's 10 requests/minute
limit (most rerank calls failed and fell back to unranked); rate-limit errors are now retried with
backoff in `app/reranker/reranker.py`, and the numbers above use a production key. The silent
fallback recurred in October (a billing-cap 402, which is not retried; see the answer contract
above). Unless run with `--allow-rerank-fallback`, `run_contract.py` now stops at the first failed
rerank call, and `run_ablation.py` runs every stage, saves the report with a `rerank_failures`
count per stage and then exits 2 if any call failed.

## Scale robustness: 6 papers vs 30 papers

Six well-separated papers leave dense retrieval near the ceiling. To add pressure without changing
the questions, a second corpus adds 24 adversarial look-alike papers, several per ground-truth paper
(RoBERTa/ALBERT/ELECTRA vs BERT; DPR/FiD/REALM/ColBERT vs RAG; adapters/prefix-tuning/QLoRA vs
LoRA; self-consistency/ReAct vs CoT, ...). No look-alike slug appears in any `source_papers`, so
retrieving one is always a scored miss. 30-paper corpus: 2,194 chunks (`rag_docs_scale30`).

```bash
DATA_DIR=./data_scale COLLECTION_NAME=rag_docs_scale30 \
    python evaluation/run_ablation.py --k 5 --label scale30
```

Current code (2026-10-07, after the tokenizer fix; 30-paper report:
[`20261007T055824Z_ablation_tokfix-scale30.md`](20261007T055824Z_ablation_tokfix-scale30.md)):

| stage | recall@5 6p | recall@5 30p | mrr 6p | mrr 30p | hit@5 6p | hit@5 30p |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (dense)   | 0.934 | 0.934 | 0.979 | 0.844 | 1.000 | 1.000 |
| +bm25 (hybrid RRF) | 0.972 | 0.924 | 0.965 | 0.833 | 1.000 | 0.979 |
| +rerank (Cohere)   | 0.962 | 0.920 | 0.990 | 0.862 | 1.000 | 0.979 |
| +graph             | 0.910 | 0.858 | 0.938 | 0.803 | 0.958 | 0.917 |

- **Look-alikes hurt ranking, not recall.** Dense recall@5 is unchanged (0.934) and a relevant
  paper still always makes the top 5 (hit@5 1.000), but MRR falls from 0.979 to 0.844. The
  look-alikes do not push the right paper out of the window; they crowd its top.
- **Reranking is the only stage that beats dense retrieval on MRR at both sizes** (0.990 vs 0.979,
  and 0.862 vs 0.844).
- **BM25 now slightly hurts at 30 papers** (recall@5 0.924 vs 0.934, MRR 0.833 vs 0.844, hit@5
  0.979: one of 48 questions has no relevant paper left in the top 5). The likely reason is keyword
  overlap: a question naming "BERT" also matches the BERT-variant papers. The ablation keeps no
  per-question retrievals, so this is not verified. Before the
  tokenizer fix, a question's last word never matched (it kept its "?"), which hid part of this;
  the pre-fix runs measured BM25 at +0.007 recall over dense and the reranked MRR at 0.877
  ([`20260708T224030Z_ablation_scale30.md`](20260708T224030Z_ablation_scale30.md); the rebuilt store
  reproduced them within 0.003 MRR on one stage,
  [`20261007T051443Z_ablation_repro-scale30.md`](20261007T051443Z_ablation_repro-scale30.md)).
  A corpus full of look-alike documents may favor `RETRIEVAL_MODE=dense` plus reranking. That is
  not measured: the ablation's +rerank stage reranks the hybrid candidates. The answer-contract
  runs above used the 6-paper corpus, where hybrid wins on recall.
- **Graph expansion is worse at scale too** (recall 0.858, MRR 0.803, hit@5 0.917).

## Latency: store caching + parallel retrieval

Measured 2026-07-10. The per-query BM25 unpickle and Qdrant session rebuild were removed: store
factories keep the Qdrant connection, BM25 index and knowledge graph in memory across queries, and
the vector, BM25 and graph legs run in parallel. Score-neutral by construction (recall@5 / MRR /
hit@5 identical before and after on both corpora), while full-pipeline retrieval p50 dropped about
2x (~1.4s → ~0.64s) and p95 1.8-3x. Before/after tables and a Redis cache-hit measurement:
[`20260710T190025Z_latency.md`](20260710T190025Z_latency.md).

## Keyword backend: local vs OpenSearch

Measured 2026-07-10, before the local tokenizer fix. OpenSearch's standard analyzer against the
local `rank_bm25` store on both corpora, testing whether the old `lower().split()` tokenizer caused
BM25's vanishing recall edge at scale. It did not; paired tables:
[`20260710T231042Z_keyword_backend.md`](20260710T231042Z_keyword_backend.md).

## Multi-turn: condense-question recovery

Measured 2026-07-12. Follow-ups with pronouns or ellipsis retrieved against the raw text and missed;
the condense step rewrites them into standalone questions. Three conditions (raw / condensed /
hand-written oracle) on 18 follow-up items, plus the condense step's latency:
[`20260712T014325Z_multiturn.md`](20260712T014325Z_multiturn.md).

## End-to-end RAGAS (June-July 2026, superseded by the answer contract)

```bash
PROMPT_MODE=basic  python evaluation/run_eval.py --label basic
PROMPT_MODE=strict python evaluation/run_eval.py --label grounded   # the prompt these runs used
```

Conditions then: what is now the strict prompt (it was called `grounded`), 5 candidates and 5 kept,
graph expansion on, provider-default temperature. Answers by gpt-4o, judge gpt-4o-mini. 48/48
succeeded for both prompts.

| prompt | faithfulness | answer_relevancy | context_recall | context_precision |
| --- | --- | --- | --- | --- |
| basic | 0.878 | 0.838 | 0.896 | 0.794 |
| grounded (strict) | 0.534 | 0.521 | 0.844 | 0.790 |

The strict prompt scored lower on faithfulness and answer_relevancy. RAGAS gives a refusal 0 on
both metrics whether or not refusing was right, and the strict prompt refused all 5 unanswerable
questions. Those five explain about a quarter of the faithfulness gap. Most of the rest sits on 15
answerable questions with the same zero/zero refusal pattern: over-refusal, which the answer
contract above traced to the strict refusal rule and, in part, to choosing the 5 kept passages
from only 5 candidates. An earlier reading on this page attributed the gap
to "RAGAS penalizing correct refusals" alone, and stated that the basic prompt answered all 5
unanswerable questions; the run's answers were not saved, and RAGAS flagged 3 of the basic
prompt's 5 answers (q039-q041) as noncommittal, so that 0/5 cannot be confirmed from the committed
files.

`context_recall` and `context_precision` are nearly identical across prompts, so retrieval was the
same in both runs and the gap is generation behavior.

Per-type faithfulness (basic → strict):

| type | basic | strict |
| --- | --- | --- |
| factual | 0.893 | 0.467 |
| multi_hop | 0.791 | 0.500 |
| comparative | 0.955 | 0.609 |
| numerical | 1.000 | 1.000 |
| unanswerable | 0.787 | 0.000 |
| long_tail | 0.853 | 0.750 |

### Tuning: RERANK_TOP_K and multi-hop over-refusal

The first run (then-default `RERANK_TOP_K=3`) refused 6 of the 10 multi_hop questions, because the
3 kept passages did not hold all the evidence. Refusals on
those 10 vs the number of passages kept:

| RERANK_TOP_K | 3 | 5 | 8 |
| --- | --- | --- | --- |
| multi_hop refused | 6/10 | 4/10 | 3/10 |

Keeping 5 passages (the default since) cut multi-hop refusals and lifted multi-hop faithfulness
from 0.150 to 0.500, with correct cited answers (q017 "BERT bidirectional vs GPT left-only [2]";
q019 "BART [3]"). The answer contract later found that
the larger lever was the refusal rule itself.

### Generation at scale (30 papers)

RAGAS with the strict prompt on the 30-paper corpus
([`20260708T231722Z_grounded_scale30.json`](20260708T231722Z_grounded_scale30.json)): faithfulness
0.625, answer_relevancy 0.603, context_recall 0.865, context_precision 0.803. All 5 unanswerable
questions were still refused ("I cannot answer this from the provided documents", verified by
reading regenerated answers). Multi-hop refusals were 4-5/10 vs 4/10 at 6 papers; q019 had the
refusal pattern in the scored run but got a correct cited answer ("BART-large [3]") when
regenerated, an early sign of the answer/refusal flips across passes that run A measured (15.7% at
the provider-default temperature).
