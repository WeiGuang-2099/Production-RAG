# Case study: measuring and fixing a cite-or-refuse RAG system

What this project is about: not wiring up RAG, but running it against live APIs, finding where it
breaks, and fixing that, with the numbers to show it. There were three rounds: the first eval
(June 2026); scale, latency and multi-turn studies (July); and an answer-contract eval with fixes
(October). Every eval number below comes from a report in
[`evaluation/results/`](../evaluation/results/README.md), except two that were measured live but not
saved as reports: the Quick start refusals in Finding 3 and the ingest time under Cost and latency.

## The system under test

Hybrid retrieval (Qdrant vectors plus BM25, fused with RRF), Cohere rerank, and generation that
must cite its passages or refuse. A LangGraph corrective-RAG agent (route, retrieve, grade, rewrite)
is the opt-in second mode, and edge guardrails screen input and output. It also has provider-agnostic
factories, per-answer cost estimates, an opt-in semantic cache, an MCP server, a React UI and a
Docker Compose stack. A lightweight knowledge-graph expansion exists but is off by default
(Finding 2).

## How it is measured

- **Corpora:** 6 classic ML papers (Attention, BERT, GPT-3, RAG, LoRA, CoT), 479 chunks. A second
  corpus adds 24 adversarial look-alike papers (RoBERTa and ALBERT next to BERT, DPR and FiD next to
  RAG, QLoRA next to LoRA, ...): 2,194 chunks.
- **Questions:** 48 hand-written questions across 6 types, each tagged with its source paper(s);
  22 near-miss questions on topics the papers cover but facts they never state, each kept only
  after two independent checks of the chunk text found no answer; and 18 follow-up questions for
  multi-turn.
- **Harnesses:**
  - `run_ablation.py`: deterministic recall@k / MRR / hit@k per retrieval stage, no judge.
  - `run_contract.py`: does it answer what the documents support and refuse the rest? It saves
    every answer, and a gpt-4o judge labels refusals, correctness and fabrication, cross-checked
    against gpt-4o-mini.
  - `run_eval.py`: RAGAS, kept for history.
  - `run_multiturn.py`: raw vs rewritten vs hand-written follow-ups.

## Finding 1: the refusal problem, and the wrong first explanation

The June RAGAS run scored the cite-or-refuse prompt far below a basic prompt (faithfulness 0.534 vs
0.878). RAGAS scores every refusal as zero, right or wrong, so the first reading was that the metric
penalized correct refusals. That explains only about a quarter of the gap: the 5 correct refusals.
Most of the rest came from 15 answerable questions that scored zero on both faithfulness and answer
relevancy, the score a refusal gets; the harness did not save answers, so that could not be checked.

The answer-contract harness (October) saves every answer and has a judge label it. On 70 questions
(43 answerable, 27 unanswerable; three passes per run), the shipped configuration refused 31.8% of
answerable questions. Because no temperature was set, 15.7% of questions flipped between an answer
and a refusal across passes.

What fixed it, one change at a time. Each configuration below is one run of three passes, except
the default, which ran twice (step 4). Every run declined all 27 unanswerable questions in every
pass, and every rate is the mean of its three passes.

1. **Temperature 0 removed the provider-default randomness, not the refusals:** in the shipped
   setup, flips fell from 15.7% to 0%, and refusals went from 31.8% to 32.6%.
2. **Under the strict prompt, 10 candidates instead of 5 cut refusals to 20.9%** (from 31.0-32.6%).
   Why is not settled: the judge found the evidence already in the kept passages for most strict
   refusals at either depth (29 of 42 at 5 candidates, 16 of 27 at 10, old tokenizer). The BM25
   tokenizer fix did not help: in a 2x2 of depth and tokenizer (one run per cell) it moved refusals
   by 0-1.6 points.
3. **The partial-answer prompt took refusals to 4.7%.** On identical retrieval (10 candidates,
   fixed tokenizer) it cut refusals from 20.9% to 4.7% (9 to 2 of 43 per pass) and raised correct
   answers from 62.0% to 69.8%, with no wrong answers in either run. It is insensitive to depth and
   tokenizer: 2.3-4.7% refused in all four runs.
4. **It replicated the next day with identical passages:** the same 5 passages in the same order
   for all 210 answers, the same two refusals in every pass, 4.7% refused, 71.3% correct and one
   wrong answer in one pass.

| answerable refused, temperature 0 | 5 candidates, old tokenizer | 5, fixed | 10, old | 10, fixed (default) |
| --- | --- | --- | --- | --- |
| strict: refuse unless the context holds "enough information" | 32.6% | 31.0% | 20.9% | 20.9% |
| partial-answer: answer what is stated, name what is missing, refuse only when nothing answers | not run | 3.9% | 2.3% | 4.7% (4.7% next day) |

At temperature 0, wrong answers stayed low: 0-1.6% under the strict prompt, 0-3.1% under the
partial-answer one (0% and 0.8% in the two default runs). The strict prompt still flipped on 0-8.6%
of questions; the partial-answer prompt flipped once, on one question in the 5-candidate run.

The October numbers had a wrong first explanation too, and how it was caught is the lesson. One
run, launched as
"strict prompt, 10 candidates", refused only 2.3%, which read as "the shipped retrieval depth was
the cause". But the prompt file had been edited while that run was queued. The prompt hash recorded
in its report showed it had run the new prompt, and control runs with the strict prompt and 10
candidates (old and fixed tokenizer) both refused 20.9%. Depth mattered only under the strict
prompt (32.6% to 20.9%); the partial-answer prompt refused 3.9% at 5 candidates and 4.7% at 10, so
at the default the prompt does the work. Two habits came out of it:

- check the prompt hash and config recorded in each report against the run's label before
  reading its numbers;
- before acting on a surprising number, run a control of the configuration it claims to be.

The second lesson came the same day. Repeats of two configurations, run under an hour after the
originals, refused more (strict 29.5% vs 20.9%, partial-answer 9.3% vs 4.7%), which read as
run-to-run noise. It was the reranker: the Cohere key had hit its billing cap in between, and the
pipeline answered from unranked passages while the reports looked valid (Finding 3). Rerun the next
day with the reranker working, the partial-answer repeat retrieved the same passages in the same
order as the original for all 210 answers and refused the same two questions (4.7% both times). A
run whose dependency fell back is not the configuration it claims, which is why the harness now
stops on a failed rerank call.

Still unsolved under the new default:

- **Still refused, in every pass of Standard and Agent mode:** a cross-paper comparison
  (Transformer vs BERT positional encoding, q024), and a question that presupposes one value of K
  while the paper uses several (5 or 10 in training, 15 or 50 at test for open-domain QA; q047,
  whose reference answer was also wrong until dataset v1.1).
- **No recurring wrong answer at the default.** Two recurred elsewhere. q020 names DeBERTa XXL as
  the largest model LoRA was evaluated on; the answer is GPT-3 175B. It appeared with the old
  tokenizer, with 5 candidates and with the reranker off, never in the three default runs. q028
  (LoRA vs full fine-tuning) was judged wrong in 12 passes across 8 runs, once at the default (E2),
  each time because the answer gave figures the reference does not contain; similar answers were
  judged partially correct in other passes.
- **Judge reliability:** two judges (gpt-4o and gpt-4o-mini) agree on every refusal and fabrication
  label, but on only 81% of "correct" vs "partially correct" calls.

## Finding 2: look-alike documents hurt ranking, not recall

Growing the corpus 4.6x with look-alike papers left dense recall@5 unchanged (0.934, and a relevant
paper always in the top 5) while MRR fell from 0.979 to 0.844: the look-alikes crowd the top of the
window rather than pushing the right paper out of it. Reranking is the only stage that beats dense
retrieval on MRR at both sizes (0.990 and 0.862, current code).

Component value depends on the corpus. BM25 lifts recall@5 at 6 papers (0.934 to 0.972) but slightly lowers it
at 30 (0.924 vs 0.934 dense; one question of 48 loses its paper from the top 5), likely because a
question naming "BERT" also matches the look-alike papers that mention BERT (not checked question by
question). Graph expansion
lowers every metric at both sizes: its hits are extracted triples that cite no document, so each
one in the top 5 pushes out a citable passage. It is off by default; the evidence changed the
default.

The reranker-off runs from October (Finding 1) also form an unplanned answer-level ablation of the
reranker. It is one run per configuration, three passes each, not a planned experiment. With 10
candidates, answering from the unranked hybrid top 5 raised refusals in all four pairs (strict
prompt 20.9% to 36.4% with the old tokenizer and to 29.5% with the fixed one, partial-answer 4.7%
to 9.3%, Agent mode 4.7% to 7.0%), lowered correct answers in all four and raised wrong answers to
2.3-3.9%. With 5 candidates the reranker can only reorder the same 5 passages, and order alone made
no measurable difference (3.9% refused with it, 4.7% without). That supports the default of
retrieving 10 candidates and keeping 5, which was set to match the retrieval ablation.

## Finding 3: bugs that only live runs found

Mocked unit tests (346 now) never touch a real API or a real container. Running the system for real
found:

- **June:**
  - The graph extractor crashed on null triples, dropping a paper's whole graph.
  - RAGAS stopped importing after a LangChain major-version drift (fixed with a small
    compatibility shim).
  - Cohere 429s on a trial key made the reranker fall back silently to unranked results, which
    quietly contaminated the first ablation. Rate-limit errors now retry with backoff.
- **October:**
  - The reranker fell back silently again. The Cohere key hit its billing cap (HTTP 402), which,
    unlike a 429, is not retried, and five answer-contract runs went on with the reranker off.
    Their reports looked valid; grepping the run logs for `rerank_failed` caught it. Since 230d254
    the answer-contract harness stops at the first failed rerank call and the ablation exits 2 on
    any, unless `--allow-rerank-fallback` is passed, and every report counts the answers that came
    from unranked passages. The affected configurations (strict and partial-answer at 10
    candidates, partial-answer at 5, Agent mode) were rerun with the reranker working; the strict
    repeat's configuration already had a valid run from before the cap.
  - The Docker Compose stack never started: Qdrant's health check called `curl`, which the image
    does not ship.
  - The app's logs were never emitted: modules logged through stdlib loggers, but only structlog
    was configured.
  - The BM25 tokenizer kept punctuation glued to words, so every question's last word ("bert?")
    never matched. Fixing it improved MRR at 6 papers and hurt at 30, because correct keyword
    matching surfaces look-alikes.
  - The Quick start's example question ("What does the Transformer eliminate?", with only that
    paper ingested) was refused in 8 of 11 live tries: the passage stating the answer missed the
    top 5. The Quick start now asks the fuller question from the chat screenshot, which answered 5
    of 5 with a citation (a hand check noted in commit eea7ec3, not a saved report).
  - The price table billed a dated gpt-4o-mini id at gpt-4o rates.
  - The shipped retrieval depth differed from the one every retrieval eval measured.

## Finding 4: Agent mode does not pay for itself on this question set

Run at the default in the same session as Standard mode, both with the reranker working, Agent mode
matched it on quality. Both refused 4.7% of answerable questions (the same two, in every pass) and
answered 71.3% correctly; Agent mode gave no wrong answer, Standard mode one in one pass. Both
declined all 27 unanswerable questions, though on one Agent mode explained the gap instead of using
the exact refusal sentence. By question type the two are equal or close: multi-hop 66.7% correct
in both, comparative 45.8% vs 37.5%, factual 75.6% vs 80.0%. Agent mode took 2.6x the latency (p50
3.39s vs 1.31s, p95 7.58s vs 3.58s; Agent mode runs one question at a time, Standard mode had 4 in
flight) at 1.27x the cost (\$10.77 vs \$8.47 per 1,000 questions).

Where the extra cost goes: it routed all 210 answers (70 questions, three passes) to retrieval.
All 81 unanswerable passes used both query rewrites (7 model calls and 3 reranks each; \$13.33 per
1,000 unanswerable questions vs \$9.16 per 1,000 answerable) to reach the refusal Standard mode
gives after one retrieval. Of 129 answerable passes, 126 needed no rewrite, and the other 3 were the
cross-paper comparison, still refused. The rewrites restate the question as keywords ("How much
did it cost in USD to train GPT-3?" became "Cost of training GPT-3 in USD") because the rewriter
sees only the question, not why the grader rejected the passages. Agent mode stays opt-in. The next
fix would pass the grader's reason to the rewriter.

## Cost and latency

- **Per question, every call priced** (models, Cohere reranks, query embeddings): \$8.47 per
  1,000 questions in Standard mode and \$10.77 in Agent mode. The in-app per-answer figure covers
  only the answer call and any follow-up rewrite.
- **Full answers:** Standard mode p50 1.3s, p95 2.7-3.6s across two sessions, measured with 4
  questions in flight. Agent mode, one question at a time: p50 3.4s.
- **Retrieval:** the July change that cached stores and parallelized the retrieval legs halved
  retrieval latency (p50 ~1.4s to ~0.64s) with bit-identical retrieval scores.
- **Ingest:** graph extraction is one gpt-4o call per chunk; 479 chunks took ~32 minutes and were
  the dominant spend in June. It is off by default now.

## What is still open

The README's
[production rollout list](../README.md#what-a-production-rollout-would-still-need) covers the
product gaps: deletion, access control, SSRF, monitoring and load. On the evaluation side:

- the judge has been cross-checked against a second model, not against human labels;
- 27 unanswerable questions is still a small sample;
- everything is English ML papers;
- and nothing yet verifies at runtime that a cited passage supports its sentence.

## Reproduce

```bash
docker-compose up -d qdrant
python evaluation/corpus/download_papers.py
GRAPH_EXTRACTOR=llm python evaluation/ingest_corpus.py --force   # graph is opt-in
python evaluation/run_ablation.py --k 5
python evaluation/run_contract.py --label final                  # answer vs refuse, 70 x 3
python evaluation/run_contract.py --mode agent --label agent
PROMPT_MODE=strict LLM_TEMPERATURE=default python evaluation/run_contract.py --top-k 5 --label before
# (the last line is the shipped setup, except that BM25 now uses the fixed tokenizer)
```
