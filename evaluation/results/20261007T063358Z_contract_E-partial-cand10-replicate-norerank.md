# Answer contract: E-partial-cand10-replicate-norerank

2026-10-07T06:33:58+00:00 | mode `standard` | 3 run(s) | 70 questions (27 unanswerable)

**Rerank failed for 210 of 210 answers (210 calls). Those answers used unranked passages, so this run does not measure the configured pipeline.**

Config: mode=standard, runs=3, top_k=10, rerank_top_k=5, llm_model=gpt-4o, llm_temperature=0.0, prompt_mode=grounded, prompt_sha=cfcd8b5d21, retrieval_mode=hybrid, reranker=cohere, graph_extractor=none, keyword_backend=local, collection=rag_docs, data_dir=./data, judge_model=gpt-4o, git_commit=05fd595, git_dirty=True, note=reranker off: the Cohere key hit its billing cap (HTTP 402) at 06:01:54 UTC on 2026-10-07, and the pipeline fell back to the unranked candidates; rerank_failures reconstructed from the run log (the harness did not count them before 4375550); cost prices every rerank call as if it had succeeded

Rates are mean over runs (min-max in parentheses).

| metric | value |
| --- | --- |
| unanswerable: declined (no fabricated answer) | 100.0% |
| unanswerable: exact refusal sentence | 96.3% |
| answerable: refused | 9.3% |
| answerable: partial answer | 0.0% |
| answerable: correct | 62.8% |
| answerable: partially correct | 25.6% |
| answerable: incorrect | 2.3% |
| answers with a citation | 100.0% |
| citations in range | 100.0% |
| answered/refused flips across runs | 0.0% |
| latency p50 / p95 (ms) | 1237 (1230.2-1249.6) / 3566.77 (3285.1-4125.8) |
| cost per 1,000 questions (USD) | 8.4297 (8.42-8.446) |
| answers from unranked passages (rerank failed) | 210 of 210 |

Answerable questions by type (pooled over runs):

| type | n | refused | correct |
| --- | --- | --- | --- |
| comparative | 24 | 0.0% | 37.5% |
| factual | 45 | 6.7% | 60.0% |
| long_tail | 15 | 40.0% | 60.0% |
| multi_hop | 30 | 10.0% | 70.0% |
| numerical | 15 | 0.0% | 100.0% |

Refused answerable questions, by whether the retrieved passages held the evidence: absent 6, partial 6
