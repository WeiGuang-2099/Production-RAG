# Answer contract: D-strict-cand10-tokfix-replicate-norerank

2026-10-07T06:28:31+00:00 | mode `standard` | 3 run(s) | 70 questions (27 unanswerable)

**Rerank failed for 210 of 210 answers (210 calls). Those answers used unranked passages, so this run does not measure the configured pipeline.**

Config: mode=standard, runs=3, top_k=10, rerank_top_k=5, llm_model=gpt-4o, llm_temperature=0.0, prompt_mode=strict, prompt_sha=8c265b584e, retrieval_mode=hybrid, reranker=cohere, graph_extractor=none, keyword_backend=local, collection=rag_docs, data_dir=./data, judge_model=gpt-4o, git_commit=05fd595, git_dirty=True, note=reranker off: the Cohere key hit its billing cap (HTTP 402) at 06:01:54 UTC on 2026-10-07, and the pipeline fell back to the unranked candidates; rerank_failures reconstructed from the run log (the harness did not count them before 4375550); cost prices every rerank call as if it had succeeded

Rates are mean over runs (min-max in parentheses).

| metric | value |
| --- | --- |
| unanswerable: declined (no fabricated answer) | 100.0% |
| unanswerable: exact refusal sentence | 100.0% |
| answerable: refused | 29.5% (27.9%-32.6%) |
| answerable: partial answer | 0.0% |
| answerable: correct | 53.5% (51.2%-55.8%) |
| answerable: partially correct | 14.7% (14.0%-16.3%) |
| answerable: incorrect | 2.3% (0.0%-4.7%) |
| answers with a citation | 100.0% |
| citations in range | 100.0% |
| answered/refused flips across runs | 4.3% |
| latency p50 / p95 (ms) | 1158.47 (1131.8-1201.3) / 2462.87 (2003.8-3033.8) |
| cost per 1,000 questions (USD) | 8.1937 (8.192-8.197) |
| answers from unranked passages (rerank failed) | 210 of 210 |

Answerable questions by type (pooled over runs):

| type | n | refused | correct |
| --- | --- | --- | --- |
| comparative | 24 | 25.0% | 29.2% |
| factual | 45 | 22.2% | 53.3% |
| long_tail | 15 | 40.0% | 60.0% |
| multi_hop | 30 | 53.3% | 46.7% |
| numerical | 15 | 0.0% | 100.0% |

Refused answerable questions, by whether the retrieved passages held the evidence: absent 6, partial 15, sufficient 17
