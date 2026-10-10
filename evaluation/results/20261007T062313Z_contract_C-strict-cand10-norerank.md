# Answer contract: C-strict-cand10-norerank

2026-10-07T06:23:13+00:00 | mode `standard` | 3 run(s) | 70 questions (27 unanswerable)

**Rerank failed for 210 of 210 answers (210 calls). Those answers used unranked passages, so this run does not measure the configured pipeline.**

Config: mode=standard, runs=3, top_k=10, rerank_top_k=5, llm_model=gpt-4o, llm_temperature=0.0, prompt_mode=strict, prompt_sha=8c265b584e, retrieval_mode=hybrid, reranker=cohere, graph_extractor=none, keyword_backend=local, collection=rag_docs, data_dir=<outside the repo>/data_v1, judge_model=gpt-4o, git_commit=05fd595, git_dirty=True, note=reranker off: the Cohere key hit its billing cap (HTTP 402) at 06:01:54 UTC on 2026-10-07, and the pipeline fell back to the unranked candidates; rerank_failures reconstructed from the run log (the harness did not count them before 4375550); cost prices every rerank call as if it had succeeded

Rates are mean over runs (min-max in parentheses).

| metric | value |
| --- | --- |
| unanswerable: declined (no fabricated answer) | 100.0% |
| unanswerable: exact refusal sentence | 100.0% |
| answerable: refused | 36.4% (34.9%-37.2%) |
| answerable: partial answer | 0.0% |
| answerable: correct | 48.8% |
| answerable: partially correct | 10.8% (9.3%-11.6%) |
| answerable: incorrect | 3.9% (2.3%-4.7%) |
| answers with a citation | 100.0% |
| citations in range | 100.0% |
| answered/refused flips across runs | 2.9% |
| latency p50 / p95 (ms) | 1143.7 (1100.4-1183.5) / 2280.1 (1859.2-2879.9) |
| cost per 1,000 questions (USD) | 8.0473 (8.044-8.053) |
| answers from unranked passages (rerank failed) | 210 of 210 |

Answerable questions by type (pooled over runs):

| type | n | refused | correct |
| --- | --- | --- | --- |
| comparative | 24 | 29.2% | 25.0% |
| factual | 45 | 42.2% | 46.7% |
| long_tail | 15 | 40.0% | 60.0% |
| multi_hop | 30 | 50.0% | 40.0% |
| numerical | 15 | 0.0% | 100.0% |

Refused answerable questions, by whether the retrieved passages held the evidence: absent 3, partial 21, sufficient 23
