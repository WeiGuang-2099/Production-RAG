# Answer contract: F-agent-mode-norerank

2026-10-07T06:01:13+00:00 | mode `agent` | 3 run(s) | 70 questions (27 unanswerable)

**Rerank failed for 198 of 210 answers (372 calls). Those answers used unranked passages, so this run does not measure the configured pipeline.**

Config: mode=agent, runs=3, top_k=10, rerank_top_k=5, llm_model=gpt-4o, llm_temperature=0.0, prompt_mode=grounded, prompt_sha=cfcd8b5d21, retrieval_mode=hybrid, reranker=cohere, graph_extractor=none, keyword_backend=local, collection=rag_docs, data_dir=./data, judge_model=gpt-4o, git_commit=05fd595, git_dirty=True, note=reranker off: the Cohere key hit its billing cap (HTTP 402) at 06:01:54 UTC on 2026-10-07, and the pipeline fell back to the unranked candidates; rerank_failures reconstructed from the run log (the harness did not count them before 4375550); cost prices every rerank call as if it had succeeded

Rates are mean over runs (min-max in parentheses).

| metric | value |
| --- | --- |
| unanswerable: declined (no fabricated answer) | 100.0% |
| unanswerable: exact refusal sentence | 100.0% |
| answerable: refused | 7.0% |
| answerable: partial answer | 0.0% |
| answerable: correct | 67.4% |
| answerable: partially correct | 23.3% |
| answerable: incorrect | 2.3% |
| answers with a citation | 100.0% |
| citations in range | 100.0% |
| answered/refused flips across runs | 0.0% |
| latency p50 / p95 (ms) | 4086.8 (3694.3-4568) / 7232.9 (6932.1-7737.6) |
| cost per 1,000 questions (USD) | 10.72 (10.681-10.74) |
| answers from unranked passages (rerank failed) | 198 of 210 |

Answerable questions by type (pooled over runs):

| type | n | refused | correct |
| --- | --- | --- | --- |
| comparative | 24 | 0.0% | 37.5% |
| factual | 45 | 0.0% | 66.7% |
| long_tail | 15 | 40.0% | 60.0% |
| multi_hop | 30 | 10.0% | 80.0% |
| numerical | 15 | 0.0% | 100.0% |

Refused answerable questions, by whether the retrieved passages held the evidence: absent 6, partial 3

Agent routes: retrieve 210
