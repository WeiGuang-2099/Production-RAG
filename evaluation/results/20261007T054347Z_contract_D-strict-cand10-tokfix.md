# Answer contract: D-strict-cand10-tokfix

2026-10-07T05:43:47+00:00 | mode `standard` | 3 run(s) | 70 questions (27 unanswerable)

Config: mode=standard, runs=3, top_k=10, rerank_top_k=5, llm_model=gpt-4o, llm_temperature=0.0, prompt_mode=strict, prompt_sha=8c265b584e, retrieval_mode=hybrid, reranker=cohere, graph_extractor=none, keyword_backend=local, collection=rag_docs, data_dir=./data, judge_model=gpt-4o, git_commit=fe5a638, git_dirty=True, note=rerank_failures reconstructed from the run log (the harness did not count them before 4375550); its log has no rerank_failed line

Rates are mean over runs (min-max in parentheses).

| metric | value |
| --- | --- |
| unanswerable: declined (no fabricated answer) | 100.0% |
| unanswerable: exact refusal sentence | 100.0% |
| answerable: refused | 20.9% (16.3%-23.3%) |
| answerable: partial answer | 1.6% (0.0%-2.3%) |
| answerable: correct | 62.0% (60.5%-65.1%) |
| answerable: partially correct | 17.1% (16.3%-18.6%) |
| answerable: incorrect | 0.0% |
| answers with a citation | 100.0% |
| citations in range | 100.0% |
| answered/refused flips across runs | 8.6% |
| latency p50 / p95 (ms) | 1285.6 (1234.2-1328) / 2188.53 (2024.8-2441.9) |
| cost per 1,000 questions (USD) | 8.25 (8.247-8.254) |
| answers from unranked passages (rerank failed) | 0 of 210 |

Answerable questions by type (pooled over runs):

| type | n | refused | correct |
| --- | --- | --- | --- |
| comparative | 24 | 12.5% | 62.5% |
| factual | 45 | 13.3% | 60.0% |
| long_tail | 15 | 20.0% | 80.0% |
| multi_hop | 30 | 50.0% | 36.7% |
| numerical | 15 | 0.0% | 100.0% |

Refused answerable questions, by whether the retrieved passages held the evidence: partial 8, sufficient 19
