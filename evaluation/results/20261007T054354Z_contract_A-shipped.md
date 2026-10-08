# Answer contract: A-shipped

2026-10-07T05:43:54+00:00 | mode `standard` | 3 run(s) | 70 questions (27 unanswerable)

Config: mode=standard, runs=3, top_k=5, rerank_top_k=5, llm_model=gpt-4o, llm_temperature=None, prompt_mode=grounded, retrieval_mode=hybrid, reranker=cohere, graph_extractor=none, keyword_backend=local, collection=rag_docs, data_dir=./data, judge_model=gpt-4o, git_commit=fe5a638, git_dirty=True, merged_from=two invocations: the 48-question set and the 22 extra unanswerable items, note=rerank_failures reconstructed from the run log (the harness did not count them before 4375550); its log has no rerank_failed line

Rates are mean over runs (min-max in parentheses).

| metric | value |
| --- | --- |
| unanswerable: declined (no fabricated answer) | 100.0% |
| unanswerable: exact refusal sentence | 100.0% |
| answerable: refused | 31.8% (25.6%-34.9%) |
| answerable: partial answer | 0.8% (0.0%-2.3%) |
| answerable: correct | 55.8% (51.2%-62.8%) |
| answerable: partially correct | 10.1% (7.0%-11.6%) |
| answerable: incorrect | 2.3% (0.0%-4.7%) |
| answers with a citation | 100.0% |
| citations in range | 100.0% |
| answered/refused flips across runs | 15.7% |
| latency p50 / p95 (ms) | 1271.87 (1248.8-1311.9) / 2773.6 (2049.6-3967.5) |
| cost per 1,000 questions (USD) | 7.975 (7.973-7.977) |
| answers from unranked passages (rerank failed) | 0 of 210 |

Answerable questions by type (pooled over runs):

| type | n | refused | correct |
| --- | --- | --- | --- |
| comparative | 24 | 29.2% | 37.5% |
| factual | 45 | 37.8% | 53.3% |
| long_tail | 15 | 26.7% | 73.3% |
| multi_hop | 30 | 43.3% | 43.3% |
| numerical | 15 | 0.0% | 100.0% |

Refused answerable questions, by whether the retrieved passages held the evidence: partial 15, sufficient 26
