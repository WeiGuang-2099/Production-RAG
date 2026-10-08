# Answer contract: E-judge-check-gpt4o-mini

2026-10-07T05:56:29+00:00 | mode `standard` | 3 run(s) | 70 questions (27 unanswerable)

Config: mode=standard, runs=3, top_k=10, rerank_top_k=5, llm_model=gpt-4o, llm_temperature=0.0, prompt_mode=grounded, prompt_sha=cfcd8b5d21, retrieval_mode=hybrid, reranker=cohere, graph_extractor=none, keyword_backend=local, collection=rag_docs, data_dir=./data, judge_model=gpt-4o-mini, git_commit=fe5a638, git_dirty=True, rejudged_from=20261007T054934Z_contract_E-prompt.json, note=rerank_failures reconstructed from the run log (the harness did not count them before 4375550); its log has no rerank_failed line

Rates are mean over runs (min-max in parentheses).

| metric | value |
| --- | --- |
| unanswerable: declined (no fabricated answer) | 100.0% |
| unanswerable: exact refusal sentence | 100.0% |
| answerable: refused | 4.7% |
| answerable: partial answer | 0.0% |
| answerable: correct | 76.7% (74.4%-79.1%) |
| answerable: partially correct | 18.6% (16.3%-20.9%) |
| answerable: incorrect | 0.0% |
| answers with a citation | 100.0% |
| citations in range | 100.0% |
| answered/refused flips across runs | 0.0% |
| latency p50 / p95 (ms) | 1302.33 (1279-1327.4) / 2720.67 (2324.8-3348.7) |
| cost per 1,000 questions (USD) | 8.4723 (8.465-8.48) |
| answers from unranked passages (rerank failed) | 0 of 210 |

Answerable questions by type (pooled over runs):

| type | n | refused | correct |
| --- | --- | --- | --- |
| comparative | 24 | 0.0% | 95.8% |
| factual | 45 | 0.0% | 73.3% |
| long_tail | 15 | 20.0% | 80.0% |
| multi_hop | 30 | 10.0% | 53.3% |
| numerical | 15 | 0.0% | 100.0% |

Refused answerable questions, by whether the retrieved passages held the evidence: sufficient 6
