# Answer contract: P-partial-cand5

2026-10-08T05:07:31+00:00 | mode `standard` | 3 run(s) | 70 questions (27 unanswerable)

Config: mode=standard, runs=3, top_k=5, rerank_top_k=5, llm_model=gpt-4o, llm_temperature=0.0, prompt_mode=grounded, prompt_sha=cfcd8b5d21, retrieval_mode=hybrid, reranker=cohere, graph_extractor=none, keyword_backend=local, collection=rag_docs, data_dir=./data, judge_model=gpt-4o, git_commit=4375550, git_dirty=True

Rates are mean over runs (min-max in parentheses).

| metric | value |
| --- | --- |
| unanswerable: declined (no fabricated answer) | 100.0% |
| unanswerable: exact refusal sentence | 100.0% |
| answerable: refused | 3.9% (2.3%-4.7%) |
| answerable: partial answer | 0.0% |
| answerable: correct | 67.4% |
| answerable: partially correct | 24.8% (20.9%-27.9%) |
| answerable: incorrect | 3.1% (2.3%-4.7%) |
| answers with a citation | 100.0% |
| citations in range | 100.0% |
| answered/refused flips across runs | 1.4% |
| latency p50 / p95 (ms) | 1247.77 (1228.1-1279.1) / 3460.67 (2124.7-5948.3) |
| cost per 1,000 questions (USD) | 8.282 (8.278-8.287) |
| answers from unranked passages (rerank failed) | 0 of 210 |

Answerable questions by type (pooled over runs):

| type | n | refused | correct |
| --- | --- | --- | --- |
| comparative | 24 | 0.0% | 37.5% |
| factual | 45 | 4.4% | 73.3% |
| long_tail | 15 | 0.0% | 80.0% |
| multi_hop | 30 | 10.0% | 60.0% |
| numerical | 15 | 0.0% | 100.0% |

Refused answerable questions, by whether the retrieved passages held the evidence: partial 3, sufficient 2
