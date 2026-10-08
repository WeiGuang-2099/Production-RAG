# Answer contract: E-partial-cand10-replicate

2026-10-08T05:05:38+00:00 | mode `standard` | 3 run(s) | 70 questions (27 unanswerable)

Config: mode=standard, runs=3, top_k=10, rerank_top_k=5, llm_model=gpt-4o, llm_temperature=0.0, prompt_mode=grounded, prompt_sha=cfcd8b5d21, retrieval_mode=hybrid, reranker=cohere, graph_extractor=none, keyword_backend=local, collection=rag_docs, data_dir=./data, judge_model=gpt-4o, git_commit=4375550, git_dirty=True

Rates are mean over runs (min-max in parentheses).

| metric | value |
| --- | --- |
| unanswerable: declined (no fabricated answer) | 100.0% |
| unanswerable: exact refusal sentence | 100.0% |
| answerable: refused | 4.7% |
| answerable: partial answer | 0.0% |
| answerable: correct | 71.3% (69.8%-72.1%) |
| answerable: partially correct | 23.3% (20.9%-25.6%) |
| answerable: incorrect | 0.8% (0.0%-2.3%) |
| answers with a citation | 100.0% |
| citations in range | 100.0% |
| answered/refused flips across runs | 0.0% |
| latency p50 / p95 (ms) | 1307.73 (1234.5-1347.1) / 3576.73 (2219-6153.9) |
| cost per 1,000 questions (USD) | 8.468 (8.459-8.481) |
| answers from unranked passages (rerank failed) | 0 of 210 |

Answerable questions by type (pooled over runs):

| type | n | refused | correct |
| --- | --- | --- | --- |
| comparative | 24 | 0.0% | 37.5% |
| factual | 45 | 0.0% | 80.0% |
| long_tail | 15 | 20.0% | 80.0% |
| multi_hop | 30 | 10.0% | 66.7% |
| numerical | 15 | 0.0% | 100.0% |

Refused answerable questions, by whether the retrieved passages held the evidence: partial 3, sufficient 3
