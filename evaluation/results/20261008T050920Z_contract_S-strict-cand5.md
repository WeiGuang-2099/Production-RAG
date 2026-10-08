# Answer contract: S-strict-cand5

2026-10-08T05:09:20+00:00 | mode `standard` | 3 run(s) | 70 questions (27 unanswerable)

Config: mode=standard, runs=3, top_k=5, rerank_top_k=5, llm_model=gpt-4o, llm_temperature=0.0, prompt_mode=strict, prompt_sha=8c265b584e, retrieval_mode=hybrid, reranker=cohere, graph_extractor=none, keyword_backend=local, collection=rag_docs, data_dir=./data, judge_model=gpt-4o, git_commit=4375550, git_dirty=True

Rates are mean over runs (min-max in parentheses).

| metric | value |
| --- | --- |
| unanswerable: declined (no fabricated answer) | 100.0% |
| unanswerable: exact refusal sentence | 100.0% |
| answerable: refused | 31.0% (27.9%-32.6%) |
| answerable: partial answer | 0.0% |
| answerable: correct | 58.1% (55.8%-60.5%) |
| answerable: partially correct | 10.1% (4.7%-14.0%) |
| answerable: incorrect | 0.8% (0.0%-2.3%) |
| answers with a citation | 100.0% |
| citations in range | 100.0% |
| answered/refused flips across runs | 7.1% |
| latency p50 / p95 (ms) | 1207.3 (1161.3-1268) / 3202.6 (1697.2-6129.3) |
| cost per 1,000 questions (USD) | 8.0027 (7.994-8.014) |
| answers from unranked passages (rerank failed) | 0 of 210 |

Answerable questions by type (pooled over runs):

| type | n | refused | correct |
| --- | --- | --- | --- |
| comparative | 24 | 37.5% | 25.0% |
| factual | 45 | 22.2% | 66.7% |
| long_tail | 15 | 20.0% | 80.0% |
| multi_hop | 30 | 60.0% | 40.0% |
| numerical | 15 | 0.0% | 100.0% |

Refused answerable questions, by whether the retrieved passages held the evidence: partial 18, sufficient 22
