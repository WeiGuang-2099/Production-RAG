# Answer contract: P-partial-cand10-oldtok

2026-10-07T05:34:58+00:00 | mode `standard` | 3 run(s) | 70 questions (27 unanswerable)

Config: mode=standard, runs=3, top_k=10, rerank_top_k=5, llm_model=gpt-4o, llm_temperature=0.0, prompt_mode=grounded, prompt_sha=cfcd8b5d21, retrieval_mode=hybrid, reranker=cohere, graph_extractor=none, keyword_backend=local, collection=rag_docs, data_dir=./data, judge_model=gpt-4o, git_commit=fe5a638, git_dirty=True, note=launched as the strict-prompt step; the recorded prompt_sha (cfcd8b5d21) shows it ran the partial-answer prompt; rerank_failures reconstructed from the run log (the harness did not count them before 4375550); its log has no rerank_failed line

Rates are mean over runs (min-max in parentheses).

| metric | value |
| --- | --- |
| unanswerable: declined (no fabricated answer) | 100.0% |
| unanswerable: exact refusal sentence | 100.0% |
| answerable: refused | 2.3% |
| answerable: partial answer | 0.0% |
| answerable: correct | 66.7% (65.1%-69.8%) |
| answerable: partially correct | 27.9% (23.3%-30.2%) |
| answerable: incorrect | 3.1% (2.3%-4.7%) |
| answers with a citation | 100.0% |
| citations in range | 100.0% |
| answered/refused flips across runs | 0.0% |
| latency p50 / p95 (ms) | 1372.8 (1267.3-1511.5) / 2719.97 (2238.5-3064.3) |
| cost per 1,000 questions (USD) | 8.459 (8.444-8.473) |
| answers from unranked passages (rerank failed) | 0 of 210 |

Answerable questions by type (pooled over runs):

| type | n | refused | correct |
| --- | --- | --- | --- |
| comparative | 24 | 0.0% | 41.7% |
| factual | 45 | 0.0% | 73.3% |
| long_tail | 15 | 0.0% | 80.0% |
| multi_hop | 30 | 10.0% | 53.3% |
| numerical | 15 | 0.0% | 100.0% |

Refused answerable questions, by whether the retrieved passages held the evidence: sufficient 3
