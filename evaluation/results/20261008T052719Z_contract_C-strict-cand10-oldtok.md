# Answer contract: C-strict-cand10-oldtok

2026-10-08T05:27:19+00:00 | mode `standard` | 3 run(s) | 70 questions (27 unanswerable)

Config: mode=standard, runs=3, top_k=10, rerank_top_k=5, llm_model=gpt-4o, llm_temperature=0.0, prompt_mode=strict, prompt_sha=8c265b584e, retrieval_mode=hybrid, reranker=cohere, graph_extractor=none, keyword_backend=local, collection=rag_docs, data_dir=docs/superpowers/evidence/2026-10-07-contract-runs/data_v1, judge_model=gpt-4o, git_commit=4375550, git_dirty=True, note=pre-fix BM25 tokenizer: run from a worktree of 4375550 with app/retrieval/bm25_store.py taken from f2036a8 and DATA_DIR pointing at the index saved before the fix

Rates are mean over runs (min-max in parentheses).

| metric | value |
| --- | --- |
| unanswerable: declined (no fabricated answer) | 100.0% |
| unanswerable: exact refusal sentence | 100.0% |
| answerable: refused | 20.9% (18.6%-23.3%) |
| answerable: partial answer | 0.0% |
| answerable: correct | 62.0% (60.5%-65.1%) |
| answerable: partially correct | 15.5% (14.0%-18.6%) |
| answerable: incorrect | 1.6% (0.0%-2.3%) |
| answers with a citation | 100.0% |
| citations in range | 100.0% |
| answered/refused flips across runs | 4.3% |
| latency p50 / p95 (ms) | 1257.6 (1211.9-1306.5) / 4235.77 (1870-8621.9) |
| cost per 1,000 questions (USD) | 8.233 (8.228-8.241) |
| answers from unranked passages (rerank failed) | 0 of 210 |

Answerable questions by type (pooled over runs):

| type | n | refused | correct |
| --- | --- | --- | --- |
| comparative | 24 | 25.0% | 50.0% |
| factual | 45 | 0.0% | 73.3% |
| long_tail | 15 | 20.0% | 80.0% |
| multi_hop | 30 | 60.0% | 26.7% |
| numerical | 15 | 0.0% | 100.0% |

Refused answerable questions, by whether the retrieved passages held the evidence: partial 11, sufficient 16
