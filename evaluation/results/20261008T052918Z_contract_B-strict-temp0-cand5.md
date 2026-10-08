# Answer contract: B-strict-temp0-cand5

2026-10-08T05:29:18+00:00 | mode `standard` | 3 run(s) | 70 questions (27 unanswerable)

Config: mode=standard, runs=3, top_k=5, rerank_top_k=5, llm_model=gpt-4o, llm_temperature=0.0, prompt_mode=grounded, retrieval_mode=hybrid, reranker=cohere, graph_extractor=none, keyword_backend=local, collection=rag_docs, data_dir=./data, judge_model=gpt-4o, git_commit=fe5a638, git_dirty=True, prompt_sha=8c265b584e, merged_from=two invocations: the 48-question set (2026-10-07, fe5a638) and the 22 extra unanswerable items (2026-10-08, 4375550), note=the 48-question part predates prompt_sha; its prompt (grounded at fe5a638) is byte-identical to PROMPT_MODE=strict (8c265b584e); the 22-item part used the pre-fix BM25 tokenizer: run from a worktree of 4375550 with app/retrieval/bm25_store.py taken from f2036a8 and DATA_DIR pointing at the index saved before the fix; 48-question part: rerank_failures reconstructed from the run log (the harness did not count them before 4375550); its log has no rerank_failed line

Rates are mean over runs (min-max in parentheses).

| metric | value |
| --- | --- |
| unanswerable: declined (no fabricated answer) | 100.0% |
| unanswerable: exact refusal sentence | 100.0% |
| answerable: refused | 32.6% |
| answerable: partial answer | 0.0% |
| answerable: correct | 53.5% |
| answerable: partially correct | 14.0% |
| answerable: incorrect | 0.0% |
| answers with a citation | 100.0% |
| citations in range | 100.0% |
| answered/refused flips across runs | 0.0% |
| latency p50 / p95 (ms) | 1246.37 (1197.3-1285.4) / 4633.4 (2023.8-9705.9) |
| cost per 1,000 questions (USD) | 7.9667 (7.962-7.971) |
| answers from unranked passages (rerank failed) | 0 of 210 |

Answerable questions by type (pooled over runs):

| type | n | refused | correct |
| --- | --- | --- | --- |
| comparative | 24 | 25.0% | 37.5% |
| factual | 45 | 40.0% | 46.7% |
| long_tail | 15 | 20.0% | 80.0% |
| multi_hop | 30 | 50.0% | 40.0% |
| numerical | 15 | 0.0% | 100.0% |

Refused answerable questions, by whether the retrieved passages held the evidence: partial 13, sufficient 29
