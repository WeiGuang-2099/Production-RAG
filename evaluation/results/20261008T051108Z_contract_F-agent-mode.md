# Answer contract: F-agent-mode

2026-10-08T05:11:08+00:00 | mode `agent` | 3 run(s) | 70 questions (27 unanswerable)

Config: mode=agent, runs=3, top_k=10, rerank_top_k=5, llm_model=gpt-4o, llm_temperature=0.0, prompt_mode=grounded, prompt_sha=cfcd8b5d21, retrieval_mode=hybrid, reranker=cohere, graph_extractor=none, keyword_backend=local, collection=rag_docs, data_dir=./data, judge_model=gpt-4o, git_commit=4375550, git_dirty=True, note=git_commit and git_dirty filled in after the run: the harness's git calls returned nothing (the process had outlived its parent shell); the reflog shows HEAD at 4375550 from 05:05 to 05:36 UTC, and the run went from 05:11 to 05:27 UTC with uncommitted doc changes

Rates are mean over runs (min-max in parentheses).

| metric | value |
| --- | --- |
| unanswerable: declined (no fabricated answer) | 100.0% |
| unanswerable: exact refusal sentence | 96.3% |
| answerable: refused | 4.7% |
| answerable: partial answer | 0.0% |
| answerable: correct | 71.3% (69.8%-72.1%) |
| answerable: partially correct | 24.0% (23.3%-25.6%) |
| answerable: incorrect | 0.0% |
| answers with a citation | 100.0% |
| citations in range | 100.0% |
| answered/refused flips across runs | 0.0% |
| latency p50 / p95 (ms) | 3388 (3334.5-3468.3) / 7576.2 (7282.2-8106.7) |
| cost per 1,000 questions (USD) | 10.7713 (10.766-10.776) |
| answers from unranked passages (rerank failed) | 0 of 210 |

Answerable questions by type (pooled over runs):

| type | n | refused | correct |
| --- | --- | --- | --- |
| comparative | 24 | 0.0% | 45.8% |
| factual | 45 | 0.0% | 75.6% |
| long_tail | 15 | 20.0% | 80.0% |
| multi_hop | 30 | 10.0% | 66.7% |
| numerical | 15 | 0.0% | 100.0% |

Refused answerable questions, by whether the retrieved passages held the evidence: partial 3, sufficient 3

Agent routes: retrieve 210
