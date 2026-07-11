# Keyword backend: local rank_bm25 vs OpenSearch (both corpora)

**Hypothesis:** at 30-paper scale, BM25's recall edge nearly vanished
(+0.038 at 6p -> +0.007 at 30p). Is that partly an artifact of the local
store's naive `text.lower().split()` tokenization? OpenSearch's standard
analyzer (proper tokenization, punctuation handling) may restore part of
the edge.

**Verdict: not restored.** The standard analyzer's recall edge over the
dense baseline is +0.028 at 6 papers (local: +0.038) and **+0.000 at 30
papers** (local: +0.007) — the vanishing keyword edge at scale is a
property of the corpus (dense retrieval already covers what the keyword
leg adds), not a tokenization artifact. The one real analyzer win is on
misses, not recall: at 30p the OpenSearch `+bm25` stage holds hit@5 at
1.000 where the local tokenizer dropped it to 0.979.

Local side = phase-1 latency-after reports (20260710T185344Z /
20260710T185555Z), same code and corpora. OpenSearch side = standard
analyzer, default BM25 similarity, index backfilled from the same chunks
(sha256-deduplicated): `20260710T230723Z_ablation_opensearch-base6` and
`20260710T230909Z_ablation_opensearch-scale30`. `baseline` (dense-only) is
identical on both sides by construction — verified bit-exact as the sanity
gate. Only the `+bm25` and `+rerank` stages can move: they are the stages
the keyword leg feeds.

## base6 (479 chunks)

| stage | recall@5 local | recall@5 os | mrr local | mrr os | hit@5 local | hit@5 os |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.934 | (gate: 0.934) | 0.979 | (gate: 0.979) | 1.000 | (gate: 1.000) |
| +bm25    | 0.972 | 0.962 | 0.958 | 0.964 | 1.000 | 1.000 |
| +rerank  | 0.962 | 0.969 | 0.979 | 0.990 | 1.000 | 1.000 |

## scale30 (2,194 chunks)

| stage | recall@5 local | recall@5 os | mrr local | mrr os | hit@5 local | hit@5 os |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.934 | (gate: 0.934) | 0.844 | (gate: 0.844) | 1.000 | (gate: 1.000) |
| +bm25    | 0.941 | 0.934 | 0.839 | 0.805 | 0.979 | 1.000 |
| +rerank  | 0.934 | 0.913 | 0.877 | 0.873 | 1.000 | 0.979 |

## Latency honesty

The local store is in-memory after the phase-1 caching work, so OpenSearch
was expected to pay a network hop per query — in practice the `+bm25` p50
is indistinguishable (base6: local 311.3 ms vs opensearch 309.4 ms;
scale30: 325.7 vs 308.2), because the keyword leg runs in parallel inside
the embedding-API call's ~300 ms shadow either way. The win claimed for
OpenSearch is incremental indexing (only new chunks indexed on ingest — no
rebuild), cross-process consistency (MCP-server ingests visible to the API
immediately, no mtime polling), and the removed scale ceiling — NOT query
latency.

## Notes

- Raw scores are on different scales (rank_bm25's Okapi vs Lucene BM25) —
  irrelevant to fusion, which is rank-based (RRF).
- Single run per side; treat small deltas (<10%) as noise. Every non-gate
  delta in the tables (e.g. base6 +rerank MRR 0.979 -> 0.990, scale30
  +rerank recall 0.934 -> 0.913) is one or two questions moving a rank.
- Both sides search the same 483 documents (479 corpus chunks + 4 stray
  chunks from an old upload test, present identically in the local pickle,
  the Qdrant collection, and the backfilled index).
