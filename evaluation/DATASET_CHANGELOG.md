# Dataset changelog

The judge compares each answer with these reference answers, so every change to them is
listed here. Each published report embeds the items it was judged with, reference answers
included, so published reports keep the references they were judged against.

Fingerprint: the first 10 hex digits of the SHA-1 of the JSON list of sorted
`(id, ground_truth)` pairs over `eval_dataset.json` and `unanswerable_extra.json`.

## v1.1 (2026-10-10): two reference answers corrected

- **q047** (RAG's K). The old reference said "K = 5 retrieved passages (with some ablations
  exploring values up to 10)". The paper trains with 5 or 10 retrieved documents, sets k at
  test time on dev data, and reports test numbers with 15 documents (RAG-Token) and 50
  (RAG-Sequence) for open-domain QA and 10 for MS-MARCO and Jeopardy question generation
  (rag.pdf: the training setup, the discussion of the number of retrieved documents and
  Appendix A; Figure 3 varies K up to 50). The question presupposes one value; the new
  reference says there is none.
- **q048** (decoding strategy). Dropped the false parenthetical "the paper later explores
  self-consistency with sampling as a separate extension": the paper decodes greedily and only
  cites follow-up work (Wang et al., 2022a), as q069's reference already said.
- Effect on published numbers: in every run except P and P5, each q047 answer was the exact
  refusal sentence, so its refused or answered outcome does not depend on the reference; only
  the evidence label of those refusals does. Every answered q048 pass said "greedy decoding"
  and was judged correct, which the new reference also supports. P and P5 answered q047 in
  every pass and were judged against the wrong reference (partially correct 5 times, incorrect
  once); their published rates are left as they were judged.
- Fingerprints: v1.0 `8826bbe058`, v1.1 `cdf2440006`.

## v1.0 (2026-10-07)

48 hand-written questions (`eval_dataset.json`, first committed 2026-06-11) and 22 near-miss
unanswerable questions (`unanswerable_extra.json`).
