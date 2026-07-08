# Scale-Robustness Evaluation (6 -> 30 Papers) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how retrieval quality holds up when the corpus grows from 6 papers (479 chunks) to 30 papers (~2,000-2,800 chunks) by adding 24 adversarial distractor papers, using the existing 48-question dataset and ablation harness unchanged.

**Architecture:** The 24 distractors are real arXiv papers deliberately confusable with the 6 ground-truth papers (RoBERTa/ALBERT vs BERT, DPR/FiD/REALM vs RAG, adapters/QLoRA vs LoRA, ...). The dataset's `source_papers` still reference only the 6 core slugs, so every retrieved distractor chunk is a scored miss — recall@k / MRR / hit@k measure robustness directly, with zero changes to `evaluation/metrics.py`. The scale corpus lives in its own `DATA_DIR` and Qdrant collection so the original 6-paper corpus stays intact and both ablations remain reproducible side by side.

**Tech Stack:** Existing harness (`run_ablation.py`, `metrics.py`, `ingest_corpus.py`, `download_papers.py`), Qdrant via docker-compose, OpenAI embeddings, Cohere rerank, LLM graph extractor.

## Global Constraints

- Run all Python via the venv interpreter: `./.venv/Scripts/python.exe` (system Python lacks deps). Commands are written for Git Bash; in PowerShell set env vars as `$env:NAME = "value"` before the command instead of prefixing.
- Commit messages: plain text, no emojis, **no AI attribution / co-author trailers**. Do not `git push` (the user pushes manually).
- The 6-paper corpus and default collection must remain untouched. All scale work uses `DATA_DIR=./data_scale` and `COLLECTION_NAME=rag_docs_scale30` (defaults are `./data` / `rag_docs`, `app/config.py:37,68`). Process env overrides `.env` values under pydantic-settings, so exporting these is sufficient.
- Qdrant must be running for ingest and ablation runs: `docker-compose up -d qdrant`.
- Real API keys from `.env` are spent. Cost checkpoints are marked in Task 2 (graph extraction, the only expensive step) and Task 7 (optional RAGAS). Ablation runs themselves cost only embeddings + rerank (cents).
- TDD does not apply to the harness scripts (they are run-once measurement tools with no unit-test suite); every change is instead verified by executing the script and checking its output, as specified per step. No files under `app/` change in this plan.
- After all code edits: `./.venv/Scripts/python.exe -m ruff check .` must pass and `./.venv/Scripts/python.exe -m pytest -q` must still report 219 passed.

---

### Task 1: Distractor manifest + `--with-distractors` download flag

**Files:**
- Modify: `evaluation/corpus/download_papers.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `DISTRACTORS: list[dict[str, str]]` (same `{slug, title, arxiv_id}` shape as `PAPERS`) importable by Task 2; CLI flag `--with-distractors`.

- [ ] **Step 1: Add `argparse` import, `DISTRACTORS` list, and the flag**

In `evaluation/corpus/download_papers.py`, add `import argparse` to the imports (after `from __future__ import annotations`, alongside `sys`). Then insert the following immediately after the existing `PAPERS` list:

```python
# 24 distractor papers for the scale-robustness corpus. Each cluster is
# deliberately confusable with one of the 6 ground-truth papers, so retrieving
# a distractor is a plausible mistake, not random noise. None of these slugs
# appear in eval_dataset.json's source_papers, so every distractor hit is a
# scored miss. If a download 404s, verify the ID at arxiv.org/abs/<id>.
DISTRACTORS: list[dict[str, str]] = [
    # Transformer/BERT-adjacent (vs attention, bert)
    {"slug": "roberta", "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)", "arxiv_id": "1907.11692"},
    {"slug": "xlnet", "title": "XLNet: Generalized Autoregressive Pretraining (Yang et al., 2019)", "arxiv_id": "1906.08237"},
    {"slug": "albert", "title": "ALBERT: A Lite BERT for Self-supervised Learning (Lan et al., 2019)", "arxiv_id": "1909.11942"},
    {"slug": "electra", "title": "ELECTRA: Pre-training Text Encoders as Discriminators (Clark et al., 2020)", "arxiv_id": "2003.10555"},
    {"slug": "transformer_xl", "title": "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (Dai et al., 2019)", "arxiv_id": "1901.02860"},
    {"slug": "t5", "title": "Exploring the Limits of Transfer Learning with T5 (Raffel et al., 2019)", "arxiv_id": "1910.10683"},
    # Scaling/GPT-adjacent (vs gpt3)
    {"slug": "scaling_laws", "title": "Scaling Laws for Neural Language Models (Kaplan et al., 2020)", "arxiv_id": "2001.08361"},
    {"slug": "palm", "title": "PaLM: Scaling Language Modeling with Pathways (Chowdhery et al., 2022)", "arxiv_id": "2204.02311"},
    {"slug": "chinchilla", "title": "Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)", "arxiv_id": "2203.15556"},
    {"slug": "llama", "title": "LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)", "arxiv_id": "2302.13971"},
    {"slug": "instructgpt", "title": "Training Language Models to Follow Instructions (Ouyang et al., 2022)", "arxiv_id": "2203.02155"},
    # Retrieval-adjacent (vs rag)
    {"slug": "dpr", "title": "Dense Passage Retrieval for Open-Domain QA (Karpukhin et al., 2020)", "arxiv_id": "2004.04906"},
    {"slug": "realm", "title": "REALM: Retrieval-Augmented Language Model Pre-Training (Guu et al., 2020)", "arxiv_id": "2002.08909"},
    {"slug": "fid", "title": "Leveraging Passage Retrieval with Generative Models (Fusion-in-Decoder; Izacard & Grave, 2020)", "arxiv_id": "2007.01282"},
    {"slug": "colbert", "title": "ColBERT: Efficient Passage Search via Late Interaction (Khattab & Zaharia, 2020)", "arxiv_id": "2004.12832"},
    {"slug": "atlas", "title": "Atlas: Few-shot Learning with Retrieval Augmented Language Models (Izacard et al., 2022)", "arxiv_id": "2208.03299"},
    {"slug": "contriever", "title": "Unsupervised Dense Information Retrieval with Contrastive Learning (Izacard et al., 2021)", "arxiv_id": "2112.09118"},
    # Parameter-efficient fine-tuning (vs lora)
    {"slug": "adapters", "title": "Parameter-Efficient Transfer Learning for NLP (Houlsby et al., 2019)", "arxiv_id": "1902.00751"},
    {"slug": "prefix_tuning", "title": "Prefix-Tuning: Optimizing Continuous Prompts for Generation (Li & Liang, 2021)", "arxiv_id": "2101.00190"},
    {"slug": "prompt_tuning", "title": "The Power of Scale for Parameter-Efficient Prompt Tuning (Lester et al., 2021)", "arxiv_id": "2104.08691"},
    {"slug": "qlora", "title": "QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)", "arxiv_id": "2305.14314"},
    # Reasoning/prompting (vs cot)
    {"slug": "self_consistency", "title": "Self-Consistency Improves Chain of Thought Reasoning (Wang et al., 2022)", "arxiv_id": "2203.11171"},
    {"slug": "zero_shot_cot", "title": "Large Language Models are Zero-Shot Reasoners (Kojima et al., 2022)", "arxiv_id": "2205.11916"},
    {"slug": "react", "title": "ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)", "arxiv_id": "2210.03629"},
]
```

Then add a `parse_args` function above `main()` and use it. Replace:

```python
def main() -> int:
    data_dir = resolve_data_dir()
    target_dir = data_dir / "papers"
    target_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for paper in PAPERS:
```

with:

```python
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download the evaluation corpus papers")
    p.add_argument(
        "--with-distractors",
        action="store_true",
        help="Also download the 24 distractor papers (scale-robustness corpus)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    papers = PAPERS + (DISTRACTORS if args.with_distractors else [])
    data_dir = resolve_data_dir()
    target_dir = data_dir / "papers"
    target_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for paper in papers:
```

- [ ] **Step 2: Gitignore the scale corpus directory**

In `.gitignore`, immediately after the existing `data/` line (line 15), add:

```
data_scale/
```

- [ ] **Step 3: Verify default behavior is unchanged (no flag, no re-download)**

Run: `./.venv/Scripts/python.exe evaluation/corpus/download_papers.py`
Expected: six `[skip] <slug>.pdf already exists` lines (the 6-paper corpus in `./data/papers` is untouched), exit 0.

- [ ] **Step 4: Download the 30-paper scale corpus (~3-5 min, network)**

Run in background: `DATA_DIR=./data_scale ./.venv/Scripts/python.exe evaluation/corpus/download_papers.py --with-distractors`
Expected: `[get ]` / `[ ok ]` lines for all 30 papers (the 6 core papers are re-downloaded into `data_scale/papers/` — intentional, the scale corpus is self-contained), final line `Done. Papers directory: ...data_scale\papers`, exit 0.
If any `[fail]` line appears: open `https://arxiv.org/abs/<arxiv_id>` for that entry, correct the ID in `DISTRACTORS`, and re-run (existing files are skipped).

- [ ] **Step 5: Verify the file count**

Run: `ls data_scale/papers/*.pdf | wc -l`
Expected: `30`

- [ ] **Step 6: Lint and commit**

Run: `./.venv/Scripts/python.exe -m ruff check evaluation/`
Expected: `All checks passed!`

```bash
git add evaluation/corpus/download_papers.py .gitignore
git commit -m "eval: add 24 distractor papers behind --with-distractors for the scale corpus"
```

---

### Task 2: DATA_DIR-aware ingest with `--with-distractors`

**Files:**
- Modify: `evaluation/ingest_corpus.py`

**Interfaces:**
- Consumes: `PAPERS`, `DISTRACTORS` from `evaluation.corpus.download_papers` (Task 1).
- Produces: a populated scale corpus (Qdrant collection `rag_docs_scale30`, BM25 pickle, graph, `ingestions.json` under `./data_scale`) for Tasks 5 and 7.

- [ ] **Step 1: Derive slugs from the download manifest and honor DATA_DIR**

In `evaluation/ingest_corpus.py`, replace:

```python
# Order mirrors evaluation/corpus/download_papers.py.
SLUGS = ["attention", "bert", "gpt3", "rag", "lora", "cot"]
```

with (the import must stay below the existing `sys.path.insert` line, hence the noqa):

```python
# Slugs come from the download manifest so the two scripts cannot drift.
from evaluation.corpus.download_papers import DISTRACTORS, PAPERS  # noqa: E402

CORE_SLUGS = [p["slug"] for p in PAPERS]
DISTRACTOR_SLUGS = [p["slug"] for p in DISTRACTORS]
```

Replace:

```python
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest the 6-paper eval corpus in-process")
    p.add_argument("--force", action="store_true", help="Re-ingest even if already tracked")
    return p.parse_args()
```

with:

```python
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest the eval corpus in-process")
    p.add_argument("--force", action="store_true", help="Re-ingest even if already tracked")
    p.add_argument(
        "--with-distractors",
        action="store_true",
        help="Also ingest the 24 distractor papers (scale-robustness corpus)",
    )
    return p.parse_args()
```

Replace:

```python
    settings = get_settings()
    print(
        f"Ingesting {len(SLUGS)} papers "
        f"(graph_extractor={settings.GRAPH_EXTRACTOR}, chunk_size={settings.CHUNK_SIZE}, "
        f"force={args.force})"
    )
```

with:

```python
    settings = get_settings()
    slugs = CORE_SLUGS + (DISTRACTOR_SLUGS if args.with_distractors else [])
    print(
        f"Ingesting {len(slugs)} papers into DATA_DIR={settings.DATA_DIR} "
        f"(graph_extractor={settings.GRAPH_EXTRACTOR}, chunk_size={settings.CHUNK_SIZE}, "
        f"force={args.force})"
    )
```

Replace:

```python
    for slug in SLUGS:
        # The source string must match how the retrieval metrics derive a paper
        # slug (basename stem of metadata["source"]): ./data/papers/<slug>.pdf.
        source = f"./data/papers/{slug}.pdf"
```

with:

```python
    for slug in slugs:
        # The source string must match how the retrieval metrics derive a paper
        # slug (basename stem of metadata["source"]): <DATA_DIR>/papers/<slug>.pdf.
        source = f"{settings.DATA_DIR}/papers/{slug}.pdf"
```

If any other reference to `SLUGS` remains in the file (check the summary print near the end), rename it to `slugs`.

- [ ] **Step 2: Smoke-test against the default corpus (free — everything is tracked)**

Run: `./.venv/Scripts/python.exe evaluation/ingest_corpus.py`
Expected: `Ingesting 6 papers into DATA_DIR=./data ...` then six `[skipped ]` lines (tracking hashes are unchanged because the default source strings are byte-identical to before), exit 0.

- [ ] **Step 3: COST CHECKPOINT — choose the graph extractor for the scale ingest**

The LLM graph extractor makes one completion call per chunk (`app/graph/builder.py:39-43`), and the ~24 new papers add roughly 1,500-2,300 chunks. Two options — **ask the user which to use before running**:

- **Option A (methodology-consistent, recommended):** keep the `.env` strong model (gpt-4o, matching how the 6-paper graph was built). Estimated one-time cost $6-10, sequential wall time 1-3 hours.
- **Option B (cheap):** prefix the ingest command with `LLM_MODEL=gpt-4o-mini` to extract with the fast model. Estimated cost under $1, same wall time; the results write-up (Task 6) must then note the extractor model differs from the 6-paper corpus.

- [ ] **Step 4: Ingest the scale corpus (long-running — run in background)**

Ensure Qdrant is up: `docker-compose up -d qdrant`

Run in background (Option A shown; for Option B prepend `LLM_MODEL=gpt-4o-mini `):

```bash
DATA_DIR=./data_scale COLLECTION_NAME=rag_docs_scale30 GRAPH_EXTRACTOR=llm \
  ./.venv/Scripts/python.exe evaluation/ingest_corpus.py --with-distractors
```

Expected: `Ingesting 30 papers into DATA_DIR=./data_scale ...`, thirty `[ingested]` lines with per-paper chunk counts, total chunks roughly 2,000-2,800, exit 0. Individual paper failures are printed as `[FAILED  ]` — if any appear, re-run the same command (tracking skips completed papers) after diagnosing.

- [ ] **Step 5: Verify the scale corpus artifacts**

Run: `ls data_scale/` and `curl -s http://localhost:6333/collections/rag_docs_scale30 | head -c 300`
Expected: `data_scale/` contains `papers/`, `bm25_index.pkl`, `ingestions.json`, and the graph store file; the curl returns collection info with a non-zero `points_count` (should be in the 2,000-2,800 range).
Also verify the original corpus is untouched: `curl -s http://localhost:6333/collections/rag_docs | head -c 300` still reports ~479 points.

- [ ] **Step 6: Lint and commit**

Run: `./.venv/Scripts/python.exe -m ruff check evaluation/`
Expected: `All checks passed!`

```bash
git add evaluation/ingest_corpus.py
git commit -m "eval: derive corpus slugs from the download manifest, honor DATA_DIR, add --with-distractors"
```

---

### Task 3: `--label` suffix for ablation reports

**Files:**
- Modify: `evaluation/run_ablation.py`

**Interfaces:**
- Produces: CLI flag `--label <name>` that turns saved filenames into `{ts}_ablation_{name}.{md,json}`; used by Tasks 4 and 5.

- [ ] **Step 1: Add the flag and thread it into the save path**

In `evaluation/run_ablation.py` `parse_args()`, after the `--no-save` line, add:

```python
    p.add_argument("--label", default=None, help="Filename suffix for saved reports (e.g. base6, scale30)")
```

In `main()`, replace the save block:

```python
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = Path(__file__).resolve().parent / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{ts}_ablation.md").write_text(table + "\n", encoding="utf-8")
        (out_dir / f"{ts}_ablation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved: evaluation/results/{ts}_ablation.md")
```

with:

```python
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = f"_{args.label}" if args.label else ""
        out_dir = Path(__file__).resolve().parent / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{ts}_ablation{suffix}.md").write_text(table + "\n", encoding="utf-8")
        (out_dir / f"{ts}_ablation{suffix}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved: evaluation/results/{ts}_ablation{suffix}.md")
```

- [ ] **Step 2: Verify with a 2-question smoke run (needs Qdrant + API keys; costs cents)**

Run: `./.venv/Scripts/python.exe evaluation/run_ablation.py --subset 2 --label smoke`
Expected: four stage tables print, final line `Saved: evaluation/results/<ts>_ablation_smoke.md`, and both `*_ablation_smoke.md` and `*_ablation_smoke.json` exist in `evaluation/results/`.

- [ ] **Step 3: Remove the smoke artifacts**

Run: `rm evaluation/results/*_ablation_smoke.md evaluation/results/*_ablation_smoke.json`

- [ ] **Step 4: Lint and commit**

Run: `./.venv/Scripts/python.exe -m ruff check evaluation/`
Expected: `All checks passed!`

```bash
git add evaluation/run_ablation.py
git commit -m "eval: add --label suffix to ablation report filenames"
```

---

### Task 4: Same-day 6-paper baseline rerun

Rerunning the 6-paper ablation on the same day/machine/network as the scale run makes the latency columns comparable (the committed 2026-06-22 numbers came from a different session).

**Files:**
- Create: `evaluation/results/<ts>_ablation_base6.md` and `.json` (written by the harness)

- [ ] **Step 1: Run the baseline ablation (~5-10 min — run in background)**

Run in background: `./.venv/Scripts/python.exe evaluation/run_ablation.py --k 5 --label base6`
Expected: all four stages complete over 48 questions; quality metrics land near the 2026-06-22 run (recall@5 roughly 0.90-0.98, hit@5 at or near 1.000 for the first three stages). Latencies may differ — that is fine and expected.

- [ ] **Step 2: Sanity-check before proceeding**

If recall@5 for `baseline` is below ~0.85 or errors appear in the output, STOP: the default corpus/collection is likely in a bad state (check `docker-compose ps`, `.env` keys, and that `rag_docs` still has ~479 points). Do not continue to Task 5 until the baseline reproduces.

- [ ] **Step 3: Commit the report**

```bash
git add evaluation/results/*_ablation_base6.md evaluation/results/*_ablation_base6.json
git commit -m "eval: same-day 6-paper ablation rerun (base6) for scale comparison"
```

---

### Task 5: 30-paper scale ablation

**Files:**
- Create: `evaluation/results/<ts>_ablation_scale30.md` and `.json` (written by the harness)

- [ ] **Step 1: Run the scale ablation (~5-15 min — run in background)**

Run in background:

```bash
DATA_DIR=./data_scale COLLECTION_NAME=rag_docs_scale30 \
  ./.venv/Scripts/python.exe evaluation/run_ablation.py --k 5 --label scale30
```

Expected: four stages over 48 questions against the 30-paper corpus. Metrics may be lower than base6 — that is the measurement, not a failure. Note whether p50 latency grows (the per-query BM25 unpickle is ~5x larger here; this number feeds the latency-optimization roadmap item).

- [ ] **Step 2: Sanity-check the run measured the right corpus**

In the printed output (or the JSON), confirm `count` is 48 for each stage and that recall differs from base6. If every number is identical to base6, the env overrides did not take — verify with `DATA_DIR=./data_scale COLLECTION_NAME=rag_docs_scale30 ./.venv/Scripts/python.exe -c "from app.config import get_settings; s=get_settings(); print(s.DATA_DIR, s.COLLECTION_NAME)"` printing `./data_scale rag_docs_scale30`.

- [ ] **Step 3: Commit the report**

```bash
git add evaluation/results/*_ablation_scale30.md evaluation/results/*_ablation_scale30.json
git commit -m "eval: 30-paper scale-robustness ablation (scale30)"
```

---

### Task 6: Results narrative in docs

**Files:**
- Modify: `evaluation/results/README.md`
- Modify: `evaluation/README.md`
- Modify: `docs/CASE_STUDY.md`
- Modify: `README.md`

All numbers below are placeholders in `{braces}` — fill every one from the committed `*_ablation_base6.json` and `*_ablation_scale30.json` before committing. Do not commit a placeholder.

- [ ] **Step 1: Add the scale section to `evaluation/results/README.md`**

Append after the existing ablation section:

```markdown
## Scale robustness: 6 papers vs 30 papers

Same 48 questions, same harness, two corpora. The 30-paper corpus adds 24
*distractor* papers chosen to be confusable with the 6 ground-truth papers
(RoBERTa/ALBERT/ELECTRA vs BERT, DPR/FiD/REALM/ColBERT vs RAG, adapters/
prefix-tuning/QLoRA vs LoRA, self-consistency/ReAct vs CoT, ...), so every
distractor retrieved is a scored miss. The dataset and metrics are unchanged.

Both runs {date}, same machine and keys, `top_k=10`, recall@5 over the
reranked top-5. 6-paper corpus: 479 chunks (`rag_docs`); 30-paper corpus:
{chunks} chunks (`rag_docs_scale30`), graph built by the {model} extractor.

| stage | recall@5 6p | recall@5 30p | mrr 6p | mrr 30p | hit@5 30p | p50_ms 6p | p50_ms 30p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (dense)   | {n} | {n} | {n} | {n} | {n} | {n} | {n} |
| +bm25 (hybrid RRF) | {n} | {n} | {n} | {n} | {n} | {n} | {n} |
| +rerank (Cohere)   | {n} | {n} | {n} | {n} | {n} | {n} | {n} |
| +graph             | {n} | {n} | {n} | {n} | {n} | {n} | {n} |

Honest read ({rewrite from the actual numbers — cover each point}):

- How much does dense-only degrade once confusable neighbors exist, and on
  which question types?
- Does the BM25 / rerank margin widen at scale — i.e. does the hybrid stack
  earn more of its keep than it could on 6 well-separated papers?
- Does +graph still hurt, or does cross-paper expansion start paying off with
  real related work in the corpus?
- Latency: p50 grows from {n} to {n} ms; the BM25 index is unpickled and a
  fresh Qdrant session opened on every query, which is the target of the
  planned latency work.
```

- [ ] **Step 2: Document the scale corpus in `evaluation/README.md`**

Add a short subsection to the corpus/setup part of `evaluation/README.md`:

```markdown
### Scale-robustness corpus (30 papers)

The ablation can also run against a 5x corpus that adds 24 distractor papers
(see `DISTRACTORS` in `corpus/download_papers.py`). It lives in its own data
dir and Qdrant collection so the 6-paper corpus stays intact:

    DATA_DIR=./data_scale python evaluation/corpus/download_papers.py --with-distractors
    DATA_DIR=./data_scale COLLECTION_NAME=rag_docs_scale30 GRAPH_EXTRACTOR=llm \
        python evaluation/ingest_corpus.py --with-distractors
    DATA_DIR=./data_scale COLLECTION_NAME=rag_docs_scale30 \
        python evaluation/run_ablation.py --k 5 --label scale30
```

- [ ] **Step 3: Add Finding 4 to `docs/CASE_STUDY.md`**

Append after Finding 3, following the existing findings' voice (numbers first, interpretation second, no adjectives without evidence):

```markdown
## Finding 4 — scale: {one-line takeaway from the real numbers}

The 6-paper ablation had a known weakness: six topically distinct papers
separate so cleanly in embedding space that dense retrieval was already at
hit@5 = 1.000, leaving the hybrid stack little to prove. So I grew the corpus
5x with 24 *adversarial* distractors — papers a retriever plausibly confuses
with the ground truth (RoBERTa vs BERT, DPR/FiD vs RAG, QLoRA vs LoRA...) —
and reran the identical 48 questions.

| stage | recall@5 (6p) | recall@5 (30p) | mrr (6p) | mrr (30p) |
| --- | --- | --- | --- | --- |
| baseline (dense)   | {n} | {n} | {n} | {n} |
| +bm25 (hybrid RRF) | {n} | {n} | {n} | {n} |
| +rerank (Cohere)   | {n} | {n} | {n} | {n} |
| +graph             | {n} | {n} | {n} | {n} |

{2-3 paragraphs of interpretation written from the real numbers: what
degraded, what recovered it, whether graph's verdict changed, and what this
says about when each component is worth its cost.}
```

- [ ] **Step 4: Update the main `README.md` evaluation section**

After the existing ablation table's "Honest read" paragraph, add one short paragraph:

```markdown
A scale-robustness run repeats the identical ablation on a 30-paper corpus
(24 adversarial distractors added, same 48 questions): {one-sentence result,
e.g. how much dense degraded and which component recovered it}. Full tables
in [`evaluation/results/`](evaluation/results/README.md).
```

- [ ] **Step 5: Full verification and commit**

Run: `./.venv/Scripts/python.exe -m ruff check .` — expected `All checks passed!`
Run: `./.venv/Scripts/python.exe -m pytest -q` — expected `219 passed`.
Re-read the four edited docs and confirm no `{placeholder}` remains.

```bash
git add evaluation/results/README.md evaluation/README.md docs/CASE_STUDY.md README.md
git commit -m "docs: scale-robustness findings (6 vs 30 papers) in results, case study, README"
```

---

### Task 7 (OPTIONAL): RAGAS refusal robustness at scale

Skippable. COST CHECKPOINT: ~48 questions with gpt-4o answers + gpt-4o-mini judge, roughly $3-6 — ask the user before running. The interesting question: with 5x more plausible-looking context available, does the grounded prompt still refuse all 5 unanswerables, and does answerable multi-hop hold up?

- [ ] **Step 1: Run RAGAS grounded against the scale corpus (background, ~15-30 min)**

```bash
DATA_DIR=./data_scale COLLECTION_NAME=rag_docs_scale30 PROMPT_MODE=grounded \
  ./.venv/Scripts/python.exe evaluation/run_eval.py --label grounded_scale30
```

Expected: a `*_grounded_scale30.json` report in `evaluation/results/`.

- [ ] **Step 2: Extract the two headline numbers**

From the report (and the per-question answers in it): refusal count on the 5 `unanswerable` questions (compare to 5/5 at 6 papers) and refusal count on the 10 answerable `multi_hop` questions (compare to the over-refusal bug history at `RERANK_TOP_K=3`).

- [ ] **Step 3: Add one paragraph to `evaluation/results/README.md` scale section and commit**

State the two numbers and their reading in the established style, then:

```bash
git add evaluation/results/README.md evaluation/results/*_grounded_scale30.json
git commit -m "eval: RAGAS grounded run on the 30-paper corpus (refusal robustness)"
```

---

## Execution notes

- Task order is strict for 1 -> 2 -> (3) -> 4 -> 5 -> 6; Task 3 only needs Task 1's commit-cleanliness, not its corpus. Task 7 is optional and last.
- Long-running commands (Task 2 Step 4, Task 4 Step 1, Task 5 Step 1, Task 7 Step 1) should use background execution; foreground tool timeouts will kill them.
- Nothing in this plan touches `app/` or `frontend/`, so it can execute in parallel with the chat-scope plan (`2026-06-23-chat-scope-and-history.md`) without conflicts.
