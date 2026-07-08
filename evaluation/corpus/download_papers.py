"""Download the 6 classic ML/AI papers used as the evaluation corpus.

Files are written to <DATA_DIR>/papers/ so they can be ingested through the
existing /ingest API (which requires source files to live under DATA_DIR).

Usage:
    python evaluation/corpus/download_papers.py
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

PAPERS: list[dict[str, str]] = [
    {
        "slug": "attention",
        "title": "Attention Is All You Need (Vaswani et al., 2017)",
        "arxiv_id": "1706.03762",
    },
    {
        "slug": "bert",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)",
        "arxiv_id": "1810.04805",
    },
    {
        "slug": "gpt3",
        "title": "Language Models are Few-Shot Learners (Brown et al., 2020)",
        "arxiv_id": "2005.14165",
    },
    {
        "slug": "rag",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP (Lewis et al., 2020)",
        "arxiv_id": "2005.11401",
    },
    {
        "slug": "lora",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)",
        "arxiv_id": "2106.09685",
    },
    {
        "slug": "cot",
        "title": "Chain-of-Thought Prompting Elicits Reasoning in LLMs (Wei et al., 2022)",
        "arxiv_id": "2201.11903",
    },
]

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

USER_AGENT = "ProductionRAG-EvalCorpus/1.0 (research; contact: project README)"


def resolve_data_dir() -> Path:
    """Resolve DATA_DIR from app.config, falling back to ./data."""
    try:
        from app.config import get_settings

        return Path(get_settings().DATA_DIR).resolve()
    except Exception:
        return Path("./data").resolve()


def download(url: str, target: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp, open(target, "wb") as f:
        while chunk := resp.read(1 << 16):
            f.write(chunk)


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
        slug = paper["slug"]
        target = target_dir / f"{slug}.pdf"
        if target.exists() and target.stat().st_size > 0:
            print(f"[skip] {slug}.pdf already exists ({target.stat().st_size / 1024:.0f} KB)")
            continue
        url = f"https://arxiv.org/pdf/{paper['arxiv_id']}"
        print(f"[get ] {slug:10s} <- {url}")
        try:
            download(url, target)
            size_kb = target.stat().st_size / 1024
            print(f"[ ok ] {slug}.pdf saved ({size_kb:.0f} KB)")
        except Exception as exc:
            print(f"[fail] {slug}: {exc}", file=sys.stderr)
            failures.append(slug)
        time.sleep(1.0)

    print()
    print(f"Done. Papers directory: {target_dir}")
    if failures:
        print(f"Failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
