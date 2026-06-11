"""Download the 6 classic ML/AI papers used as the evaluation corpus.

Files are written to <DATA_DIR>/papers/ so they can be ingested through the
existing /ingest API (which requires source files to live under DATA_DIR).

Usage:
    python evaluation/corpus/download_papers.py
"""
from __future__ import annotations

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


def main() -> int:
    data_dir = resolve_data_dir()
    target_dir = data_dir / "papers"
    target_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for paper in PAPERS:
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
