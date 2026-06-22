"""Ingest the 6-paper evaluation corpus directly into the stores.

Unlike the curl-the-API loop in ``evaluation/README.md``, this drives the
ingest pipeline in-process, so it needs only Qdrant running (not the HTTP API).
It honors the same ``.env`` config (``GRAPH_EXTRACTOR``, ``CHUNK_SIZE``, ...),
so the corpus it builds matches how the service would ingest it.

Usage:
    python evaluation/ingest_corpus.py            # ingest all 6 (skip tracked)
    python evaluation/ingest_corpus.py --force    # re-ingest from scratch
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Order mirrors evaluation/corpus/download_papers.py.
SLUGS = ["attention", "bert", "gpt3", "rag", "lora", "cot"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest the 6-paper eval corpus in-process")
    p.add_argument("--force", action="store_true", help="Re-ingest even if already tracked")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from app.config import get_settings
    from app.core.pipeline import ingest_pipeline

    settings = get_settings()
    print(
        f"Ingesting {len(SLUGS)} papers "
        f"(graph_extractor={settings.GRAPH_EXTRACTOR}, chunk_size={settings.CHUNK_SIZE}, "
        f"force={args.force})"
    )

    total_chunks = 0
    failures: list[str] = []
    t_all = time.time()
    for slug in SLUGS:
        # The source string must match how the retrieval metrics derive a paper
        # slug (basename stem of metadata["source"]): ./data/papers/<slug>.pdf.
        source = f"./data/papers/{slug}.pdf"
        t0 = time.time()
        try:
            result = ingest_pipeline(source, force=args.force)
            dt = time.time() - t0
            total_chunks += result.get("chunks", 0)
            print(f"[{result['status']:8s}] {slug:10s} chunks={result.get('chunks', 0):4d}  {dt:7.1f}s")
        except Exception as exc:  # noqa: BLE001
            print(f"[FAILED  ] {slug:10s} {exc}")
            failures.append(slug)

    print(
        f"\nDone in {time.time() - t_all:.1f}s. "
        f"total_chunks={total_chunks} failures={failures or 'none'}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
