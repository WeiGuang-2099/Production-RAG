"""Backfill the OpenSearch keyword index from the local BM25 pickle.

Zero-cost migration: reads the full list[Document] that the local store
happens to persist in <DATA_DIR>/bm25_index.pkl and bulk-indexes it into
OpenSearch — no re-embedding, no re-parsing. A production migration would
re-drive from the source documents instead of relying on this accident of
the local store's persistence format.

Usage:
    python evaluation/backfill_keyword_index.py                # settings defaults (base6)
    DATA_DIR=./data_scale OPENSEARCH_INDEX=rag_chunks_scale30 \
        python evaluation/backfill_keyword_index.py            # scale corpus
    python evaluation/backfill_keyword_index.py --recreate     # drop + remap the index first
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.retrieval.opensearch_store import OpenSearchStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    s = get_settings()
    p = argparse.ArgumentParser(description="Backfill OpenSearch from the local BM25 pickle")
    p.add_argument("--data-dir", default=s.DATA_DIR, help="corpus dir holding bm25_index.pkl")
    p.add_argument("--url", default=s.OPENSEARCH_URL)
    p.add_argument("--index", default=s.OPENSEARCH_INDEX)
    p.add_argument("--recreate", action="store_true", help="delete and remap the index first")
    p.add_argument("--batch-size", type=int, default=500)
    return p.parse_args()


def backfill(data_dir: str, url: str, index: str, recreate: bool, batch_size: int) -> int:
    pkl = Path(data_dir) / "bm25_index.pkl"
    if not pkl.exists():
        print(f"error: {pkl} not found — ingest the corpus locally first", file=sys.stderr)
        return 1
    with open(pkl, "rb") as f:
        documents = pickle.load(f)["documents"]

    store = OpenSearchStore(url=url, index_name=index)
    if recreate and store._client.indices.exists(index=index):
        store._client.indices.delete(index=index)
    for i in range(0, len(documents), batch_size):
        store.add_documents(documents[i : i + batch_size])

    print(f"indexed {len(documents)} documents into {index}")
    return 0


def main() -> int:
    args = parse_args()
    return backfill(args.data_dir, args.url, args.index, args.recreate, args.batch_size)


if __name__ == "__main__":
    sys.exit(main())
