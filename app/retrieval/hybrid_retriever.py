from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.documents import Document

if TYPE_CHECKING:
    from app.retrieval.bm25_store import BM25Store
    from app.retrieval.vector_store import VectorStore


def rrf_fuse(
    ranked_lists: list[list[tuple[Document, float]]],
    k: int = 60,
) -> list[tuple[Document, float]]:
    scores: dict[str, tuple[Document, float]] = {}

    for ranked in ranked_lists:
        for rank, (doc, _original_score) in enumerate(ranked):
            content = doc.page_content
            rrf_score = 1.0 / (k + rank + 1)
            if content in scores:
                existing_doc, existing_score = scores[content]
                scores[content] = (existing_doc, existing_score + rrf_score)
            else:
                scores[content] = (doc, rrf_score)

    sorted_results = sorted(scores.values(), key=lambda x: x[1], reverse=True)
    return sorted_results


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25_store: BM25Store) -> None:
        self.vector_store = vector_store
        self.bm25_store = bm25_store

    def retrieve(
        self, query: str, top_k: int = 5, sources: list[str] | None = None
    ) -> list[tuple[Document, float]]:
        vector_results = self.vector_store.search(query, top_k=top_k, sources=sources)
        bm25_results = self.bm25_store.search(query, top_k=top_k, sources=sources)
        fused = rrf_fuse([vector_results, bm25_results])
        return fused[:top_k]
