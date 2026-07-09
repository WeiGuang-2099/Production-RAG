from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.retrieval.hybrid_retriever import HybridRetriever, rrf_fuse


def test_rrf_fuse_basic():
    doc_a = Document(page_content="doc A")
    doc_b = Document(page_content="doc B")
    doc_c = Document(page_content="doc C")

    ranked_lists = [
        [(doc_a, 0.9), (doc_b, 0.8)],
        [(doc_b, 0.95), (doc_c, 0.7)],
    ]
    results = rrf_fuse(ranked_lists, k=60)
    assert len(results) == 3
    assert results[0][0].page_content == "doc B"


def test_rrf_fuse_single_list():
    doc_a = Document(page_content="doc A")
    results = rrf_fuse([[(doc_a, 0.9)]], k=60)
    assert len(results) == 1


def test_hybrid_retriever():
    mock_vector = MagicMock()
    mock_vector.search.return_value = [
        (Document(page_content="vec result 1"), 0.9),
        (Document(page_content="vec result 2"), 0.7),
    ]

    mock_bm25 = MagicMock()
    mock_bm25.search.return_value = [
        (Document(page_content="vec result 1"), 2.5),
        (Document(page_content="bm25 result"), 1.8),
    ]

    retriever = HybridRetriever(vector_store=mock_vector, bm25_store=mock_bm25)
    results = retriever.retrieve("test query", top_k=3)
    assert len(results) <= 3
    assert len(results) > 0


def test_retrieve_passes_sources_to_both_stores():
    vs, bm25 = MagicMock(), MagicMock()
    vs.search.return_value = []
    bm25.search.return_value = []
    HybridRetriever(vector_store=vs, bm25_store=bm25).retrieve("q", top_k=5, sources=["x"])

    assert vs.search.call_args.kwargs["sources"] == ["x"]
    assert bm25.search.call_args.kwargs["sources"] == ["x"]
