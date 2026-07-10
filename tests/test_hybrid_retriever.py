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


def _doc(text: str) -> Document:
    return Document(page_content=text)


def test_parallel_retrieve_matches_serial_fusion():
    d1, d2, d3 = _doc("one"), _doc("two"), _doc("three")
    vs, bm = MagicMock(), MagicMock()
    vs.search.return_value = [(d1, 0.9), (d2, 0.8)]
    bm.search.return_value = [(d2, 7.0), (d3, 5.0)]

    result = HybridRetriever(vector_store=vs, bm25_store=bm).retrieve("q", top_k=3)

    expected = rrf_fuse([vs.search.return_value, bm.search.return_value])[:3]
    assert [(d.page_content, s) for d, s in result] == [(d.page_content, s) for d, s in expected]


def test_vector_leg_failure_degrades_to_bm25_only():
    d3 = _doc("three")
    vs, bm = MagicMock(), MagicMock()
    vs.search.side_effect = RuntimeError("qdrant down")
    bm.search.return_value = [(d3, 5.0)]

    result = HybridRetriever(vector_store=vs, bm25_store=bm).retrieve("q", top_k=3)

    assert [d.page_content for d, _ in result] == ["three"]


def test_bm25_leg_failure_degrades_to_vector_only():
    d1 = _doc("one")
    vs, bm = MagicMock(), MagicMock()
    vs.search.return_value = [(d1, 0.9)]
    bm.search.side_effect = RuntimeError("index corrupt")

    result = HybridRetriever(vector_store=vs, bm25_store=bm).retrieve("q", top_k=3)

    assert [d.page_content for d, _ in result] == ["one"]
