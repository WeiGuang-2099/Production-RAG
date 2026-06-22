from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.reranker.reranker import RerankerService, _is_rate_limit


def test_is_rate_limit_classification():
    assert _is_rate_limit(Exception("status_code: 429, body: trial key"))
    assert _is_rate_limit(Exception("Too Many Requests"))
    assert _is_rate_limit(type("E", (Exception,), {"status_code": 429})())
    assert not _is_rate_limit(ValueError("invalid request"))


def test_rerank_retries_on_rate_limit():
    """A Cohere 429 (e.g. trial-key limit) is retried, not dropped on first hit."""
    mock_cohere = MagicMock()
    mock_cohere.rerank.side_effect = [
        Exception("status_code: 429 too many requests"),
        [{"index": 0, "relevance_score": 0.9}],
    ]
    service = RerankerService(reranker=mock_cohere)
    with patch("tenacity.nap.time.sleep"):  # don't actually wait out the backoff
        results = service.rerank("q", [Document(page_content="d")], top_k=1)
    assert mock_cohere.rerank.call_count == 2
    assert results[0].metadata["relevance_score"] == 0.9


def test_rerank_does_not_retry_non_rate_limit():
    """Non-rate-limit errors surface immediately (no wasted retries)."""
    mock_cohere = MagicMock()
    mock_cohere.rerank.side_effect = ValueError("invalid request")
    service = RerankerService(reranker=mock_cohere)
    with patch("tenacity.nap.time.sleep"):
        try:
            service.rerank("q", [Document(page_content="d")], top_k=1)
            raise AssertionError("expected ValueError to propagate")
        except ValueError:
            pass
    assert mock_cohere.rerank.call_count == 1


def test_rerank_with_cohere():
    mock_cohere = MagicMock()
    mock_cohere.rerank.return_value = [
        {"index": 0, "relevance_score": 0.95},
        {"index": 1, "relevance_score": 0.7},
    ]

    service = RerankerService(reranker=mock_cohere)
    docs = [
        Document(page_content="relevant doc"),
        Document(page_content="less relevant doc"),
    ]
    results = service.rerank("test query", docs, top_k=2)
    assert len(results) == 2
    assert results[0].metadata["relevance_score"] == 0.95


def test_rerank_no_reranker():
    service = RerankerService(reranker=None)
    docs = [Document(page_content="doc1"), Document(page_content="doc2")]
    results = service.rerank("test query", docs, top_k=2)
    assert len(results) == 2
    assert results[0].page_content == "doc1"


def test_rerank_fewer_than_top_k():
    service = RerankerService(reranker=None)
    docs = [Document(page_content="doc1")]
    results = service.rerank("test query", docs, top_k=5)
    assert len(results) == 1


def test_rerank_immutability():
    """Verify that reranking does not mutate original documents."""
    mock_cohere = MagicMock()
    mock_cohere.rerank.return_value = [
        {"index": 0, "relevance_score": 0.95},
    ]
    service = RerankerService(reranker=mock_cohere)
    original_doc = Document(page_content="test", metadata={"source": "file.pdf"})
    results = service.rerank("query", [original_doc], top_k=1)
    # Original should NOT have relevance_score
    assert "relevance_score" not in original_doc.metadata
    # Result should have it
    assert results[0].metadata["relevance_score"] == 0.95
    assert results[0].metadata["source"] == "file.pdf"
