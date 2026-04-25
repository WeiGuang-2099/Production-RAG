import pytest
from unittest.mock import MagicMock
from app.reranker.reranker import RerankerService
from langchain_core.documents import Document


def test_rerank_with_cohere():
    mock_cohere = MagicMock()
    mock_result_1 = MagicMock(index=0, relevance_score=0.95)
    mock_result_2 = MagicMock(index=1, relevance_score=0.7)
    mock_cohere.rerank.return_value = MagicMock(results=[mock_result_1, mock_result_2])

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
