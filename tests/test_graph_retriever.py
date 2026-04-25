import pytest
from unittest.mock import MagicMock
from app.retrieval.graph_retriever import GraphRetriever
from langchain_core.documents import Document


def test_graph_retriever():
    mock_graph_store = MagicMock()
    mock_graph_store.get_neighbors.return_value = [
        ("machine learning", "is_subset_of"),
        ("neural networks", "uses"),
    ]

    retriever = GraphRetriever(graph_store=mock_graph_store)
    docs = retriever.retrieve("artificial intelligence")
    assert len(docs) > 0
    assert any("machine learning" in d.page_content for d in docs)


def test_graph_retriever_no_neighbors():
    mock_graph_store = MagicMock()
    mock_graph_store.get_neighbors.return_value = []

    retriever = GraphRetriever(graph_store=mock_graph_store)
    docs = retriever.retrieve("unknown entity")
    assert docs == []
