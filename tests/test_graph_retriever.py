import pytest
from unittest.mock import MagicMock
import networkx as nx
from app.retrieval.graph_retriever import GraphRetriever
from langchain_core.documents import Document


def _make_store_with_nodes(*nodes):
    """Create a mock GraphStore with real NetworkX graph containing given nodes."""
    store = MagicMock()
    graph = nx.DiGraph()
    for node in nodes:
        graph.add_node(node)
    store.graph = graph
    return store


def test_graph_retriever():
    store = _make_store_with_nodes("artificial intelligence")
    store.get_neighbors.return_value = [
        ("machine learning", "is_subset_of"),
        ("neural networks", "uses"),
    ]

    retriever = GraphRetriever(graph_store=store)
    docs = retriever.retrieve("What is artificial intelligence?")
    assert len(docs) == 2
    assert any("machine learning" in d.page_content for d in docs)


def test_graph_retriever_substring_match():
    store = _make_store_with_nodes("machine learning")
    store.get_neighbors.return_value = [("deep learning", "subset_of")]

    retriever = GraphRetriever(graph_store=store)
    docs = retriever.retrieve("Tell me about machine learning applications")
    assert len(docs) == 1
    assert "deep learning" in docs[0].page_content


def test_graph_retriever_no_match():
    store = _make_store_with_nodes("quantum computing")

    retriever = GraphRetriever(graph_store=store)
    docs = retriever.retrieve("What is machine learning?")
    assert docs == []


def test_graph_retriever_empty_graph():
    store = MagicMock()
    store.graph = nx.DiGraph()

    retriever = GraphRetriever(graph_store=store)
    docs = retriever.retrieve("anything")
    assert docs == []
