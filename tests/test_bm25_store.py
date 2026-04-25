import pytest
from app.retrieval.bm25_store import BM25Store
from langchain_core.documents import Document


def test_add_and_search(tmp_path):
    store = BM25Store(data_dir=str(tmp_path))
    docs = [
        Document(page_content="Machine learning is a subset of artificial intelligence.", metadata={"source": "a"}),
        Document(page_content="Python is a popular programming language.", metadata={"source": "b"}),
        Document(page_content="Neural networks are used in deep learning.", metadata={"source": "c"}),
    ]
    store.add_documents(docs)

    results = store.search("machine learning", top_k=2)
    assert len(results) >= 1
    assert results[0][0].page_content == docs[0].page_content
    assert results[0][1] > 0


def test_search_empty(tmp_path):
    store = BM25Store(data_dir=str(tmp_path))
    results = store.search("test", top_k=5)
    assert results == []


def test_save_and_load(tmp_path):
    store1 = BM25Store(data_dir=str(tmp_path))
    docs = [
        Document(page_content="Machine learning is a subset of artificial intelligence.", metadata={"source": "a"}),
        Document(page_content="Python is a popular programming language.", metadata={"source": "b"}),
        Document(page_content="Neural networks are used in deep learning.", metadata={"source": "c"}),
    ]
    store1.add_documents(docs)

    store2 = BM25Store(data_dir=str(tmp_path))
    results = store2.search("machine learning", top_k=2)
    assert len(results) >= 1
    assert results[0][0].page_content == docs[0].page_content
