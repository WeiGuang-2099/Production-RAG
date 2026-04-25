from app.ingestion.chunkers import chunk_documents
from langchain_core.documents import Document


def test_chunk_documents():
    docs = [Document(page_content="Word " * 200, metadata={"source": "test.txt"})]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all("source" in c.metadata for c in chunks)
    assert all(c.metadata["chunk_index"] is not None for c in chunks)


def test_chunk_documents_empty():
    chunks = chunk_documents([])
    assert chunks == []


def test_chunk_preserves_metadata():
    docs = [Document(page_content="Short text.", metadata={"source": "a.txt", "custom": "val"})]
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].metadata["source"] == "a.txt"
    assert chunks[0].metadata["custom"] == "val"
