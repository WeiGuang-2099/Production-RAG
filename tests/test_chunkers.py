from langchain_core.documents import Document

from app.ingestion.chunkers import chunk_documents


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


def test_chunks_are_token_sized_not_char_sized():
    """CHUNK_SIZE is documented as tokens. A token-aware splitter fills close
    to the 50-token budget; a char-based one (50 chars) caps every chunk near
    ~12 tokens, so the max chunk size distinguishes the two."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    docs = [Document(page_content="word " * 300, metadata={"source": "t"})]
    chunks = chunk_documents(docs, chunk_size=50, chunk_overlap=10)
    token_counts = [len(enc.encode(c.page_content)) for c in chunks]

    assert len(chunks) > 1
    assert max(token_counts) > 40    # fails for char-based splitting
    assert max(token_counts) <= 50   # never exceeds the token budget
