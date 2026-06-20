from unittest.mock import MagicMock

from app.ingestion.embedder import Embedder


def test_embed_documents():
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    embedder = Embedder(mock_embeddings)
    texts = ["hello", "world"]
    vectors = embedder.embed_documents(texts)
    assert len(vectors) == 2
    mock_embeddings.embed_documents.assert_called_once_with(texts)


def test_embed_query():
    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

    embedder = Embedder(mock_embeddings)
    vector = embedder.embed_query("test query")
    assert len(vector) == 3
    mock_embeddings.embed_query.assert_called_once_with("test query")
