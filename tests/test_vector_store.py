import pytest
from unittest.mock import MagicMock, patch
from app.retrieval.vector_store import VectorStore
from langchain_core.documents import Document


@pytest.fixture
def mock_qdrant():
    with patch("app.retrieval.vector_store.QdrantVectorStore") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.from_existing_collection.return_value = mock_instance
        mock_cls.return_value = mock_instance
        yield mock_instance, mock_cls


def test_upsert_documents(mock_qdrant):
    mock_instance, mock_cls = mock_qdrant
    mock_instance.add_documents.return_value = ["id1", "id2"]

    with patch("app.retrieval.vector_store.get_embedder") as mock_emb, \
         patch("app.retrieval.vector_store.get_settings") as mock_s:
        mock_emb.return_value = MagicMock()
        mock_s.return_value.QDRANT_URL = "http://localhost:6333"
        mock_s.return_value.COLLECTION_NAME = "test_col"
        mock_s.return_value.QDRANT_API_KEY = None

        vs = VectorStore()
        docs = [
            Document(page_content="hello", metadata={"source": "a"}),
            Document(page_content="world", metadata={"source": "b"}),
        ]
        ids = vs.upsert(docs)
        assert len(ids) == 2


def test_search(mock_qdrant):
    mock_instance, mock_cls = mock_qdrant
    mock_doc = Document(page_content="result", metadata={"score": 0.9})
    mock_instance.similarity_search_with_score.return_value = [(mock_doc, 0.9)]

    with patch("app.retrieval.vector_store.get_embedder") as mock_emb, \
         patch("app.retrieval.vector_store.get_settings") as mock_s:
        mock_emb.return_value = MagicMock()
        mock_s.return_value.QDRANT_URL = "http://localhost:6333"
        mock_s.return_value.COLLECTION_NAME = "test_col"
        mock_s.return_value.QDRANT_API_KEY = None

        vs = VectorStore()
        results = vs.search("test query", top_k=5)
        assert len(results) == 1
        assert results[0][1] == 0.9
