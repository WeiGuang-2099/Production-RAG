import pytest
from unittest.mock import MagicMock, patch
from app.retrieval.vector_store import VectorStore
from langchain_core.documents import Document


@pytest.fixture
def mock_qdrant():
    with patch("app.retrieval.vector_store.QdrantVectorStore") as mock_cls, \
         patch("app.retrieval.vector_store.QdrantClient") as mock_client_cls:
        mock_instance = MagicMock()
        mock_cls.from_existing_collection.return_value = mock_instance
        mock_cls.return_value = mock_instance
        mock_client_cls.return_value.collection_exists.return_value = True
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


def test_upsert_creates_missing_collection(mock_qdrant):
    """First ingest against a fresh Qdrant must create the collection, not crash."""
    mock_instance, mock_cls = mock_qdrant
    mock_instance.add_documents.return_value = ["id1"]

    with patch("app.retrieval.vector_store.get_embedder") as mock_emb, \
         patch("app.retrieval.vector_store.get_settings") as mock_s, \
         patch("app.retrieval.vector_store.QdrantClient") as mock_client_cls:
        mock_emb.return_value.embed_query.return_value = [0.0] * 8
        mock_s.return_value.QDRANT_URL = "http://localhost:6333"
        mock_s.return_value.COLLECTION_NAME = "test_col"
        mock_s.return_value.QDRANT_API_KEY = None

        mock_client = mock_client_cls.return_value
        mock_client.collection_exists.return_value = False

        vs = VectorStore()
        vs.upsert([Document(page_content="hello")])

        mock_client.create_collection.assert_called_once()
        _, kwargs = mock_client.create_collection.call_args
        assert kwargs["collection_name"] == "test_col"
        assert kwargs["vectors_config"].size == 8


def test_upsert_uses_deterministic_ids(mock_qdrant):
    """Re-ingesting identical content must upsert the same point IDs, not duplicate."""
    mock_instance, mock_cls = mock_qdrant
    mock_instance.add_documents.return_value = ["id1"]

    with patch("app.retrieval.vector_store.get_embedder") as mock_emb, \
         patch("app.retrieval.vector_store.get_settings") as mock_s, \
         patch("app.retrieval.vector_store.QdrantClient") as mock_client_cls:
        mock_emb.return_value = MagicMock()
        mock_s.return_value.QDRANT_URL = "http://localhost:6333"
        mock_s.return_value.COLLECTION_NAME = "test_col"
        mock_s.return_value.QDRANT_API_KEY = None
        mock_client_cls.return_value.collection_exists.return_value = True

        docs = [Document(page_content="hello"), Document(page_content="world")]
        vs = VectorStore()
        vs.upsert(docs)
        vs.upsert(docs)

        calls = mock_instance.add_documents.call_args_list
        assert len(calls) == 2
        ids_first = calls[0].kwargs["ids"]
        ids_second = calls[1].kwargs["ids"]
        assert ids_first == ids_second
        assert len(set(ids_first)) == 2


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
