import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


def test_ingest_endpoint(client):
    with patch("app.api.routes_ingest.ingest_pipeline") as mock_ingest:
        mock_ingest.return_value = {"source": "test.pdf", "chunks": 5}
        response = client.post("/ingest", json={"source": "test.pdf"})
        assert response.status_code == 200
        data = response.json()
        assert data["chunks"] == 5


def test_chat_endpoint(client):
    with patch("app.api.routes_chat.query_pipeline") as mock_query:
        mock_query.return_value = {
            "answer": "AI is artificial intelligence.",
            "sources": [{"content": "context", "metadata": {}}],
            "latency_ms": 100.0,
        }
        response = client.post("/chat", json={"question": "What is AI?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["sources"]) == 1


def test_chat_missing_question(client):
    response = client.post("/chat", json={})
    assert response.status_code == 422


def test_ingest_missing_source(client):
    response = client.post("/ingest", json={})
    assert response.status_code == 422
