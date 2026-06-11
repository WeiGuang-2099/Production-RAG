import pytest
from unittest.mock import patch, MagicMock


def test_ingest_endpoint(client):
    with patch("app.api.routes_ingest.ingest_pipeline") as mock_ingest:
        mock_ingest.return_value = {"source": "test.pdf", "chunks": 5, "status": "ingested"}
        response = client.post("/ingest", json={"source": "https://example.com/doc.pdf"})
        assert response.status_code == 200
        data = response.json()
        assert data["chunks"] == 5
        assert data["status"] == "ingested"


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
        assert "total_sources" in data


def test_chat_passes_top_k_to_pipeline(client):
    with patch("app.api.routes_chat.query_pipeline") as mock_query:
        mock_query.return_value = {
            "answer": "ok",
            "sources": [],
            "latency_ms": 1.0,
        }
        response = client.post("/chat", json={"question": "What is AI?", "top_k": 7})
        assert response.status_code == 200
        mock_query.assert_called_once_with("What is AI?", 7)


def test_ingest_rejects_sibling_prefix_path(client, tmp_path):
    """A path like <DATA_DIR>-evil must not pass the DATA_DIR containment check."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    evil_dir = tmp_path / "data-evil"
    evil_dir.mkdir()
    evil_file = evil_dir / "doc.md"
    evil_file.write_text("outside data dir")

    with patch("app.api.routes_ingest.get_settings") as mock_s, \
         patch("app.api.routes_ingest.ingest_pipeline") as mock_ingest:
        mock_s.return_value.DATA_DIR = str(data_dir)
        mock_s.return_value.MAX_FILE_SIZE_MB = 100
        response = client.post("/ingest", json={"source": str(evil_file)})
        assert response.status_code == 422
        mock_ingest.assert_not_called()


def test_ingest_accepts_path_inside_data_dir(client, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    good_file = data_dir / "doc.md"
    good_file.write_text("inside data dir")

    with patch("app.api.routes_ingest.get_settings") as mock_s, \
         patch("app.api.routes_ingest.ingest_pipeline") as mock_ingest:
        mock_s.return_value.DATA_DIR = str(data_dir)
        mock_s.return_value.MAX_FILE_SIZE_MB = 100
        mock_ingest.return_value = {"source": str(good_file), "chunks": 1, "status": "ingested"}
        response = client.post("/ingest", json={"source": str(good_file)})
        assert response.status_code == 200


def test_chat_missing_question(client):
    response = client.post("/chat", json={})
    assert response.status_code == 422


def test_ingest_missing_source(client):
    response = client.post("/ingest", json={})
    assert response.status_code == 422


def test_health_live(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"
