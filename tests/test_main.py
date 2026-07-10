def test_health_live(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_health_ready_reports_opensearch_when_configured(client, monkeypatch):
    from unittest.mock import MagicMock, patch

    from app.config import get_settings

    monkeypatch.setenv("KEYWORD_BACKEND", "opensearch")
    get_settings.cache_clear()
    try:
        store = MagicMock()
        store.ping.return_value = True
        with patch("app.retrieval.vector_store.VectorStore"), \
             patch("app.core.factories.get_keyword_store", return_value=store):
            resp = client.get("/health/ready")
        assert resp.status_code == 200
        assert resp.json()["checks"]["opensearch"] == "ok"
    finally:
        get_settings.cache_clear()


def test_health_ready_omits_opensearch_for_local_backend(client):
    from unittest.mock import patch

    with patch("app.retrieval.vector_store.VectorStore"):
        resp = client.get("/health/ready")
    assert "opensearch" not in resp.json()["checks"]
