import hashlib
from unittest.mock import patch

KEY = "secret-test-key"
KEY_HASH = hashlib.sha256(KEY.encode()).hexdigest()


def test_auth_disabled_when_hash_not_configured(client):
    with patch("app.api.deps.get_settings") as mock_s, \
         patch("app.api.routes_chat.query_pipeline") as mock_query:
        mock_s.return_value.API_KEY_HASH = ""
        mock_query.return_value = {"answer": "ok", "sources": [], "latency_ms": 1.0}
        response = client.post("/chat", json={"question": "q"})
        assert response.status_code == 200


def test_auth_rejects_missing_header(client):
    with patch("app.api.deps.get_settings") as mock_s:
        mock_s.return_value.API_KEY_HASH = KEY_HASH
        response = client.post("/chat", json={"question": "q"})
        assert response.status_code == 401


def test_auth_rejects_wrong_key(client):
    with patch("app.api.deps.get_settings") as mock_s:
        mock_s.return_value.API_KEY_HASH = KEY_HASH
        response = client.post(
            "/chat",
            json={"question": "q"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert response.status_code == 401


def test_auth_accepts_correct_key(client):
    with patch("app.api.deps.get_settings") as mock_s, \
         patch("app.api.routes_chat.query_pipeline") as mock_query:
        mock_s.return_value.API_KEY_HASH = KEY_HASH
        mock_query.return_value = {"answer": "ok", "sources": [], "latency_ms": 1.0}
        response = client.post(
            "/chat",
            json={"question": "q"},
            headers={"Authorization": f"Bearer {KEY}"},
        )
        assert response.status_code == 200
