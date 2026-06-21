import json as _json
from unittest.mock import patch


def test_agent_endpoint_returns_route_and_usage(client):
    with patch("app.api.routes_agent.run_agent") as mock_run:
        mock_run.return_value = {
            "answer": "A", "sources": [], "latency_ms": 5.0,
            "usage": {"cost_usd": 0.0}, "route": "retrieve", "attempts": 1,
        }
        resp = client.post("/agent", json={"question": "q"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["route"] == "retrieve"
        assert data["attempts"] == 1


def test_agent_stream_endpoint_emits_events(client):
    async def fake_stream(question, top_k=None):
        yield {"event": "step", "node": "route"}
        yield {"event": "done", "answer": "A", "usage": {}}

    with patch("app.api.routes_agent.stream_agent", fake_stream):
        resp = client.post("/agent/stream", json={"question": "q"})
        assert resp.status_code == 200
        events = [_json.loads(ln) for ln in resp.text.strip().split("\n") if ln]
        assert events[0]["event"] == "step"
        assert events[-1]["event"] == "done"


def test_agent_missing_question(client):
    assert client.post("/agent", json={}).status_code == 422
