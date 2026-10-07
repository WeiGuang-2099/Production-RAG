import json
import logging

import pytest
import structlog
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.observability.logging import RequestIDMiddleware, setup_logging


@pytest.fixture
def log_lines(capsys):
    """Configure app logging; return a reader of the JSON lines it wrote to stdout."""
    setup_logging()
    capsys.readouterr()  # drop anything written before the test body
    seen: list[dict] = []

    def read() -> list[dict]:
        out = capsys.readouterr().out
        seen.extend(json.loads(line) for line in out.splitlines() if line.startswith("{"))
        return seen

    yield read
    structlog.contextvars.clear_contextvars()


def test_stdlib_logs_render_as_json_with_request_id(log_lines):
    # App modules log through stdlib loggers; they must reach the JSON output.
    structlog.contextvars.bind_contextvars(request_id="rid-123")
    logging.getLogger("app.core.pipeline").info("query_complete: cost_usd=%.6f", 0.0065)

    [line] = [ln for ln in log_lines() if ln.get("logger") == "app.core.pipeline"]
    assert line["event"] == "query_complete: cost_usd=0.006500"
    assert line["level"] == "info"
    assert line["request_id"] == "rid-123"
    assert "timestamp" in line


def test_setup_logging_is_idempotent(log_lines):
    setup_logging()
    setup_logging()
    logging.getLogger("app.test").info("once")

    assert [ln["event"] for ln in log_lines() if ln.get("logger") == "app.test"] == ["once"]


def test_exceptions_render_inside_the_json_line(log_lines):
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("app.test").exception("ingest_failed")

    [line] = [ln for ln in log_lines() if ln.get("event") == "ingest_failed"]
    assert "ValueError: boom" in line["exception"]


@pytest.mark.parametrize("name", ["httpx", "httpx2", "opensearch"])
def test_http_client_request_chatter_is_suppressed(log_lines, name):
    logging.getLogger(name).info("HTTP Request: POST https://api.openai.com/v1/embeddings")

    assert not [ln for ln in log_lines() if ln.get("logger") == name]


def test_middleware_request_id_reaches_route_logs(log_lines):
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/ping")
    def ping():  # sync route: runs in a worker thread, like the pipeline calls
        logging.getLogger("app.api.test").info("ping_handled")
        return {"ok": True}

    resp = TestClient(app).get("/ping", headers={"X-Request-ID": "rid-abc"})

    assert resp.headers["X-Request-ID"] == "rid-abc"
    [line] = [ln for ln in log_lines() if ln.get("event") == "ping_handled"]
    assert line["request_id"] == "rid-abc"
