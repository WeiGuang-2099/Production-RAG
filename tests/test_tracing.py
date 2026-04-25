import pytest
import importlib
from unittest.mock import patch, MagicMock


def test_setup_tracing_enabled():
    with patch("app.observability.tracing.get_settings") as mock_s:
        mock_s.return_value.LANGSMITH_TRACING = True
        mock_s.return_value.LANGSMITH_API_KEY = "test-key"
        mock_s.return_value.LANGSMITH_PROJECT = "test-project"
        with patch.dict("os.environ", {}) as mock_env:
            from app.observability.tracing import setup_tracing
            setup_tracing()
            assert mock_env.get("LANGSMITH_TRACING") == "true"
            assert mock_env.get("LANGSMITH_API_KEY") == "test-key"


def test_setup_tracing_disabled():
    with patch("app.observability.tracing.get_settings") as mock_s:
        mock_s.return_value.LANGSMITH_TRACING = False
        mock_s.return_value.LANGSMITH_API_KEY = ""
        with patch.dict("os.environ", {}) as mock_env:
            from app.observability.tracing import setup_tracing
            setup_tracing()
            assert mock_env.get("LANGSMITH_TRACING", "false") != "true"


def test_trace_retrieval():
    with patch("app.observability.tracing.traceable", side_effect=lambda **kwargs: (lambda f: f)):
        import app.observability.tracing as tracing_mod
        importlib.reload(tracing_mod)
        result = tracing_mod.trace_retrieval(
            "test query", [{"doc": "result", "score": 0.9}], 500.0
        )
        assert result["query"] == "test query"
        assert result["hit_count"] == 1
        assert result["latency_ms"] == 500.0
