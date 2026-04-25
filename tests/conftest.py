import os
import pytest

# Set required env vars BEFORE importing app to prevent validation errors
os.environ.setdefault("LLM_API_KEY", "sk-test-key")
os.environ.setdefault("EMBEDDING_API_KEY", "sk-test-key")
os.environ.setdefault("COHERE_API_KEY", "test-cohere-key")

from fastapi.testclient import TestClient
from app.main import app
from app.core.factories import clear_caches


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_factory_caches():
    """Clear factory caches before each test."""
    clear_caches()
    yield
    clear_caches()
