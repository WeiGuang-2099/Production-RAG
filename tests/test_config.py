import os
import pytest
from app.config import Settings


def test_default_settings():
    settings = Settings()
    assert settings.LLM_PROVIDER == "openai"
    assert settings.LLM_MODEL == "gpt-4o"
    assert settings.EMBEDDING_PROVIDER == "openai"
    assert settings.CHUNK_SIZE == 512
    assert settings.CHUNK_OVERLAP == 64
    assert settings.TOP_K == 5
    assert settings.RERANK_TOP_K == 3
    assert settings.COLLECTION_NAME == "rag_docs"


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-20250514")
    monkeypatch.setenv("CHUNK_SIZE", "1024")
    settings = Settings()
    assert settings.LLM_PROVIDER == "anthropic"
    assert settings.LLM_MODEL == "claude-sonnet-4-20250514"
    assert settings.CHUNK_SIZE == 1024
