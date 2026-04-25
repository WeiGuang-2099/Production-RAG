import os
import pytest
from app.config import Settings


def test_default_settings():
    """Test defaults with required API keys provided."""
    settings = Settings(
        LLM_API_KEY="sk-test",
        EMBEDDING_API_KEY="sk-test",
        COHERE_API_KEY="test-key",
    )
    assert settings.LLM_PROVIDER == "openai"
    assert settings.LLM_MODEL == "gpt-4o"
    assert settings.EMBEDDING_PROVIDER == "openai"
    assert settings.CHUNK_SIZE == 512
    assert settings.CHUNK_OVERLAP == 64
    assert settings.TOP_K == 5
    assert settings.RERANK_TOP_K == 3
    assert settings.COLLECTION_NAME == "rag_docs"
    assert settings.API_KEY_HASH == ""
    assert settings.CORS_ORIGINS == "*"
    assert settings.MAX_FILE_SIZE_MB == 100


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-20250514")
    monkeypatch.setenv("LLM_API_KEY", "sk-ant-test")
    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test")
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    monkeypatch.setenv("CHUNK_SIZE", "1024")
    settings = Settings()
    assert settings.LLM_PROVIDER == "anthropic"
    assert settings.LLM_MODEL == "claude-sonnet-4-20250514"
    assert settings.CHUNK_SIZE == 1024


def test_invalid_llm_provider():
    with pytest.raises(ValueError, match="LLM_PROVIDER must be"):
        Settings(
            LLM_PROVIDER="ollama",
            LLM_API_KEY="test",
            EMBEDDING_API_KEY="test",
            COHERE_API_KEY="test",
        )


def test_missing_api_key_openai():
    with pytest.raises(ValueError, match="LLM_API_KEY required"):
        Settings(LLM_PROVIDER="openai", LLM_API_KEY="", EMBEDDING_API_KEY="test", COHERE_API_KEY="test")


def test_invalid_reranker_provider():
    with pytest.raises(ValueError, match="RERANKER_PROVIDER must be"):
        Settings(
            RERANKER_PROVIDER="invalid",
            LLM_API_KEY="test",
            EMBEDDING_API_KEY="test",
            COHERE_API_KEY="test",
        )


def test_reranker_none_no_cohere_key():
    """When reranker is 'none', COHERE_API_KEY is not required."""
    settings = Settings(
        RERANKER_PROVIDER="none",
        LLM_API_KEY="test",
        EMBEDDING_API_KEY="test",
        COHERE_API_KEY="",
    )
    assert settings.RERANKER_PROVIDER == "none"
