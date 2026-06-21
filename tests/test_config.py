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


def test_prompt_mode_default_is_grounded():
    settings = Settings(
        LLM_API_KEY="test", EMBEDDING_API_KEY="test", COHERE_API_KEY="test"
    )
    assert settings.PROMPT_MODE == "grounded"


def test_invalid_prompt_mode():
    with pytest.raises(ValueError, match="PROMPT_MODE must be"):
        Settings(
            PROMPT_MODE="creative",
            LLM_API_KEY="test",
            EMBEDDING_API_KEY="test",
            COHERE_API_KEY="test",
        )


def test_retrieval_mode_default_is_hybrid():
    settings = Settings(
        LLM_API_KEY="test", EMBEDDING_API_KEY="test", COHERE_API_KEY="test"
    )
    assert settings.RETRIEVAL_MODE == "hybrid"


def test_invalid_retrieval_mode():
    with pytest.raises(ValueError, match="RETRIEVAL_MODE must be"):
        Settings(
            RETRIEVAL_MODE="magic",
            LLM_API_KEY="test",
            EMBEDDING_API_KEY="test",
            COHERE_API_KEY="test",
        )


def test_query_transform_default_is_none():
    settings = Settings(
        LLM_API_KEY="test", EMBEDDING_API_KEY="test", COHERE_API_KEY="test"
    )
    assert settings.QUERY_TRANSFORM == "none"


def test_invalid_query_transform():
    with pytest.raises(ValueError, match="QUERY_TRANSFORM must be"):
        Settings(
            QUERY_TRANSFORM="telepathy",
            LLM_API_KEY="test",
            EMBEDDING_API_KEY="test",
            COHERE_API_KEY="test",
        )


def test_cache_defaults():
    settings = Settings(
        LLM_API_KEY="test", EMBEDDING_API_KEY="test", COHERE_API_KEY="test"
    )
    assert settings.CACHE_ENABLED is False
    assert settings.CACHE_SIMILARITY_THRESHOLD == 0.95


def test_invalid_cache_threshold():
    with pytest.raises(ValueError, match="CACHE_SIMILARITY_THRESHOLD"):
        Settings(
            CACHE_SIMILARITY_THRESHOLD=1.5,
            LLM_API_KEY="test",
            EMBEDDING_API_KEY="test",
            COHERE_API_KEY="test",
        )


def test_agent_max_rewrites_default():
    settings = Settings(
        LLM_API_KEY="test", EMBEDDING_API_KEY="test", COHERE_API_KEY="test"
    )
    assert settings.AGENT_MAX_REWRITES == 2


def test_invalid_agent_max_rewrites():
    with pytest.raises(ValueError, match="AGENT_MAX_REWRITES must be"):
        Settings(
            AGENT_MAX_REWRITES=-1,
            LLM_API_KEY="test",
            EMBEDDING_API_KEY="test",
            COHERE_API_KEY="test",
        )


def test_llm_routing_defaults():
    s = Settings(LLM_API_KEY="t", EMBEDDING_API_KEY="t", COHERE_API_KEY="t")
    assert s.LLM_MODEL_FAST == "gpt-4o-mini"
    assert s.LLM_FALLBACK_MODEL == "gpt-4o-mini"
    assert s.LLM_TIMEOUT == 30


def test_invalid_llm_timeout():
    with pytest.raises(ValueError, match="LLM_TIMEOUT must be"):
        Settings(LLM_TIMEOUT=0, LLM_API_KEY="t", EMBEDDING_API_KEY="t", COHERE_API_KEY="t")
