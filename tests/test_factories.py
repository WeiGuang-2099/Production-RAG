import pytest
from unittest.mock import MagicMock, patch
from app.core.factories import get_llm, get_embedder, get_reranker, clear_caches


@pytest.fixture(autouse=True)
def _reset_caches():
    clear_caches()
    yield
    clear_caches()


def test_get_llm_openai():
    with patch("app.core.factories.ChatOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        with patch("app.core.factories.get_settings") as mock_settings:
            mock_settings.return_value.LLM_PROVIDER = "openai"
            mock_settings.return_value.LLM_MODEL = "gpt-4o"
            mock_settings.return_value.LLM_API_KEY = "sk-test"
            mock_settings.return_value.LLM_BASE_URL = None
            llm = get_llm()
            mock_cls.assert_called_once()


def test_get_llm_anthropic():
    with patch("app.core.factories.ChatAnthropic") as mock_cls:
        mock_cls.return_value = MagicMock()
        with patch("app.core.factories.get_settings") as mock_settings:
            mock_settings.return_value.LLM_PROVIDER = "anthropic"
            mock_settings.return_value.LLM_MODEL = "claude-sonnet-4-20250514"
            mock_settings.return_value.LLM_API_KEY = "sk-ant-test"
            mock_settings.return_value.LLM_BASE_URL = None
            llm = get_llm()
            mock_cls.assert_called_once()


def test_get_llm_unsupported():
    with patch("app.core.factories.get_settings") as mock_settings:
        mock_settings.return_value.LLM_PROVIDER = "ollama"
        mock_settings.return_value.LLM_MODEL = "llama3"
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_llm()


def test_get_embedder_openai():
    with patch("app.core.factories.OpenAIEmbeddings") as mock_cls:
        mock_cls.return_value = MagicMock()
        with patch("app.core.factories.get_settings") as mock_settings:
            mock_settings.return_value.EMBEDDING_PROVIDER = "openai"
            mock_settings.return_value.EMBEDDING_MODEL = "text-embedding-3-small"
            mock_settings.return_value.EMBEDDING_API_KEY = "sk-test"
            mock_settings.return_value.EMBEDDING_BASE_URL = None
            embedder = get_embedder()
            mock_cls.assert_called_once()


def test_get_reranker_cohere():
    with patch("app.core.factories.CohereRerank") as mock_cls:
        mock_cls.return_value = MagicMock()
        with patch("app.core.factories.get_settings") as mock_settings:
            mock_settings.return_value.RERANKER_PROVIDER = "cohere"
            mock_settings.return_value.RERANKER_MODEL = "rerank-v3"
            mock_settings.return_value.COHERE_API_KEY = "test-key"
            reranker = get_reranker()
            mock_cls.assert_called_once()


def test_get_reranker_none():
    with patch("app.core.factories.get_settings") as mock_settings:
        mock_settings.return_value.RERANKER_PROVIDER = "none"
        mock_settings.return_value.RERANKER_MODEL = "rerank-v3"
        reranker = get_reranker()
        assert reranker is None


def test_caching_returns_same_instance():
    """Verify that calling get_llm twice returns the cached instance."""
    with patch("app.core.factories.ChatOpenAI") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        with patch("app.core.factories.get_settings") as mock_settings:
            mock_settings.return_value.LLM_PROVIDER = "openai"
            mock_settings.return_value.LLM_MODEL = "gpt-4o"
            mock_settings.return_value.LLM_API_KEY = "sk-test"
            mock_settings.return_value.LLM_BASE_URL = None
            llm1 = get_llm()
            llm2 = get_llm()
            assert llm1 is llm2
            mock_cls.assert_called_once()  # Only created once


def test_clear_caches():
    with patch("app.core.factories.ChatOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        with patch("app.core.factories.get_settings") as mock_settings:
            mock_settings.return_value.LLM_PROVIDER = "openai"
            mock_settings.return_value.LLM_MODEL = "gpt-4o"
            mock_settings.return_value.LLM_API_KEY = "sk-test"
            mock_settings.return_value.LLM_BASE_URL = None
            get_llm()
            clear_caches()
            get_llm()
            assert mock_cls.call_count == 2  # Created twice after cache clear
