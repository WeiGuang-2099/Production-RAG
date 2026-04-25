import pytest
from unittest.mock import MagicMock, patch
from app.core.factories import get_llm, get_embedder, get_reranker


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
        reranker = get_reranker()
        assert reranker is None
