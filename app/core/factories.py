import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_cohere import CohereRerank
from app.config import get_settings

logger = logging.getLogger(__name__)

_llm_cache: dict[tuple, object] = {}
_embedder_cache: dict[tuple, object] = {}
_reranker_cache: dict[tuple, object] = {}


def get_llm():
    settings = get_settings()
    key = (settings.LLM_PROVIDER, settings.LLM_MODEL)
    if key in _llm_cache:
        return _llm_cache[key]
    if settings.LLM_PROVIDER == "openai":
        instance = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY or None,
            base_url=settings.LLM_BASE_URL,
        )
    elif settings.LLM_PROVIDER == "anthropic":
        instance = ChatAnthropic(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY or None,
            base_url=settings.LLM_BASE_URL,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
    _llm_cache[key] = instance
    logger.info("Created LLM client: provider=%s model=%s", settings.LLM_PROVIDER, settings.LLM_MODEL)
    return instance


def get_embedder():
    settings = get_settings()
    key = (settings.EMBEDDING_PROVIDER, settings.EMBEDDING_MODEL)
    if key in _embedder_cache:
        return _embedder_cache[key]
    if settings.EMBEDDING_PROVIDER == "openai":
        instance = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.EMBEDDING_API_KEY or None,
            base_url=settings.EMBEDDING_BASE_URL,
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}")
    _embedder_cache[key] = instance
    logger.info("Created embedder client: provider=%s model=%s", settings.EMBEDDING_PROVIDER, settings.EMBEDDING_MODEL)
    return instance


def get_reranker():
    settings = get_settings()
    key = (settings.RERANKER_PROVIDER, settings.RERANKER_MODEL)
    if key in _reranker_cache:
        return _reranker_cache[key]
    if settings.RERANKER_PROVIDER == "none":
        _reranker_cache[key] = None
        return None
    if settings.RERANKER_PROVIDER == "cohere":
        instance = CohereRerank(
            model=settings.RERANKER_MODEL,
            cohere_api_key=settings.COHERE_API_KEY,
        )
    else:
        raise ValueError(f"Unsupported reranker provider: {settings.RERANKER_PROVIDER}")
    _reranker_cache[key] = instance
    logger.info("Created reranker client: provider=%s model=%s", settings.RERANKER_PROVIDER, settings.RERANKER_MODEL)
    return instance


def clear_caches() -> None:
    """Clear all factory caches. Useful for testing."""
    _llm_cache.clear()
    _embedder_cache.clear()
    _reranker_cache.clear()
