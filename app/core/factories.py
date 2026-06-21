import logging

from langchain_anthropic import ChatAnthropic
from langchain_cohere import CohereRerank
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import get_settings

logger = logging.getLogger(__name__)

_llm_cache: dict[tuple, object] = {}
_embedder_cache: dict[tuple, object] = {}
_reranker_cache: dict[tuple, object] = {}


def get_llm(model: str | None = None):
    settings = get_settings()
    model = model or settings.LLM_MODEL
    key = (settings.LLM_PROVIDER, model)
    if key in _llm_cache:
        return _llm_cache[key]
    if settings.LLM_PROVIDER == "openai":
        instance = ChatOpenAI(
            model=model,
            api_key=settings.LLM_API_KEY or None,
            base_url=settings.LLM_BASE_URL,
            timeout=settings.LLM_TIMEOUT,
        )
    elif settings.LLM_PROVIDER == "anthropic":
        instance = ChatAnthropic(
            model=model,
            api_key=settings.LLM_API_KEY or None,
            base_url=settings.LLM_BASE_URL,
            timeout=settings.LLM_TIMEOUT,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
    _llm_cache[key] = instance
    logger.info("Created LLM client: provider=%s model=%s", settings.LLM_PROVIDER, model)
    return instance


def _content(resp) -> str:
    return getattr(resp, "content", "") or ""


def complete(prompt: str, *, fast: bool = False) -> str:
    """Invoke the routed model (fast vs strong) and fall back to
    LLM_FALLBACK_MODEL (same provider) on any error."""
    settings = get_settings()
    model = settings.LLM_MODEL_FAST if fast else settings.LLM_MODEL
    try:
        return _content(get_llm(model).invoke(prompt))
    except Exception as exc:  # noqa: BLE001
        fallback = settings.LLM_FALLBACK_MODEL
        if not fallback or fallback == model:
            raise
        logger.warning("llm_fallback: model=%s failed (%s), retrying with %s", model, exc, fallback)
        return _content(get_llm(fallback).invoke(prompt))


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
