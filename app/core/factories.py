from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_cohere import CohereRerank
from app.config import get_settings


def get_llm():
    settings = get_settings()
    if settings.LLM_PROVIDER == "openai":
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY or None,
            base_url=settings.LLM_BASE_URL,
        )
    elif settings.LLM_PROVIDER == "anthropic":
        return ChatAnthropic(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY or None,
            base_url=settings.LLM_BASE_URL,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")


def get_embedder():
    settings = get_settings()
    if settings.EMBEDDING_PROVIDER == "openai":
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.EMBEDDING_API_KEY or None,
            base_url=settings.EMBEDDING_BASE_URL,
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}")


def get_reranker():
    settings = get_settings()
    if settings.RERANKER_PROVIDER == "none":
        return None
    if settings.RERANKER_PROVIDER == "cohere":
        return CohereRerank(
            model=settings.RERANKER_MODEL,
            cohere_api_key=settings.COHERE_API_KEY,
        )
    raise ValueError(f"Unsupported reranker provider: {settings.RERANKER_PROVIDER}")
