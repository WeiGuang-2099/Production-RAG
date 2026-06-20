from functools import lru_cache

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str | None = None

    # Embedding
    EMBEDDING_PROVIDER: str = "openai"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str | None = None

    # Reranker
    RERANKER_PROVIDER: str = "cohere"
    RERANKER_MODEL: str = "rerank-v3"
    COHERE_API_KEY: str = ""

    # Graph
    GRAPH_EXTRACTOR: str = "llm"

    # Generation
    PROMPT_MODE: str = "grounded"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    COLLECTION_NAME: str = "rag_docs"

    # LangSmith
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "production-rag"
    LANGSMITH_TRACING: bool = False

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # Retrieval
    RETRIEVAL_MODE: str = "hybrid"
    TOP_K: int = 5
    RERANK_TOP_K: int = 3

    # Data
    DATA_DIR: str = "./data"

    # Security
    API_KEY_HASH: str = ""
    CORS_ORIGINS: str = "*"
    MAX_FILE_SIZE_MB: int = 100

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @field_validator("LLM_PROVIDER")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        if v not in ("openai", "anthropic"):
            raise ValueError(f"LLM_PROVIDER must be 'openai' or 'anthropic', got '{v}'")
        return v

    @field_validator("EMBEDDING_PROVIDER")
    @classmethod
    def validate_embedding_provider(cls, v: str) -> str:
        if v not in ("openai",):
            raise ValueError(f"EMBEDDING_PROVIDER must be 'openai', got '{v}'")
        return v

    @field_validator("RERANKER_PROVIDER")
    @classmethod
    def validate_reranker_provider(cls, v: str) -> str:
        if v not in ("none", "cohere"):
            raise ValueError(f"RERANKER_PROVIDER must be 'none' or 'cohere', got '{v}'")
        return v

    @field_validator("GRAPH_EXTRACTOR")
    @classmethod
    def validate_graph_extractor(cls, v: str) -> str:
        if v not in ("none", "llm", "nlp"):
            raise ValueError(f"GRAPH_EXTRACTOR must be 'none', 'llm', or 'nlp', got '{v}'")
        return v

    @field_validator("PROMPT_MODE")
    @classmethod
    def validate_prompt_mode(cls, v: str) -> str:
        if v not in ("basic", "grounded"):
            raise ValueError(f"PROMPT_MODE must be 'basic' or 'grounded', got '{v}'")
        return v

    @field_validator("RETRIEVAL_MODE")
    @classmethod
    def validate_retrieval_mode(cls, v: str) -> str:
        if v not in ("dense", "hybrid"):
            raise ValueError(f"RETRIEVAL_MODE must be 'dense' or 'hybrid', got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_api_keys(self) -> "Settings":
        if self.LLM_PROVIDER == "openai" and self.LLM_API_KEY == "":
            raise ValueError("LLM_API_KEY required when LLM_PROVIDER='openai'")
        if self.EMBEDDING_PROVIDER == "openai" and self.EMBEDDING_API_KEY == "":
            raise ValueError("EMBEDDING_API_KEY required when EMBEDDING_PROVIDER='openai'")
        if self.RERANKER_PROVIDER == "cohere" and self.COHERE_API_KEY == "":
            raise ValueError("COHERE_API_KEY required when RERANKER_PROVIDER='cohere'")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
