import os
from langsmith import traceable
from app.config import get_settings


def setup_tracing() -> None:
    settings = get_settings()
    if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    else:
        os.environ.pop("LANGSMITH_TRACING", None)


@traceable(name="retrieval", run_type="retriever")
def trace_retrieval(query: str, results: list[dict], latency_ms: float) -> dict:
    return {
        "query": query,
        "hit_count": len(results),
        "latency_ms": latency_ms,
        "results": results,
    }
