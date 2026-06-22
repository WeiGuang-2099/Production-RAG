import logging

from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def _is_rate_limit(exc: BaseException) -> bool:
    """True for Cohere 429 / rate-limit errors (notably trial keys: 10 req/min).

    Only these are worth retrying; other errors (bad request, auth) should fail
    fast so the caller can degrade to unranked results immediately.
    """
    if getattr(exc, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


class RerankerService:
    def __init__(self, reranker: CohereRerank | None = None):
        self.reranker = reranker

    @retry(
        retry=retry_if_exception(_is_rate_limit),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _rerank_call(self, query: str, documents: list[Document], top_k: int):
        """Call Cohere, retrying with backoff on rate-limit (429) only.

        After the attempts are exhausted the error is re-raised; the pipeline
        catches it and falls back to unranked results (graceful degradation),
        rather than silently dropping rerank on the first transient 429.
        """
        return self.reranker.rerank(
            documents=[doc.page_content for doc in documents],
            query=query,
            top_n=top_k,
        )

    def rerank(self, query: str, documents: list[Document], top_k: int = 3) -> list[Document]:
        if self.reranker is None:
            return documents[:top_k]

        results = self._rerank_call(query, documents, top_k)
        reranked = []
        for result in results:
            doc = documents[result["index"]]
            new_doc = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "relevance_score": result["relevance_score"]},
            )
            reranked.append(new_doc)
        return reranked
