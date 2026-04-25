import logging
import time
from langchain_core.embeddings import Embeddings
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class Embedder:
    BATCH_SIZE = 100

    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if len(texts) <= self.BATCH_SIZE:
            return self.embeddings.embed_documents(texts)

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            logger.debug("Embedding batch %d/%d (%d texts)", i // self.BATCH_SIZE + 1, (len(texts) - 1) // self.BATCH_SIZE + 1, len(batch))
            embeddings = self.embeddings.embed_documents(batch)
            all_embeddings.extend(embeddings)
            if i + self.BATCH_SIZE < len(texts):
                time.sleep(0.05)
        logger.info("Embedded %d texts in %d batches", len(texts), (len(texts) - 1) // self.BATCH_SIZE + 1)
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        return self.embeddings.embed_query(query)
