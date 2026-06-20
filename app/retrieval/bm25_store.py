import logging
import pickle
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Store:
    def __init__(self, data_dir: str = "./data"):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._documents: list[Document] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._load()

    def add_documents(self, documents: list[Document]) -> None:
        # Skip content already indexed so re-ingestion stays idempotent
        existing = {doc.page_content for doc in self._documents}
        new_docs = []
        for doc in documents:
            if doc.page_content not in existing:
                new_docs.append(doc)
                existing.add(doc.page_content)
        if not new_docs:
            return
        self._documents.extend(new_docs)
        self._tokenized_corpus.extend(_tokenize(doc.page_content) for doc in new_docs)
        if len(self._documents) > 50000:
            logger.warning("BM25 index has %d documents; consider migrating to Elasticsearch for scale", len(self._documents))
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._save()

    def search(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        if self._bm25 is None or not self._documents:
            return []
        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._documents[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    def _save(self) -> None:
        if self._bm25 is None:
            return
        try:
            with open(self._data_dir / "bm25_index.pkl", "wb") as f:
                pickle.dump({
                    "documents": self._documents,
                    "tokenized_corpus": self._tokenized_corpus,
                    "bm25": self._bm25,
                }, f)
            logger.debug("BM25 index saved: %d documents", len(self._documents))
        except Exception as exc:
            logger.error("Failed to save BM25 index: %s", exc)

    def _load(self) -> None:
        path = self._data_dir / "bm25_index.pkl"
        if not path.exists():
            return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._documents = data["documents"]
            self._tokenized_corpus = data.get("tokenized_corpus", [])
            self._bm25 = data["bm25"]
            logger.info("BM25 index loaded: %d documents", len(self._documents))
        except Exception as exc:
            logger.error("Failed to load BM25 index: %s", exc)
