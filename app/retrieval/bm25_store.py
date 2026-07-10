import logging
import pickle
import threading
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Store:
    """BM25 index persisted to <data_dir>/bm25_index.pkl.

    Instances are shared across queries (cached by app.core.factories), so
    state is held in a single immutable tuple swapped atomically, and reads
    check the pickle's mtime to pick up writes from other processes (the MCP
    server can ingest while the API serves queries).
    """

    def __init__(self, data_dir: str = "./data"):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # (documents, tokenized_corpus, bm25) — replaced wholesale, never mutated
        self._state: tuple[list[Document], list[list[str]], BM25Okapi | None] = ([], [], None)
        self._signature: tuple[int, int] | None = None
        self._load()

    # Kept as properties: existing tests and debugging poke these names.
    @property
    def _documents(self) -> list[Document]:
        return self._state[0]

    @property
    def _tokenized_corpus(self) -> list[list[str]]:
        return self._state[1]

    @property
    def _bm25(self) -> BM25Okapi | None:
        return self._state[2]

    def _index_path(self) -> Path:
        return self._data_dir / "bm25_index.pkl"

    def _index_signature(self) -> tuple[int, int] | None:
        """(st_mtime_ns, st_size) of the pickle; None when absent. Size is
        included because filesystem mtime granularity can miss writes that
        land within the same clock tick."""
        try:
            st = self._index_path().stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def refresh_if_stale(self) -> None:
        """Reload from disk if another process rewrote the index file."""
        sig = self._index_signature()
        if sig is None or sig == self._signature:
            return
        with self._lock:
            if self._index_signature() != self._signature:
                self._load()

    def add_documents(self, documents: list[Document]) -> None:
        with self._lock:
            docs, corpus, _ = self._state
            existing = {doc.page_content for doc in docs}
            new_docs = []
            for doc in documents:
                if doc.page_content not in existing:
                    new_docs.append(doc)
                    existing.add(doc.page_content)
            if not new_docs:
                return
            docs = docs + new_docs
            corpus = corpus + [_tokenize(doc.page_content) for doc in new_docs]
            if len(docs) > 50000:
                logger.warning("BM25 index has %d documents; consider migrating to Elasticsearch for scale", len(docs))
            self._state = (docs, corpus, BM25Okapi(corpus))
            self._save()

    def search(
        self, query: str, top_k: int = 5, sources: list[str] | None = None
    ) -> list[tuple[Document, float]]:
        self.refresh_if_stale()
        docs, _, bm25 = self._state
        if bm25 is None or not docs:
            return []
        tokenized_query = _tokenize(query)
        scores = bm25.get_scores(tokenized_query)
        indices = range(len(scores))
        if sources:
            allowed = set(sources)
            indices = [i for i in indices if docs[i].metadata.get("source") in allowed]
        top_indices = sorted(indices, key=lambda i: scores[i], reverse=True)[:top_k]
        return [(docs[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    def _save(self) -> None:
        docs, corpus, bm25 = self._state
        if bm25 is None:
            return
        try:
            with open(self._index_path(), "wb") as f:
                pickle.dump({
                    "documents": docs,
                    "tokenized_corpus": corpus,
                    "bm25": bm25,
                }, f)
            self._signature = self._index_signature()
            logger.debug("BM25 index saved: %d documents", len(docs))
        except Exception as exc:
            logger.error("Failed to save BM25 index: %s", exc)

    def _load(self) -> None:
        path = self._index_path()
        if not path.exists():
            return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._state = (data["documents"], data.get("tokenized_corpus", []), data["bm25"])
            self._signature = self._index_signature()
            logger.info("BM25 index loaded: %d documents", len(data["documents"]))
        except Exception as exc:
            logger.error("Failed to load BM25 index: %s", exc)
