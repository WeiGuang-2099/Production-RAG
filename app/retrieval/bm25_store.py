import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Store:
    def __init__(self, data_dir: str = "./data"):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._documents: list[Document] = []
        self._bm25: BM25Okapi | None = None
        self._load()

    def add_documents(self, documents: list[Document]) -> None:
        self._documents.extend(documents)
        tokenized = [_tokenize(doc.page_content) for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized)
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
        with open(self._data_dir / "bm25_index.pkl", "wb") as f:
            pickle.dump({"documents": self._documents, "bm25": self._bm25}, f)

    def _load(self) -> None:
        path = self._data_dir / "bm25_index.pkl"
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._documents = data["documents"]
            self._bm25 = data["bm25"]
