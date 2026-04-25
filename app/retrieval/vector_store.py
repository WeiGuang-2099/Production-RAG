from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from app.config import get_settings
from app.core.factories import get_embedder


class VectorStore:
    def __init__(self):
        settings = get_settings()
        self.embedder = get_embedder()
        self.collection_name = settings.COLLECTION_NAME
        self.url = settings.QDRANT_URL
        self.api_key = settings.QDRANT_API_KEY
        self._store: QdrantVectorStore | None = None

    def _get_store(self) -> QdrantVectorStore:
        if self._store is None:
            self._store = QdrantVectorStore.from_existing_collection(
                embedding=self.embedder,
                url=self.url,
                api_key=self.api_key,
                collection_name=self.collection_name,
            )
        return self._store

    def upsert(self, documents: list[Document]) -> list[str]:
        store = self._get_store()
        return store.add_documents(documents)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        store = self._get_store()
        return store.similarity_search_with_score(query, k=top_k)
