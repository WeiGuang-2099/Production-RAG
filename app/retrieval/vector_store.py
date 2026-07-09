import uuid

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from qdrant_client.models import Distance, VectorParams

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

    def _ensure_collection(self) -> None:
        client = QdrantClient(url=self.url, api_key=self.api_key)
        try:
            if not client.collection_exists(self.collection_name):
                dim = len(self.embedder.embed_query("dimension probe"))
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
        finally:
            client.close()

    def _get_store(self) -> QdrantVectorStore:
        if self._store is None:
            self._store = QdrantVectorStore.from_existing_collection(
                embedding=self.embedder,
                url=self.url,
                api_key=self.api_key,
                collection_name=self.collection_name,
            )
        return self._store

    @staticmethod
    def _point_id(document: Document) -> str:
        # Deterministic ID so re-ingesting identical content overwrites instead of duplicating
        return str(uuid.uuid5(uuid.NAMESPACE_URL, document.page_content))

    def upsert(self, documents: list[Document]) -> list[str]:
        self._ensure_collection()
        store = self._get_store()
        ids = [self._point_id(doc) for doc in documents]
        return store.add_documents(documents, ids=ids)

    def search(
        self, query: str, top_k: int = 5, sources: list[str] | None = None
    ) -> list[tuple[Document, float]]:
        store = self._get_store()
        flt = None
        if sources:
            flt = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="metadata.source",
                        match=qmodels.MatchAny(any=list(sources)),
                    )
                ]
            )
        return store.similarity_search_with_score(query, k=top_k, filter=flt)
