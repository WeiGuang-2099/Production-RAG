import hashlib
import logging

from langchain_core.documents import Document
from opensearchpy import NotFoundError, OpenSearch, helpers

logger = logging.getLogger(__name__)

# metadata.source is an explicit keyword field so source scoping is an exact
# terms filter, not an analyzed match; everything else in metadata stays
# dynamic. Single shard / zero replicas: single-node dev topology.
_INDEX_BODY = {
    "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
    "mappings": {
        "properties": {
            "content": {"type": "text"},
            "metadata": {
                "type": "object",
                "dynamic": True,
                "properties": {"source": {"type": "keyword"}},
            },
        }
    },
}


def _doc_id(content: str) -> str:
    # Deterministic ID: re-ingesting identical content overwrites instead of
    # duplicating (mirrors VectorStore._point_id's uuid5 approach).
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class OpenSearchStore:
    """Keyword store backed by OpenSearch: incremental indexing, shared state.

    Duck-typed twin of BM25Store (add_documents / search). No staleness
    logic — OpenSearch is server-side shared state, so ingests from another
    process (e.g. the MCP server) are visible immediately. Errors propagate:
    HybridRetriever's per-leg degradation turns a dead keyword leg into
    vector-only results.
    """

    def __init__(self, url: str, index_name: str) -> None:
        self.index_name = index_name
        self._client = OpenSearch(url)
        self._index_ensured = False

    def ping(self) -> bool:
        return bool(self._client.ping())

    def _ensure_index(self) -> None:
        if self._index_ensured:
            return
        if not self._client.indices.exists(index=self.index_name):
            self._client.indices.create(index=self.index_name, body=_INDEX_BODY)
            logger.info("opensearch_index_created: %s", self.index_name)
        self._index_ensured = True

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        self._ensure_index()
        # Only the passed documents are indexed — nothing is rebuilt. This is
        # the incremental win over the local BM25 store. refresh=True makes an
        # ingest-then-query flow see new chunks immediately; fine at demo
        # scale, throughput-hostile at production write rates.
        actions = [
            {
                "_index": self.index_name,
                "_id": _doc_id(doc.page_content),
                "_source": {"content": doc.page_content, "metadata": doc.metadata},
            }
            for doc in documents
        ]
        helpers.bulk(self._client, actions, refresh=True)
        logger.debug("opensearch_indexed: %d documents", len(documents))

    def search(
        self, query: str, top_k: int = 5, sources: list[str] | None = None
    ) -> list[tuple[Document, float]]:
        bool_query: dict = {"must": [{"match": {"content": query}}]}
        if sources:
            bool_query["filter"] = [{"terms": {"metadata.source": list(sources)}}]
        try:
            resp = self._client.search(
                index=self.index_name,
                body={"query": {"bool": bool_query}, "size": top_k},
            )
        except NotFoundError:
            # Nothing ingested yet — matches the empty local store's [].
            return []
        return [
            (
                Document(
                    page_content=hit["_source"]["content"],
                    metadata=hit["_source"].get("metadata", {}),
                ),
                float(hit["_score"]),
            )
            for hit in resp["hits"]["hits"]
        ]
