import hashlib
from unittest.mock import patch

from langchain_core.documents import Document
from opensearchpy import NotFoundError

from app.retrieval.opensearch_store import OpenSearchStore


def _store():
    with patch("app.retrieval.opensearch_store.OpenSearch") as mock_cls:
        store = OpenSearchStore(url="http://test:9200", index_name="test_idx")
    return store, mock_cls.return_value


def _hits_response(*pairs):
    return {
        "hits": {
            "hits": [
                {"_source": {"content": c, "metadata": m}, "_score": s}
                for c, m, s in pairs
            ]
        }
    }


def test_index_created_once_with_keyword_source_mapping():
    store, client = _store()
    client.indices.exists.return_value = False

    with patch("app.retrieval.opensearch_store.helpers"):
        store.add_documents([Document(page_content="a")])
        store.add_documents([Document(page_content="b")])

    client.indices.create.assert_called_once()
    body = client.indices.create.call_args.kwargs["body"]
    assert body["mappings"]["properties"]["content"]["type"] == "text"
    assert body["mappings"]["properties"]["metadata"]["properties"]["source"]["type"] == "keyword"
    assert body["settings"]["index"]["number_of_replicas"] == 0


def test_add_documents_bulk_uses_sha256_ids_and_refresh():
    store, client = _store()
    client.indices.exists.return_value = True
    docs = [
        Document(page_content="alpha", metadata={"source": "a.pdf"}),
        Document(page_content="beta", metadata={"source": "b.pdf"}),
    ]

    with patch("app.retrieval.opensearch_store.helpers") as mock_helpers:
        store.add_documents(docs)

    (passed_client, actions), kwargs = mock_helpers.bulk.call_args
    assert passed_client is client
    assert kwargs["refresh"] is True
    actions = list(actions)
    assert [a["_id"] for a in actions] == [
        hashlib.sha256(b"alpha").hexdigest(),
        hashlib.sha256(b"beta").hexdigest(),
    ]
    assert actions[0]["_source"] == {"content": "alpha", "metadata": {"source": "a.pdf"}}
    assert all(a["_index"] == "test_idx" for a in actions)


def test_add_documents_empty_is_noop():
    store, client = _store()
    with patch("app.retrieval.opensearch_store.helpers") as mock_helpers:
        store.add_documents([])
    mock_helpers.bulk.assert_not_called()
    client.indices.create.assert_not_called()


def test_search_maps_hits_to_documents_and_scores():
    store, client = _store()
    client.search.return_value = _hits_response(
        ("alpha text", {"source": "a.pdf"}, 3.2),
        ("beta text", {"source": "b.pdf"}, 1.1),
    )

    results = store.search("alpha", top_k=2)

    body = client.search.call_args.kwargs["body"]
    assert body["size"] == 2
    assert body["query"]["bool"]["must"] == [{"match": {"content": "alpha"}}]
    assert "filter" not in body["query"]["bool"]
    assert [(d.page_content, s) for d, s in results] == [
        ("alpha text", 3.2), ("beta text", 1.1),
    ]
    assert results[0][0].metadata == {"source": "a.pdf"}


def test_search_scoped_adds_terms_filter():
    store, client = _store()
    client.search.return_value = _hits_response()

    store.search("q", top_k=5, sources=["a.pdf", "b.pdf"])

    body = client.search.call_args.kwargs["body"]
    assert body["query"]["bool"]["filter"] == [{"terms": {"metadata.source": ["a.pdf", "b.pdf"]}}]


def test_search_missing_index_returns_empty():
    store, client = _store()
    client.search.side_effect = NotFoundError(404, "index_not_found_exception", {})
    assert store.search("anything") == []


def test_ping_delegates_to_client():
    store, client = _store()
    client.ping.return_value = True
    assert store.ping() is True
