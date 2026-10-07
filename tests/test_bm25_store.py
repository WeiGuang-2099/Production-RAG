from langchain_core.documents import Document

from app.retrieval.bm25_store import BM25Store


def test_add_and_search(tmp_path):
    store = BM25Store(data_dir=str(tmp_path))
    docs = [
        Document(page_content="Machine learning is a subset of artificial intelligence.", metadata={"source": "a"}),
        Document(page_content="Python is a popular programming language.", metadata={"source": "b"}),
        Document(page_content="Neural networks are used in deep learning.", metadata={"source": "c"}),
    ]
    store.add_documents(docs)

    results = store.search("machine learning", top_k=2)
    assert len(results) >= 1
    assert results[0][0].page_content == docs[0].page_content
    assert results[0][1] > 0


def test_search_empty(tmp_path):
    store = BM25Store(data_dir=str(tmp_path))
    results = store.search("test", top_k=5)
    assert results == []


def test_add_documents_is_idempotent(tmp_path):
    """Re-ingesting the same chunks must not duplicate them in the index."""
    store = BM25Store(data_dir=str(tmp_path))
    docs = [
        Document(page_content="Machine learning is a subset of artificial intelligence.", metadata={"source": "a"}),
        Document(page_content="Python is a popular programming language.", metadata={"source": "b"}),
        Document(page_content="Neural networks are used in deep learning.", metadata={"source": "c"}),
    ]
    store.add_documents(docs)
    store.add_documents(docs)

    results = store.search("machine learning", top_k=10)
    matching = [d for d, _ in results if d.page_content == docs[0].page_content]
    assert len(matching) == 1

    # Persisted store must also be free of duplicates
    store2 = BM25Store(data_dir=str(tmp_path))
    results2 = store2.search("machine learning", top_k=10)
    matching2 = [d for d, _ in results2 if d.page_content == docs[0].page_content]
    assert len(matching2) == 1


def test_search_filters_by_source(tmp_path):
    store = BM25Store(data_dir=str(tmp_path))
    docs = [
        Document(page_content="Machine learning is a subset of artificial intelligence.", metadata={"source": "a"}),
        Document(page_content="Machine learning powers modern search.", metadata={"source": "b"}),
        Document(page_content="Neural networks are used in deep learning.", metadata={"source": "c"}),
    ]
    store.add_documents(docs)

    results = store.search("machine learning", top_k=10, sources=["b"])
    assert len(results) == 1
    assert results[0][0].metadata["source"] == "b"


def test_search_empty_sources_means_all(tmp_path):
    store = BM25Store(data_dir=str(tmp_path))
    docs = [
        Document(page_content="Machine learning is a subset of artificial intelligence.", metadata={"source": "a"}),
        Document(page_content="Machine learning powers modern search.", metadata={"source": "b"}),
        # A non-matching doc keeps the query terms' IDF positive (BM25 scores 0 when a term is in every doc)
        Document(page_content="Neural networks are used in deep vision.", metadata={"source": "c"}),
    ]
    store.add_documents(docs)

    assert len(store.search("machine learning", top_k=10, sources=[])) >= 2
    assert len(store.search("machine learning", top_k=10)) >= 2


def test_save_and_load(tmp_path):
    store1 = BM25Store(data_dir=str(tmp_path))
    docs = [
        Document(page_content="Machine learning is a subset of artificial intelligence.", metadata={"source": "a"}),
        Document(page_content="Python is a popular programming language.", metadata={"source": "b"}),
        Document(page_content="Neural networks are used in deep learning.", metadata={"source": "c"}),
    ]
    store1.add_documents(docs)

    store2 = BM25Store(data_dir=str(tmp_path))
    results = store2.search("machine learning", top_k=2)
    assert len(results) >= 1
    assert results[0][0].page_content == docs[0].page_content


def test_search_reloads_after_external_write(tmp_path):
    writer = BM25Store(data_dir=str(tmp_path))
    # Filler docs keep the query terms' IDF positive (BM25 scores <= 0 when a
    # term is in every doc, and search drops non-positive scores)
    writer.add_documents([
        Document(page_content="alpha beta"),
        Document(page_content="epsilon zeta"),
        Document(page_content="eta theta"),
    ])

    reader = BM25Store(data_dir=str(tmp_path))
    assert reader.search("alpha")

    # Simulate another process (e.g. the MCP server) ingesting a new doc
    other = BM25Store(data_dir=str(tmp_path))
    other.add_documents([Document(page_content="gamma delta")])

    results = reader.search("gamma")
    assert results
    assert "gamma" in results[0][0].page_content


def test_search_does_not_reload_when_unchanged(tmp_path, monkeypatch):
    store = BM25Store(data_dir=str(tmp_path))
    store.add_documents([Document(page_content="alpha beta")])

    calls = {"n": 0}
    original = store._load

    def counting_load():
        calls["n"] += 1
        original()

    monkeypatch.setattr(store, "_load", counting_load)
    store.search("alpha")
    store.search("alpha")
    assert calls["n"] == 0


# Under the old lower().split() tokenizer, "autoregressive?" and "gpt-3:" never match the
# chunk's "autoregressive." and "GPT-3:", and the shorter BERT chunk wins on "is" alone.
_PUNCT_DOCS = [
    Document(page_content="BERT is bidirectional.", metadata={"source": "bert"}),
    Document(page_content="LoRA freezes the pretrained weights.", metadata={"source": "lora"}),
    Document(page_content="GPT-3: a large language model trained to be autoregressive.", metadata={"source": "gpt3"}),
]
_PUNCT_QUERY = "Is GPT-3 autoregressive?"


def test_punctuation_does_not_block_a_match(tmp_path):
    store = BM25Store(data_dir=str(tmp_path))
    store.add_documents(_PUNCT_DOCS)

    results = store.search(_PUNCT_QUERY, top_k=1)
    assert results and results[0][0].metadata["source"] == "gpt3"


def test_index_from_an_older_tokenizer_is_retokenized_on_load(tmp_path):
    import pickle

    from rank_bm25 import BM25Okapi

    from app.retrieval.bm25_store import TOKENIZER_VERSION

    old_corpus = [d.page_content.lower().split() for d in _PUNCT_DOCS]
    with open(tmp_path / "bm25_index.pkl", "wb") as f:  # version-1 pickle: no tokenizer_version key
        pickle.dump({"documents": _PUNCT_DOCS, "tokenized_corpus": old_corpus, "bm25": BM25Okapi(old_corpus)}, f)

    store = BM25Store(data_dir=str(tmp_path))

    assert store.search(_PUNCT_QUERY, top_k=1)[0][0].metadata["source"] == "gpt3"
    with open(tmp_path / "bm25_index.pkl", "rb") as f:
        assert pickle.load(f)["tokenizer_version"] == TOKENIZER_VERSION
