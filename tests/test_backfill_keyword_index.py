import pickle
from unittest.mock import patch

from langchain_core.documents import Document

from evaluation.backfill_keyword_index import backfill


def _write_pickle(tmp_path, n_docs):
    docs = [Document(page_content=f"doc {i}") for i in range(n_docs)]
    with open(tmp_path / "bm25_index.pkl", "wb") as f:
        pickle.dump({"documents": docs, "tokenized_corpus": [], "bm25": None}, f)
    return docs


def test_backfill_batches_documents(tmp_path):
    _write_pickle(tmp_path, 5)
    with patch("evaluation.backfill_keyword_index.OpenSearchStore") as mock_cls:
        store = mock_cls.return_value
        rc = backfill(str(tmp_path), "http://x:9200", "idx", recreate=False, batch_size=2)

    assert rc == 0
    mock_cls.assert_called_once_with(url="http://x:9200", index_name="idx")
    assert store.add_documents.call_count == 3  # 2 + 2 + 1
    sizes = [len(c.args[0]) for c in store.add_documents.call_args_list]
    assert sizes == [2, 2, 1]


def test_backfill_recreate_deletes_existing_index(tmp_path):
    _write_pickle(tmp_path, 1)
    with patch("evaluation.backfill_keyword_index.OpenSearchStore") as mock_cls:
        store = mock_cls.return_value
        store._client.indices.exists.return_value = True
        rc = backfill(str(tmp_path), "http://x:9200", "idx", recreate=True, batch_size=500)

    assert rc == 0
    store._client.indices.delete.assert_called_once_with(index="idx")


def test_backfill_missing_pickle_exits_nonzero(tmp_path):
    with patch("evaluation.backfill_keyword_index.OpenSearchStore") as mock_cls:
        rc = backfill(str(tmp_path), "http://x:9200", "idx", recreate=False, batch_size=500)
    assert rc == 1
    mock_cls.assert_not_called()
