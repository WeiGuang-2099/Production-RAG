import sys
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.graph.builder import GraphBuilder
from app.graph.store import GraphStore


def test_llm_extractor():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='[{"head": "Python", "relation": "is_a", "tail": "programming language"}]'
    )

    builder = GraphBuilder(extractor_type="llm", llm=mock_llm)
    docs = [Document(page_content="Python is a programming language.")]
    triples = builder.extract(docs)
    assert len(triples) >= 1
    assert triples[0]["head"] == "Python"
    assert triples[0]["relation"] == "is_a"


def test_llm_extractor_skips_null_and_empty_fields():
    """LLMs sometimes emit triples with a null/blank head, relation, or tail.
    Those must be filtered out, not passed through — a None head/tail crashes
    the graph build with 'None cannot be a node'."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content=(
            "["
            '{"head": "Transformer", "relation": "uses", "tail": "attention"},'
            '{"head": null, "relation": "uses", "tail": "attention"},'
            '{"head": "X", "relation": "", "tail": "Y"},'
            '{"head": "A", "relation": "r", "tail": null}'
            "]"
        )
    )
    builder = GraphBuilder(extractor_type="llm", llm=mock_llm)
    triples = builder.extract([Document(page_content="Transformers use attention.")])
    assert triples == [{"head": "Transformer", "relation": "uses", "tail": "attention"}]


def test_graph_store_skips_incomplete_triples(tmp_path):
    """add_triples must defend against null/missing endpoints rather than raise."""
    store = GraphStore(data_dir=str(tmp_path))
    store.add_triples([
        {"head": "A", "relation": "r", "tail": "B"},
        {"head": None, "relation": "r", "tail": "B"},
        {"head": "C", "relation": "r"},
    ])
    assert store.graph.has_edge("A", "B")
    assert store.graph.number_of_edges() == 1


def test_nlp_extractor():
    mock_nlp = MagicMock()
    mock_ent1 = MagicMock(text="Python", label_="LANGUAGE")
    mock_ent2 = MagicMock(text="Google", label_="ORG")
    mock_sent = MagicMock(text="Python was created at Google.")
    mock_doc = MagicMock(ents=[mock_ent1, mock_ent2], sents=[mock_sent])
    mock_nlp.return_value = mock_doc

    mock_spacy = MagicMock()
    mock_spacy.load.return_value = mock_nlp

    with patch.dict(sys.modules, {"spacy": mock_spacy}):
        builder = GraphBuilder(extractor_type="nlp")
        docs = [Document(page_content="Python was created at Google.")]
        triples = builder.extract(docs)
        assert len(triples) >= 1


def test_graph_store_add_and_query(tmp_path):
    store = GraphStore(data_dir=str(tmp_path))
    store.add_triples([
        {"head": "A", "relation": "connects", "tail": "B"},
        {"head": "B", "relation": "connects", "tail": "C"},
        {"head": "X", "relation": "connects", "tail": "Y"},
    ])

    neighbors = store.get_neighbors("A", depth=2)
    names = [n for n, _ in neighbors]
    assert "B" in names
    assert "C" in names


def test_graph_store_persistence(tmp_path):
    store1 = GraphStore(data_dir=str(tmp_path))
    store1.add_triples([{"head": "A", "relation": "r", "tail": "B"}])

    store2 = GraphStore(data_dir=str(tmp_path))
    neighbors = store2.get_neighbors("A", depth=1)
    assert any(n == "B" for n, _ in neighbors)


def test_refresh_if_stale_picks_up_external_write(tmp_path):
    writer = GraphStore(data_dir=str(tmp_path))
    writer.add_triples([{"head": "A", "relation": "r", "tail": "B"}])

    reader = GraphStore(data_dir=str(tmp_path))
    assert "A" in reader.graph

    other = GraphStore(data_dir=str(tmp_path))
    other.add_triples([{"head": "C", "relation": "r", "tail": "D"}])

    reader.refresh_if_stale()
    assert "C" in reader.graph
