from unittest.mock import MagicMock

from app.retrieval.query_transform import build_queries, parse_multi_queries


def test_parse_multi_queries_strips_numbering_and_bullets():
    text = "1. first?\n2) second?\n- third?\n\n"
    assert parse_multi_queries(text) == ["first?", "second?", "third?"]


def test_parse_multi_queries_ignores_blank_lines():
    assert parse_multi_queries("\n\n  \n") == []


def test_build_queries_none_returns_original_without_calling_llm():
    llm = MagicMock()
    assert build_queries("q", "none", llm) == ["q"]
    llm.invoke.assert_not_called()


def test_build_queries_unknown_mode_is_passthrough():
    llm = MagicMock()
    assert build_queries("q", "something", llm) == ["q"]
    llm.invoke.assert_not_called()


def test_build_queries_multi_query_keeps_original_first():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="1. paraphrase one\n2. paraphrase two")
    qs = build_queries("orig", "multi_query", llm)
    assert qs[0] == "orig"
    assert "paraphrase one" in qs
    assert "paraphrase two" in qs


def test_build_queries_multi_query_dedups_and_caps():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="orig\norig\nq2\nq3\nq4\nq5\nq6")
    qs = build_queries("orig", "multi_query", llm)
    assert qs[0] == "orig"
    assert len(qs) == len(set(qs))   # no duplicates
    assert len(qs) <= 5               # capped


def test_build_queries_hyde_returns_hypothetical_document():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="A hypothetical passage answering the question.")
    qs = build_queries("q", "hyde", llm)
    assert qs == ["A hypothetical passage answering the question."]
