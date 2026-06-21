from unittest.mock import patch

from langchain_core.documents import Document

from app.agent import nodes


def test_route_question_uses_fast_model():
    with patch("app.agent.nodes.complete", return_value="answer") as mock_c:
        assert nodes.route_question({"question": "hi"}) == {"route": "answer"}
        assert mock_c.call_args.kwargs.get("fast") is True


def test_route_question_defaults_to_retrieve_on_error():
    with patch("app.agent.nodes.complete", side_effect=RuntimeError("boom")):
        assert nodes.route_question({"question": "hi"}) == {"route": "retrieve"}


def test_retrieve_node_uses_current_query():
    doc = Document(page_content="d", metadata={"source": "a"})
    with patch("app.agent.nodes._retrieve_and_rerank", return_value=[doc]) as mock_rr, \
         patch("app.agent.nodes.get_settings") as mock_s:
        mock_s.return_value.TOP_K = 5
        out = nodes.retrieve_node({"question": "orig", "query": "rewritten", "top_k": 7})
        assert out["documents"] == [doc]
        assert mock_rr.call_args[0][0] == "rewritten"
        assert mock_rr.call_args[0][1] == 7


def test_grade_documents_no_docs_is_irrelevant():
    assert nodes.grade_documents({"question": "q", "documents": []}) == {"relevant": False}


def test_grade_documents_yes_uses_fast_model():
    doc = Document(page_content="d", metadata={"source": "a"})
    with patch("app.agent.nodes.complete", return_value="yes") as mock_c:
        assert nodes.grade_documents({"question": "q", "documents": [doc]}) == {"relevant": True}
        assert mock_c.call_args.kwargs.get("fast") is True


def test_grade_documents_no():
    doc = Document(page_content="d", metadata={"source": "a"})
    with patch("app.agent.nodes.complete", return_value="no"):
        assert nodes.grade_documents({"question": "q", "documents": [doc]}) == {"relevant": False}


def test_rewrite_query_uses_fast_model_and_increments_attempts():
    with patch("app.agent.nodes.complete", return_value="better query") as mock_c:
        out = nodes.rewrite_query({"question": "q", "attempts": 0})
        assert out["query"] == "better query"
        assert out["attempts"] == 1
        assert mock_c.call_args.kwargs.get("fast") is True


def test_generate_node_uses_strong_model_with_cited_sources_and_usage():
    doc = Document(page_content="ctx", metadata={"source": "a.pdf"})
    with patch("app.agent.nodes.complete_with_model", return_value=("the answer", "gpt-4o")) as mock_c, \
         patch("app.agent.nodes.get_settings") as mock_s:
        mock_s.return_value.PROMPT_MODE = "grounded"
        out = nodes.generate_node({"question": "q", "documents": [doc]})
        assert out["answer"] == "the answer"
        assert out["sources"][0]["metadata"]["citation"] == 1
        assert "cost_usd" in out["usage"]
        assert mock_c.call_args.kwargs.get("fast") in (False, None)


def test_generate_node_attributes_usage_to_model_that_answered():
    """On fallback the answer comes from the fallback model; usage must reflect it."""
    doc = Document(page_content="ctx", metadata={"source": "a.pdf"})
    with patch("app.agent.nodes.complete_with_model", return_value=("ans", "gpt-4o-mini")), \
         patch("app.agent.nodes.get_settings") as mock_s:
        mock_s.return_value.PROMPT_MODE = "grounded"
        out = nodes.generate_node({"question": "q", "documents": [doc]})
        assert out["usage"]["model"] == "gpt-4o-mini"


def test_answer_directly_has_no_sources():
    with patch("app.agent.nodes.complete_with_model", return_value=("general answer", "gpt-4o")):
        out = nodes.answer_directly({"question": "hi"})
        assert out["answer"] == "general answer"
        assert out["sources"] == []
