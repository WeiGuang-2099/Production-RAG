from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.agent import nodes


def _llm(content):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=content)
    return llm


def test_route_question_uses_parsed_label():
    with patch("app.agent.nodes.get_llm", return_value=_llm("answer")):
        assert nodes.route_question({"question": "hi"}) == {"route": "answer"}


def test_route_question_defaults_to_retrieve_on_error():
    with patch("app.agent.nodes.get_llm", side_effect=RuntimeError("boom")):
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


def test_grade_documents_yes():
    doc = Document(page_content="d", metadata={"source": "a"})
    with patch("app.agent.nodes.get_llm", return_value=_llm("yes")):
        assert nodes.grade_documents({"question": "q", "documents": [doc]}) == {"relevant": True}


def test_grade_documents_no():
    doc = Document(page_content="d", metadata={"source": "a"})
    with patch("app.agent.nodes.get_llm", return_value=_llm("no")):
        assert nodes.grade_documents({"question": "q", "documents": [doc]}) == {"relevant": False}


def test_rewrite_query_increments_attempts():
    with patch("app.agent.nodes.get_llm", return_value=_llm("better query")):
        out = nodes.rewrite_query({"question": "q", "attempts": 0})
        assert out["query"] == "better query"
        assert out["attempts"] == 1


def test_generate_node_produces_cited_sources_and_usage():
    doc = Document(page_content="ctx", metadata={"source": "a.pdf"})
    with patch("app.agent.nodes.get_llm", return_value=_llm("the answer")), \
         patch("app.agent.nodes.get_settings") as mock_s:
        mock_s.return_value.PROMPT_MODE = "grounded"
        mock_s.return_value.LLM_MODEL = "gpt-4o"
        out = nodes.generate_node({"question": "q", "documents": [doc]})
        assert out["answer"] == "the answer"
        assert out["sources"][0]["metadata"]["citation"] == 1
        assert "cost_usd" in out["usage"]


def test_answer_directly_has_no_sources():
    with patch("app.agent.nodes.get_llm", return_value=_llm("general answer")), \
         patch("app.agent.nodes.get_settings") as mock_s:
        mock_s.return_value.LLM_MODEL = "gpt-4o"
        out = nodes.answer_directly({"question": "hi"})
        assert out["answer"] == "general answer"
        assert out["sources"] == []
