from langchain_core.documents import Document

from app.core.prompts import (
    BASIC_PROMPT,
    GROUNDED_PROMPT,
    STRICT_PROMPT,
    format_context,
    select_prompt,
)


def test_select_prompt_grounded():
    assert select_prompt("grounded") == GROUNDED_PROMPT


def test_select_prompt_basic():
    assert select_prompt("basic") == BASIC_PROMPT


def test_select_prompt_unknown_defaults_to_grounded():
    # Unknown/garbage modes must fail safe to the stricter grounded prompt.
    assert select_prompt("nonsense") == GROUNDED_PROMPT


def test_select_prompt_strict():
    assert select_prompt("strict") == STRICT_PROMPT


def test_all_prompts_have_required_variables():
    for tmpl in (BASIC_PROMPT, GROUNDED_PROMPT, STRICT_PROMPT):
        assert "{context}" in tmpl
        assert "{question}" in tmpl


def test_grounded_prompt_enforces_grounding_citation_and_refusal():
    """Guard test: the grounded prompt must keep its core contract so it
    cannot be silently weakened. These three behaviors are the whole point."""
    lowered = GROUNDED_PROMPT.lower()
    # grounding: answer strictly from context
    assert "only" in lowered
    # do-not-hallucinate instruction
    assert "do not" in lowered or "don't" in lowered
    # citation instruction
    assert "cite" in lowered
    # explicit refusal contract
    assert "cannot answer" in lowered


def test_format_context_numbers_sources_for_citation():
    docs = [
        Document(page_content="Alpha content", metadata={"source": "a.pdf"}),
        Document(page_content="Beta content", metadata={"source": "b.pdf"}),
    ]
    out = format_context(docs)
    assert "[1]" in out
    assert "[2]" in out
    assert "Alpha content" in out
    assert "Beta content" in out
    # the source label is surfaced so the model can attribute citations
    assert "a.pdf" in out
    # numbering order must follow doc order: [1] precedes [2]
    assert out.index("[1]") < out.index("[2]")


def test_format_context_empty():
    assert format_context([]) == ""


def test_format_context_handles_missing_source_metadata():
    docs = [Document(page_content="no source meta")]
    out = format_context(docs)
    assert "[1]" in out
    assert "no source meta" in out


def test_grounded_prompt_answers_partly_and_names_the_gap():
    # Over-refusal fix: a partial answer must be allowed, with what is missing stated.
    lowered = GROUNDED_PROMPT.lower()
    assert "only part of the question" in lowered
    assert "does not state" in lowered


def test_strict_prompt_keeps_the_all_or_nothing_refusal():
    # The pre-fix prompt stays selectable so before/after evals stay reproducible.
    lowered = STRICT_PROMPT.lower()
    assert "does not contain enough information" in lowered
    assert "cannot answer" in lowered

