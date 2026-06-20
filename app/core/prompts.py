"""RAG answer-generation prompts.

Two modes:

- ``basic``    — the original "answer from context" prompt. Kept so the
                 evaluation harness can A/B it against ``grounded`` and show
                 the quality delta.
- ``grounded`` — strictly grounds the answer in the retrieved context, asks
                 for inline ``[n]`` citations, and refuses when the answer is
                 not present. This is what a production RAG system should do;
                 it is the default.
"""
from __future__ import annotations

from langchain_core.documents import Document

BASIC_PROMPT = """Answer the question based on the following context.

Context:
{context}

Question: {question}

Answer:"""


GROUNDED_PROMPT = """You are a careful research assistant. Answer the question using ONLY the numbered context below.

Rules:
- Use only information stated in the context. Do not use prior knowledge and do not make up facts.
- After each claim, cite the supporting source(s) by their bracket number, e.g. [1] or [2][3].
- If the context does not contain enough information to answer, reply exactly: "I cannot answer this from the provided documents." Do not guess.

Context:
{context}

Question: {question}

Answer:"""


_PROMPTS = {"basic": BASIC_PROMPT, "grounded": GROUNDED_PROMPT}


def select_prompt(mode: str) -> str:
    """Return the prompt template for ``mode``.

    Unknown modes fail safe to the stricter grounded prompt rather than
    silently falling back to ungrounded generation.
    """
    return _PROMPTS.get(mode, GROUNDED_PROMPT)


def format_context(documents: list[Document]) -> str:
    """Render retrieved chunks as a numbered, source-labelled context block.

    The numbering is what makes inline ``[n]`` citations meaningful: source
    ``i`` in the returned answer maps to ``documents[i-1]``.
    """
    blocks: list[str] = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(blocks)
