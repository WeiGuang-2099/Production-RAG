from typing import TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    question: str            # original question, never mutated
    query: str               # current search query; rewritten by `rewrite`
    top_k: int
    scope_sources: list[str]  # optional retrieval scope: restrict to these source paths
    route: str               # retrieve | answer | clarify
    documents: list[Document]
    relevant: bool
    attempts: int
    answer: str
    sources: list[dict]
    usage: dict
