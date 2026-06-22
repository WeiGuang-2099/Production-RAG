"""Compatibility shims so the eval stack imports under langchain v1.

ragas 0.4.3 imports VertexAI wrappers from langchain-community paths that were
removed in the langchain v1 line (langchain-community 0.4.x)::

    from langchain_community.chat_models.vertexai import ChatVertexAI
    from langchain_community.llms import VertexAI

This project uses OpenAI, so those classes are never instantiated — but the
imports execute at ``import ragas`` time and would raise ModuleNotFoundError.
Rather than pin the entire langchain stack back to 0.3.x (which the app no
longer targets), we register minimal import-satisfying stubs.

Call :func:`ensure_ragas_importable` before importing ragas.
"""
from __future__ import annotations

import sys
import types


def ensure_ragas_importable() -> None:
    """Register stub modules/attrs for langchain-community APIs ragas expects."""
    _stub_chat_models_vertexai()
    _stub_llms_vertexai()


def _stub_chat_models_vertexai() -> None:
    name = "langchain_community.chat_models.vertexai"
    try:
        __import__(name)
        return  # still present (older langchain-community) — nothing to do
    except ModuleNotFoundError:
        pass

    module = types.ModuleType(name)

    class ChatVertexAI:  # import-satisfying stub; never instantiated (OpenAI is used)
        """Stub for the removed langchain_community ChatVertexAI."""

    module.ChatVertexAI = ChatVertexAI
    sys.modules[name] = module

    import langchain_community.chat_models as parent
    parent.vertexai = module  # so `from ...chat_models.vertexai import X` resolves


def _stub_llms_vertexai() -> None:
    try:
        from langchain_community.llms import VertexAI  # noqa: F401
        return  # still present — nothing to do
    except Exception:
        pass

    import langchain_community.llms as llms

    class VertexAI:  # import-satisfying stub; never instantiated (OpenAI is used)
        """Stub for the removed langchain_community VertexAI."""

    llms.VertexAI = VertexAI
