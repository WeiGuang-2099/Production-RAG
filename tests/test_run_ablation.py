import os

import pytest
from langchain_core.documents import Document

import app.reranker.reranker as reranker_mod
import evaluation.run_ablation as ablation
from app.config import get_settings
from app.reranker.reranker import RerankerService


class _BillingCapCohere:
    def rerank(self, documents, query, top_n):
        raise RuntimeError("status_code: 402, Maximum billing reached for this API key")


_STAGE_KEYS = ("RETRIEVAL_MODE", "RERANKER_PROVIDER", "GRAPH_EXTRACTOR", "RERANK_TOP_K")


@pytest.fixture
def isolated(monkeypatch):
    """run_stage writes os.environ directly and main patches RerankerService; undo both."""
    saved = {k: os.environ.get(k) for k in _STAGE_KEYS}
    monkeypatch.setattr(reranker_mod.RerankerService, "rerank", reranker_mod.RerankerService.rerank)
    monkeypatch.setattr(ablation, "STAGES", [("+rerank", {"RETRIEVAL_MODE": "hybrid"})])
    yield
    for key, val in saved.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val
    get_settings.cache_clear()


def _use_retrieval(monkeypatch, reranker):
    import app.core.pipeline as pipeline

    def fake_retrieve_sources(question, top_k):
        docs = [Document(page_content="p", metadata={"source": "paper.pdf"})]
        try:
            docs = RerankerService(reranker=reranker).rerank(question, docs, top_k=top_k)
        except Exception:  # noqa: BLE001 - mirrors the pipeline's silent fallback
            pass
        return [{"content": d.page_content, "metadata": d.metadata} for d in docs]

    monkeypatch.setattr(pipeline, "retrieve_sources", fake_retrieve_sources)


def test_failed_reranks_are_reported_and_exit_2(isolated, monkeypatch, capsys):
    _use_retrieval(monkeypatch, _BillingCapCohere())

    assert ablation.main(["--subset", "2", "--no-save"]) == 2

    out = capsys.readouterr()
    assert "**Rerank failed in +rerank (2 of 2 questions)" in out.out
    assert "| rerank_failures |" in out.out
    assert "--allow-rerank-fallback" in out.err


def test_fallback_can_be_allowed(isolated, monkeypatch):
    _use_retrieval(monkeypatch, _BillingCapCohere())
    assert ablation.main(["--subset", "2", "--no-save", "--allow-rerank-fallback"]) == 0


def test_no_failures_exit_0_with_a_zero_column(isolated, monkeypatch, capsys):
    _use_retrieval(monkeypatch, None)  # RerankerService without a client just truncates

    assert ablation.main(["--subset", "2", "--no-save"]) == 0

    out = capsys.readouterr().out
    assert "Rerank failed" not in out
    assert out.rstrip().splitlines()[-1].endswith("| 0 |")


def test_counter_resets_per_stage(isolated, monkeypatch):
    _use_retrieval(monkeypatch, _BillingCapCohere())
    ablation.install_rerank_counter()
    dataset = ablation.load_dataset(None, 3)
    first = ablation.run_stage("+rerank", {}, dataset, 5, 10)
    second = ablation.run_stage("+graph", {}, dataset, 5, 10)
    assert first["rerank_failures"] == 3
    assert second["rerank_failures"] == 3


def test_counter_reraises_so_the_pipeline_fallback_still_runs(isolated):
    ablation.install_rerank_counter()
    with pytest.raises(RuntimeError, match="402"):
        RerankerService(reranker=_BillingCapCohere()).rerank("q", [Document(page_content="p")], top_k=1)


def test_only_the_failing_stage_is_named(isolated, monkeypatch, capsys):
    import app.core.pipeline as pipeline

    monkeypatch.setattr(ablation, "STAGES", [("+rerank", {"GRAPH_EXTRACTOR": "none"}),
                                             ("+graph", {"GRAPH_EXTRACTOR": "llm"})])

    def fake_retrieve_sources(question, top_k):
        reranker = _BillingCapCohere() if os.environ["GRAPH_EXTRACTOR"] == "llm" else None
        try:
            RerankerService(reranker=reranker).rerank(question, [Document(page_content="p")], top_k=top_k)
        except Exception:  # noqa: BLE001
            pass
        return []

    monkeypatch.setattr(pipeline, "retrieve_sources", fake_retrieve_sources)

    assert ablation.main(["--subset", "2", "--no-save"]) == 2

    [warning] = [line for line in capsys.readouterr().out.splitlines() if line.startswith("**Rerank failed")]
    assert warning == "**Rerank failed in +graph (2 of 2 questions); those questions kept the unranked order.**"
