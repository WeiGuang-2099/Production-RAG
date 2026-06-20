import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


def test_ingest_pipeline(tmp_path):
    with patch("app.core.pipeline.get_embedder") as mock_emb, \
         patch("app.core.pipeline.VectorStore") as mock_vs_cls, \
         patch("app.core.pipeline.BM25Store") as mock_bm25_cls, \
         patch("app.core.pipeline.GraphBuilder") as mock_gb_cls, \
         patch("app.core.pipeline.GraphStore") as mock_gs_cls, \
         patch("app.core.pipeline.load_documents") as mock_load, \
         patch("app.core.pipeline.chunk_documents") as mock_chunk, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.get_llm") as mock_llm_f, \
         patch("app.core.pipeline._load_tracking", return_value={}), \
         patch("app.core.pipeline._save_tracking"):

        mock_s.return_value.GRAPH_EXTRACTOR = "llm"
        mock_s.return_value.DATA_DIR = str(tmp_path)
        mock_s.return_value.CHUNK_SIZE = 512
        mock_s.return_value.CHUNK_OVERLAP = 64

        mock_load.return_value = [Document(page_content="test content")]
        mock_chunk.return_value = [Document(page_content="chunked content")]

        mock_vs = MagicMock()
        mock_vs_cls.return_value = mock_vs
        mock_vs.upsert.return_value = ["id1"]

        mock_bm25 = MagicMock()
        mock_bm25_cls.return_value = mock_bm25

        mock_llm = MagicMock()
        mock_llm_f.return_value = mock_llm

        mock_gb = MagicMock()
        mock_gb_cls.return_value = mock_gb
        mock_gb.extract.return_value = [{"head": "A", "relation": "r", "tail": "B"}]

        mock_gs = MagicMock()
        mock_gs_cls.return_value = mock_gs

        from app.core.pipeline import ingest_pipeline
        result = ingest_pipeline("test.md")
        assert result["chunks"] == 1
        assert result["status"] == "ingested"


def test_query_pipeline():
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MagicMock(content="Generated answer")

    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)

    with patch("app.core.pipeline.get_llm") as mock_llm_f, \
         patch("app.core.pipeline.get_embedder") as mock_emb, \
         patch("app.core.pipeline.get_reranker") as mock_rr_f, \
         patch("app.core.pipeline.VectorStore") as mock_vs_cls, \
         patch("app.core.pipeline.BM25Store") as mock_bm25_cls, \
         patch("app.core.pipeline.HybridRetriever") as mock_hr_cls, \
         patch("app.core.pipeline.GraphRetriever") as mock_gr_cls, \
         patch("app.core.pipeline.GraphStore") as mock_gs_cls, \
         patch("app.core.pipeline.RerankerService") as mock_rs_cls, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval") as mock_trace, \
         patch("app.core.pipeline.ChatPromptTemplate") as mock_cpt:

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3
        mock_s.return_value.DATA_DIR = "/tmp/test"
        mock_s.return_value.GRAPH_EXTRACTOR = "none"

        mock_llm = MagicMock()
        mock_llm_f.return_value = mock_llm

        mock_cpt.from_template.return_value = mock_prompt

        mock_rr = MagicMock()
        mock_rr_f.return_value = mock_rr

        doc = Document(page_content="context doc")
        mock_hr = MagicMock()
        mock_hr_cls.return_value = mock_hr
        mock_hr.retrieve.return_value = [(doc, 0.9)]

        mock_rs = MagicMock()
        mock_rs_cls.return_value = mock_rs
        mock_rs.rerank.return_value = [doc]

        from app.core.pipeline import query_pipeline
        result = query_pipeline("What is AI?")
        assert result["answer"] == "Generated answer"
        assert len(result["sources"]) > 0


def test_query_pipeline_returns_full_source_content():
    """Sources must carry full chunk content; truncation corrupts RAGAS evaluation."""
    long_content = "x" * 600

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MagicMock(content="Generated answer")
    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)

    with patch("app.core.pipeline.get_llm"), \
         patch("app.core.pipeline.get_reranker"), \
         patch("app.core.pipeline.VectorStore"), \
         patch("app.core.pipeline.BM25Store"), \
         patch("app.core.pipeline.HybridRetriever") as mock_hr_cls, \
         patch("app.core.pipeline.RerankerService") as mock_rs_cls, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"), \
         patch("app.core.pipeline.ChatPromptTemplate") as mock_cpt:

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3
        mock_s.return_value.DATA_DIR = "/tmp/test"
        mock_s.return_value.GRAPH_EXTRACTOR = "none"
        mock_cpt.from_template.return_value = mock_prompt

        doc = Document(page_content=long_content)
        mock_hr_cls.return_value.retrieve.return_value = [(doc, 0.9)]
        mock_rs_cls.return_value.rerank.return_value = [doc]

        from app.core.pipeline import query_pipeline
        result = query_pipeline("What is AI?")
        assert result["sources"][0]["content"] == long_content


def test_query_pipeline_respects_top_k_argument():
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MagicMock(content="Generated answer")
    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)

    with patch("app.core.pipeline.get_llm"), \
         patch("app.core.pipeline.get_reranker"), \
         patch("app.core.pipeline.VectorStore"), \
         patch("app.core.pipeline.BM25Store"), \
         patch("app.core.pipeline.HybridRetriever") as mock_hr_cls, \
         patch("app.core.pipeline.RerankerService") as mock_rs_cls, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"), \
         patch("app.core.pipeline.ChatPromptTemplate") as mock_cpt:

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3
        mock_s.return_value.DATA_DIR = "/tmp/test"
        mock_s.return_value.GRAPH_EXTRACTOR = "none"
        mock_cpt.from_template.return_value = mock_prompt

        doc = Document(page_content="context doc")
        mock_retriever = mock_hr_cls.return_value
        mock_retriever.retrieve.return_value = [(doc, 0.9)]
        mock_rs_cls.return_value.rerank.return_value = [doc]

        from app.core.pipeline import query_pipeline
        query_pipeline("What is AI?", top_k=12)
        mock_retriever.retrieve.assert_called_once_with("What is AI?", top_k=12)


def test_query_pipeline_passes_numbered_context_to_llm():
    """Grounded generation needs the context numbered so [n] citations resolve."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MagicMock(content="answer [1]")
    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)

    with patch("app.core.pipeline.get_llm"), \
         patch("app.core.pipeline.get_reranker"), \
         patch("app.core.pipeline.VectorStore"), \
         patch("app.core.pipeline.BM25Store"), \
         patch("app.core.pipeline.HybridRetriever") as mock_hr_cls, \
         patch("app.core.pipeline.RerankerService") as mock_rs_cls, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"), \
         patch("app.core.pipeline.ChatPromptTemplate") as mock_cpt:

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3
        mock_s.return_value.DATA_DIR = "/tmp/test"
        mock_s.return_value.GRAPH_EXTRACTOR = "none"
        mock_s.return_value.PROMPT_MODE = "grounded"
        mock_cpt.from_template.return_value = mock_prompt

        doc = Document(page_content="Alpha content", metadata={"source": "a.pdf"})
        mock_hr_cls.return_value.retrieve.return_value = [(doc, 0.9)]
        mock_rs_cls.return_value.rerank.return_value = [doc]

        from app.core.pipeline import query_pipeline
        query_pipeline("What is AI?")

        passed = mock_chain.invoke.call_args[0][0]
        assert "[1]" in passed["context"]
        assert "Alpha content" in passed["context"]


def test_query_pipeline_stamps_citation_numbers_on_sources():
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MagicMock(content="answer")
    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)

    with patch("app.core.pipeline.get_llm"), \
         patch("app.core.pipeline.get_reranker"), \
         patch("app.core.pipeline.VectorStore"), \
         patch("app.core.pipeline.BM25Store"), \
         patch("app.core.pipeline.HybridRetriever") as mock_hr_cls, \
         patch("app.core.pipeline.RerankerService") as mock_rs_cls, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"), \
         patch("app.core.pipeline.ChatPromptTemplate") as mock_cpt:

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3
        mock_s.return_value.DATA_DIR = "/tmp/test"
        mock_s.return_value.GRAPH_EXTRACTOR = "none"
        mock_s.return_value.PROMPT_MODE = "grounded"
        mock_cpt.from_template.return_value = mock_prompt

        doc1 = Document(page_content="first", metadata={"source": "a.pdf"})
        doc2 = Document(page_content="second", metadata={"source": "b.pdf"})
        mock_hr_cls.return_value.retrieve.return_value = [(doc1, 0.9), (doc2, 0.8)]
        mock_rs_cls.return_value.rerank.return_value = [doc1, doc2]

        from app.core.pipeline import query_pipeline
        result = query_pipeline("What is AI?")

        assert result["sources"][0]["metadata"]["citation"] == 1
        assert result["sources"][1]["metadata"]["citation"] == 2


def test_query_pipeline_dense_mode_uses_vector_only():
    """RETRIEVAL_MODE=dense is the ablation baseline: vector search only,
    no BM25 store and no RRF fusion."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MagicMock(content="answer")
    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)

    with patch("app.core.pipeline.get_llm"), \
         patch("app.core.pipeline.get_reranker"), \
         patch("app.core.pipeline.VectorStore") as mock_vs_cls, \
         patch("app.core.pipeline.BM25Store") as mock_bm25_cls, \
         patch("app.core.pipeline.HybridRetriever") as mock_hr_cls, \
         patch("app.core.pipeline.RerankerService") as mock_rs_cls, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"), \
         patch("app.core.pipeline.ChatPromptTemplate") as mock_cpt:

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3
        mock_s.return_value.DATA_DIR = "/tmp/test"
        mock_s.return_value.GRAPH_EXTRACTOR = "none"
        mock_s.return_value.PROMPT_MODE = "grounded"
        mock_s.return_value.RETRIEVAL_MODE = "dense"
        mock_cpt.from_template.return_value = mock_prompt

        doc = Document(page_content="context doc", metadata={"source": "a.pdf"})
        mock_vs = mock_vs_cls.return_value
        mock_vs.search.return_value = [(doc, 0.9)]
        mock_rs_cls.return_value.rerank.return_value = [doc]

        from app.core.pipeline import query_pipeline
        query_pipeline("What is AI?", top_k=7)

        mock_vs.search.assert_called_once_with("What is AI?", top_k=7)
        mock_hr_cls.return_value.retrieve.assert_not_called()
        mock_bm25_cls.assert_not_called()


def test_retrieve_sources_returns_citations_without_generation():
    """retrieve_sources powers the cheap retrieval-only eval and the UI:
    it must return cited sources without ever calling the generation LLM."""
    with patch("app.core.pipeline.get_llm") as mock_llm_f, \
         patch("app.core.pipeline.get_reranker"), \
         patch("app.core.pipeline.VectorStore"), \
         patch("app.core.pipeline.BM25Store"), \
         patch("app.core.pipeline.HybridRetriever") as mock_hr_cls, \
         patch("app.core.pipeline.RerankerService") as mock_rs_cls, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"):

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3
        mock_s.return_value.DATA_DIR = "/tmp/test"
        mock_s.return_value.GRAPH_EXTRACTOR = "none"
        mock_s.return_value.RETRIEVAL_MODE = "hybrid"

        doc = Document(page_content="ctx", metadata={"source": "a.pdf"})
        mock_hr_cls.return_value.retrieve.return_value = [(doc, 0.9)]
        mock_rs_cls.return_value.rerank.return_value = [doc]

        from app.core.pipeline import retrieve_sources
        sources = retrieve_sources("q")

        assert sources[0]["content"] == "ctx"
        assert sources[0]["metadata"]["citation"] == 1
        mock_llm_f.assert_not_called()


def test_query_pipeline_reports_token_usage_and_cost():
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MagicMock(content="Generated answer")
    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)

    with patch("app.core.pipeline.get_llm"), \
         patch("app.core.pipeline.get_reranker"), \
         patch("app.core.pipeline.VectorStore"), \
         patch("app.core.pipeline.BM25Store"), \
         patch("app.core.pipeline.HybridRetriever") as mock_hr_cls, \
         patch("app.core.pipeline.RerankerService") as mock_rs_cls, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"), \
         patch("app.core.pipeline.ChatPromptTemplate") as mock_cpt:

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3
        mock_s.return_value.DATA_DIR = "/tmp/test"
        mock_s.return_value.GRAPH_EXTRACTOR = "none"
        mock_s.return_value.RETRIEVAL_MODE = "hybrid"
        mock_s.return_value.PROMPT_MODE = "grounded"
        mock_s.return_value.LLM_MODEL = "gpt-4o"
        mock_cpt.from_template.return_value = mock_prompt

        doc = Document(page_content="context doc", metadata={"source": "a.pdf"})
        mock_hr_cls.return_value.retrieve.return_value = [(doc, 0.9)]
        mock_rs_cls.return_value.rerank.return_value = [doc]

        from app.core.pipeline import query_pipeline
        result = query_pipeline("What is AI?")

        assert "usage" in result
        assert result["usage"]["model"] == "gpt-4o"
        assert result["usage"]["output_tokens"] > 0
        assert result["usage"]["cost_usd"] >= 0


async def test_stream_query_streams_tokens_then_done():
    """Real streaming: emit sources first, then one event per token, then a
    terminal 'done' with the assembled answer and usage."""
    class Chunk:
        def __init__(self, content):
            self.content = content

    async def fake_astream(_text):
        for tok in ["Hello", " world"]:
            yield Chunk(tok)

    doc = Document(page_content="ctx", metadata={"source": "a.pdf"})
    with patch("app.core.pipeline._retrieve_and_rerank", return_value=[doc]), \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.trace_retrieval"), \
         patch("app.core.pipeline.get_llm") as mock_llm_f:

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3
        mock_s.return_value.PROMPT_MODE = "grounded"
        mock_s.return_value.LLM_MODEL = "gpt-4o"

        mock_llm = MagicMock()
        mock_llm.astream = fake_astream
        mock_llm_f.return_value = mock_llm

        from app.core.pipeline import stream_query
        events = [e async for e in stream_query("q")]

    kinds = [e["event"] for e in events]
    assert kinds[0] == "sources"
    assert kinds.count("token") == 2
    assert kinds[-1] == "done"
    assert events[-1]["answer"] == "Hello world"
    assert events[-1]["usage"]["model"] == "gpt-4o"
