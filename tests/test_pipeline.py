import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


def test_ingest_pipeline():
    with patch("app.core.pipeline.get_embedder") as mock_emb, \
         patch("app.core.pipeline.VectorStore") as mock_vs_cls, \
         patch("app.core.pipeline.BM25Store") as mock_bm25_cls, \
         patch("app.core.pipeline.GraphBuilder") as mock_gb_cls, \
         patch("app.core.pipeline.GraphStore") as mock_gs_cls, \
         patch("app.core.pipeline.load_documents") as mock_load, \
         patch("app.core.pipeline.chunk_documents") as mock_chunk, \
         patch("app.core.pipeline.get_settings") as mock_s, \
         patch("app.core.pipeline.get_llm") as mock_llm_f:

        mock_s.return_value.GRAPH_EXTRACTOR = "llm"
        mock_s.return_value.DATA_DIR = "/tmp/test"
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
