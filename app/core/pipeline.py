"""Ingest and query pipelines with error handling, logging, and incremental ingestion."""
import hashlib
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from app.config import get_settings
from app.core.factories import get_llm, get_embedder, get_reranker
from app.ingestion.loaders import load_documents
from app.ingestion.chunkers import chunk_documents
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.graph_retriever import GraphRetriever
from app.graph.builder import GraphBuilder
from app.graph.store import GraphStore
from app.reranker.reranker import RerankerService
from app.observability.tracing import trace_retrieval

logger = logging.getLogger(__name__)

RAG_PROMPT = """Answer the question based on the following context.

Context:
{context}

Question: {question}

Answer:"""

# ── Ingestion tracking helpers ──────────────────────────

def _load_tracking(data_dir: str) -> dict:
    path = Path(data_dir) / "ingestions.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_tracking(data_dir: str, tracking: dict) -> None:
    path = Path(data_dir) / "ingestions.json"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(tracking, f, indent=2)


# ── Ingest Pipeline ─────────────────────────────────────

def ingest_pipeline(source: str, force: bool = False) -> dict:
    settings = get_settings()
    start = time.time()
    logger.info("ingest_start: source=%s", source)

    # Incremental ingestion: check if already ingested
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    tracking = _load_tracking(settings.DATA_DIR)
    if source_hash in tracking and not force:
        logger.info("ingest_skipped: source=%s (already ingested)", source)
        return {
            "source": source,
            "chunks": tracking[source_hash]["chunks"],
            "status": "skipped",
        }

    # Load documents
    try:
        documents = load_documents(source)
        logger.info("documents_loaded: count=%d source=%s", len(documents), source)
    except Exception as exc:
        logger.error("document_load_failed: source=%s error=%s", source, exc)
        raise

    # Chunk documents
    chunks = chunk_documents(documents, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    logger.info("chunks_created: count=%d", len(chunks))

    # Vector store upsert
    try:
        vs = VectorStore()
        vs.upsert(chunks)
        logger.info("vector_upserted: count=%d", len(chunks))
    except Exception as exc:
        logger.error("vector_upsert_failed: %s", exc)
        raise

    # BM25 index
    try:
        bm25 = BM25Store(data_dir=settings.DATA_DIR)
        bm25.add_documents(chunks)
        logger.info("bm25_indexed: count=%d", len(chunks))
    except Exception as exc:
        logger.error("bm25_index_failed: %s", exc)
        # Non-critical: continue without BM25

    # Graph extraction (optional)
    if settings.GRAPH_EXTRACTOR != "none":
        try:
            llm = get_llm() if settings.GRAPH_EXTRACTOR == "llm" else None
            builder = GraphBuilder(extractor_type=settings.GRAPH_EXTRACTOR, llm=llm)
            triples = builder.extract(chunks)
            gs = GraphStore(data_dir=settings.DATA_DIR)
            gs.add_triples(triples)
            logger.info("graph_extracted: triples=%d", len(triples))
        except Exception as exc:
            logger.error("graph_extraction_failed: %s", exc)
            # Non-critical: continue without graph

    # Record ingestion
    elapsed = (time.time() - start) * 1000
    tracking[source_hash] = {
        "source": source,
        "chunks": len(chunks),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_tracking(settings.DATA_DIR, tracking)
    logger.info("ingest_complete: source=%s chunks=%d latency_ms=%.1f", source, len(chunks), elapsed)

    return {"source": source, "chunks": len(chunks), "status": "ingested"}


# ── Query Pipeline ───────────────────────────────────────

def query_pipeline(question: str, top_k: int | None = None) -> dict:
    settings = get_settings()
    if top_k is None:
        top_k = settings.TOP_K
    start = time.time()
    logger.info("query_start: question=%s", question[:100])

    # Hybrid retrieval
    hybrid_results: list[tuple[Document, float]] = []
    try:
        vs = VectorStore()
        bm25 = BM25Store(data_dir=settings.DATA_DIR)
        retriever = HybridRetriever(vector_store=vs, bm25_store=bm25)
        hybrid_results = retriever.retrieve(question, top_k=top_k)
        logger.info("hybrid_retrieval_complete: hits=%d", len(hybrid_results))
    except Exception as exc:
        logger.error("hybrid_retrieval_failed: %s", exc)
        # Continue with empty results

    # Graph retrieval (optional)
    graph_docs: list[Document] = []
    if settings.GRAPH_EXTRACTOR != "none":
        try:
            gs = GraphStore(data_dir=settings.DATA_DIR)
            gr = GraphRetriever(graph_store=gs)
            graph_docs = gr.retrieve(question, depth=1)
            logger.info("graph_retrieval_complete: hits=%d", len(graph_docs))
        except Exception as exc:
            logger.error("graph_retrieval_failed: %s", exc)
            # Continue without graph results

    all_docs = [doc for doc, _ in hybrid_results] + graph_docs

    # Reranking
    try:
        reranker_model = get_reranker()
        reranker_svc = RerankerService(reranker=reranker_model)
        reranked = reranker_svc.rerank(question, all_docs, top_k=settings.RERANK_TOP_K)
        logger.info("rerank_complete: input=%d output=%d", len(all_docs), len(reranked))
    except Exception as exc:
        logger.warning("rerank_failed: %s — using unranked results", exc)
        reranked = all_docs[: settings.RERANK_TOP_K]

    retrieval_ms = (time.time() - start) * 1000
    trace_data = [
        {"content": d.page_content[:100], "score": d.metadata.get("relevance_score", 0)}
        for d in reranked
    ]
    trace_retrieval(question, trace_data, retrieval_ms)

    # LLM generation
    context = "\n\n".join(d.page_content for d in reranked)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    try:
        llm = get_llm()
        chain = prompt | llm
        response = chain.invoke({"context": context, "question": question})
        answer = response.content
        logger.info("llm_complete: latency_ms=%.1f", (time.time() - start) * 1000 - retrieval_ms)
    except Exception as exc:
        logger.error("llm_generation_failed: %s", exc)
        raise

    total_ms = (time.time() - start) * 1000
    logger.info("query_complete: latency_ms=%.1f", total_ms)

    return {
        "answer": answer,
        "sources": [{"content": d.page_content, "metadata": d.metadata} for d in reranked],
        "latency_ms": total_ms,
    }
