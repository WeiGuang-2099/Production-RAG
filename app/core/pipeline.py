"""Ingest and query pipelines with error handling, logging, and incremental ingestion."""
import asyncio
import hashlib
import json
import time
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from app.config import get_settings
from app.core.factories import get_llm, get_embedder, get_reranker
from app.core.prompts import select_prompt, format_context
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
from app.observability.cost import usage_for

logger = logging.getLogger(__name__)

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

def _retrieve_and_rerank(question: str, top_k: int, settings) -> list[Document]:
    """Retrieve (dense/hybrid + optional graph) and rerank to the final docs.

    Shared by query_pipeline (generation) and retrieve_sources (eval/UI), so
    both measure the exact same retrieval path.
    """
    # RETRIEVAL_MODE=dense is the ablation baseline (vector only); hybrid
    # additionally fuses BM25 keyword hits via RRF.
    hybrid_results: list[tuple[Document, float]] = []
    try:
        vs = VectorStore()
        if settings.RETRIEVAL_MODE == "dense":
            hybrid_results = vs.search(question, top_k=top_k)
            logger.info("dense_retrieval_complete: hits=%d", len(hybrid_results))
        else:
            bm25 = BM25Store(data_dir=settings.DATA_DIR)
            retriever = HybridRetriever(vector_store=vs, bm25_store=bm25)
            hybrid_results = retriever.retrieve(question, top_k=top_k)
            logger.info("hybrid_retrieval_complete: hits=%d", len(hybrid_results))
    except Exception as exc:
        logger.error("retrieval_failed: %s", exc)
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

    return reranked


def _docs_to_sources(docs: list[Document]) -> list[dict]:
    """Render reranked docs as source dicts, stamping the 1-based citation
    number so callers can resolve the [n] markers in a grounded answer."""
    sources = []
    for i, d in enumerate(docs, 1):
        md = dict(d.metadata)
        md["citation"] = i
        sources.append({"content": d.page_content, "metadata": md})
    return sources


def retrieve_sources(question: str, top_k: int | None = None) -> list[dict]:
    """Retrieval-only entry point (no generation): cited sources for a query.

    Used by the retrieval-only evaluation and the demo UI; avoids paying for
    a generation call when only the retrieved context is needed.
    """
    settings = get_settings()
    if top_k is None:
        top_k = settings.TOP_K
    start = time.time()
    reranked = _retrieve_and_rerank(question, top_k, settings)
    retrieval_ms = (time.time() - start) * 1000
    trace_data = [
        {"content": d.page_content[:100], "score": d.metadata.get("relevance_score", 0)}
        for d in reranked
    ]
    trace_retrieval(question, trace_data, retrieval_ms)
    return _docs_to_sources(reranked)


def query_pipeline(question: str, top_k: int | None = None) -> dict:
    settings = get_settings()
    if top_k is None:
        top_k = settings.TOP_K
    start = time.time()
    logger.info("query_start: question=%s", question[:100])

    reranked = _retrieve_and_rerank(question, top_k, settings)

    retrieval_ms = (time.time() - start) * 1000
    trace_data = [
        {"content": d.page_content[:100], "score": d.metadata.get("relevance_score", 0)}
        for d in reranked
    ]
    trace_retrieval(question, trace_data, retrieval_ms)

    # LLM generation. Context is numbered so the grounded prompt's [n]
    # citations map back to the sources returned below.
    context = format_context(reranked)
    prompt = ChatPromptTemplate.from_template(select_prompt(settings.PROMPT_MODE))
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

    prompt_text = select_prompt(settings.PROMPT_MODE).format(context=context, question=question)
    usage = usage_for(prompt_text, answer, str(settings.LLM_MODEL))
    logger.info(
        "query_usage: input_tokens=%d output_tokens=%d cost_usd=%.6f",
        usage["input_tokens"], usage["output_tokens"], usage["cost_usd"],
    )

    return {
        "answer": answer,
        "sources": _docs_to_sources(reranked),
        "latency_ms": total_ms,
        "usage": usage,
    }


async def stream_query(question: str, top_k: int | None = None) -> AsyncIterator[dict]:
    """Stream a grounded answer token-by-token.

    Retrieval is synchronous (and not streamable), so it runs in a thread and
    is emitted as a single ``sources`` event; generation is then streamed from
    the LLM one token at a time, finishing with a ``done`` event that carries
    the assembled answer and token/cost usage.
    """
    settings = get_settings()
    if top_k is None:
        top_k = settings.TOP_K
    start = time.time()

    reranked = await asyncio.to_thread(_retrieve_and_rerank, question, top_k, settings)
    retrieval_ms = (time.time() - start) * 1000
    trace_data = [
        {"content": d.page_content[:100], "score": d.metadata.get("relevance_score", 0)}
        for d in reranked
    ]
    trace_retrieval(question, trace_data, retrieval_ms)
    yield {"event": "sources", "sources": _docs_to_sources(reranked), "latency_ms": retrieval_ms}

    context = format_context(reranked)
    prompt_text = select_prompt(settings.PROMPT_MODE).format(context=context, question=question)
    llm = get_llm()
    parts: list[str] = []
    async for chunk in llm.astream(prompt_text):
        token = getattr(chunk, "content", "") or ""
        if token:
            parts.append(token)
            yield {"event": "token", "token": token}

    answer = "".join(parts)
    usage = usage_for(prompt_text, answer, str(settings.LLM_MODEL))
    total_ms = (time.time() - start) * 1000
    logger.info("stream_complete: latency_ms=%.1f cost_usd=%.6f", total_ms, usage["cost_usd"])
    yield {"event": "done", "answer": answer, "usage": usage, "latency_ms": total_ms}
