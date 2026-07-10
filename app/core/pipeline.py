"""Ingest and query pipelines with error handling, logging, and incremental ingestion."""
import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document

from app.config import get_settings
from app.core.cache import QueryCache
from app.core.factories import (
    complete_with_model,
    get_embedder,
    get_graph_store,
    get_keyword_store,
    get_llm,
    get_reranker,
    get_vector_store,
)
from app.core.prompts import format_context, select_prompt
from app.graph.builder import GraphBuilder
from app.ingestion.chunkers import chunk_documents
from app.ingestion.loaders import load_documents
from app.observability.cost import usage_for
from app.observability.tracing import trace_retrieval
from app.reranker.reranker import RerankerService
from app.retrieval.graph_retriever import GraphRetriever
from app.retrieval.hybrid_retriever import HybridRetriever, rrf_fuse

logger = logging.getLogger(__name__)

# Outer retrieval pool: per-query hybrid/dense tasks + the graph leg. The
# hybrid legs themselves run on hybrid_retriever._leg_executor — a separate
# pool, so an outer task waiting on legs can never deadlock this one.
_retrieval_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="retrieval")

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


def list_documents() -> list[dict]:
    tracking = _load_tracking(get_settings().DATA_DIR)
    return [
        {"id": k, "source": v["source"], "chunks": v["chunks"], "ingested_at": v.get("ingested_at", "")}
        for k, v in tracking.items()
    ]


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
        vs = get_vector_store()
        vs.upsert(chunks)
        logger.info("vector_upserted: count=%d", len(chunks))
    except Exception as exc:
        logger.error("vector_upsert_failed: %s", exc)
        raise

    # Keyword index
    try:
        keyword_store = get_keyword_store()
        keyword_store.add_documents(chunks)
        logger.info("keyword_indexed: backend=%s count=%d", settings.KEYWORD_BACKEND, len(chunks))
    except Exception as exc:
        logger.error("keyword_index_failed: %s", exc)
        # Non-critical: continue without the keyword index

    # Graph extraction (optional)
    if settings.GRAPH_EXTRACTOR != "none":
        try:
            llm = get_llm() if settings.GRAPH_EXTRACTOR == "llm" else None
            builder = GraphBuilder(extractor_type=settings.GRAPH_EXTRACTOR, llm=llm)
            triples = builder.extract(chunks)
            gs = get_graph_store()
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

def _queries_for(question: str, settings) -> list[str]:
    """Expand the question per QUERY_TRANSFORM. Only calls the LLM when an
    actual transform (multi_query / hyde) is configured."""
    mode = getattr(settings, "QUERY_TRANSFORM", "none")
    if mode not in ("multi_query", "hyde"):
        return [question]
    try:
        from app.retrieval.query_transform import build_queries
        return build_queries(question, mode, get_llm())
    except Exception as exc:
        logger.warning("query_transform_failed: %s — using original query", exc)
        return [question]


def _graph_retrieve(question: str) -> list[Document]:
    gr = GraphRetriever(graph_store=get_graph_store())
    return gr.retrieve(question, depth=1)


def _retrieve_and_rerank(
    question: str, top_k: int, settings, sources: list[str] | None = None
) -> list[Document]:
    """Retrieve (dense/hybrid + optional graph) and rerank to the final docs.

    Shared by query_pipeline (generation) and retrieve_sources (eval/UI), so
    both measure the exact same retrieval path.
    """
    # RETRIEVAL_MODE=dense is the ablation baseline (vector only); hybrid
    # additionally fuses BM25 keyword hits via RRF. With QUERY_TRANSFORM,
    # we retrieve for each expanded query and RRF-fuse across them too.
    queries = _queries_for(question, settings)

    # Graph leg is submitted first so it overlaps with the (slower) vector
    # legs; skipped when scoped — graph docs carry no per-source payload to
    # filter on.
    graph_future = None
    if settings.GRAPH_EXTRACTOR != "none" and not sources:
        graph_future = _retrieval_executor.submit(_graph_retrieve, question)

    hybrid_results: list[tuple[Document, float]] = []
    try:
        vs = get_vector_store()
        if settings.RETRIEVAL_MODE == "dense":
            futures = [
                _retrieval_executor.submit(vs.search, q, top_k=top_k, sources=sources)
                for q in queries
            ]
        else:
            retriever = HybridRetriever(vector_store=vs, bm25_store=get_keyword_store())
            futures = [
                _retrieval_executor.submit(retriever.retrieve, q, top_k=top_k, sources=sources)
                for q in queries
            ]
        result_lists: list[list[tuple[Document, float]]] = []
        for q, future in zip(queries, futures):
            try:
                result_lists.append(future.result())
            except Exception as exc:  # noqa: BLE001 — one bad query must not kill the rest
                logger.error("retrieval_failed: query=%s error=%s", q[:80], exc)
                result_lists.append([])
        hybrid_results = (
            result_lists[0] if len(result_lists) == 1 else rrf_fuse(result_lists)[:top_k]
        )
        logger.info(
            "retrieval_complete: mode=%s queries=%d hits=%d",
            settings.RETRIEVAL_MODE, len(queries), len(hybrid_results),
        )
    except Exception as exc:
        logger.error("retrieval_failed: %s", exc)
        # Continue with empty results

    graph_docs: list[Document] = []
    if graph_future is not None:
        try:
            graph_docs = graph_future.result()
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


def retrieve_sources(
    question: str, top_k: int | None = None, sources: list[str] | None = None
) -> list[dict]:
    """Retrieval-only entry point (no generation): cited sources for a query.

    Used by the retrieval-only evaluation and the demo UI; avoids paying for
    a generation call when only the retrieved context is needed.
    """
    settings = get_settings()
    if top_k is None:
        top_k = settings.TOP_K
    start = time.time()
    reranked = _retrieve_and_rerank(question, top_k, settings, sources=sources)
    retrieval_ms = (time.time() - start) * 1000
    trace_data = [
        {"content": d.page_content[:100], "score": d.metadata.get("relevance_score", 0)}
        for d in reranked
    ]
    trace_retrieval(question, trace_data, retrieval_ms)
    return _docs_to_sources(reranked)


_query_cache: QueryCache | None = None


def _make_cache_backend(settings):
    """Redis when REDIS_URL is set and reachable; in-memory otherwise."""
    from app.core.cache import InMemoryBackend, RedisBackend

    url = getattr(settings, "REDIS_URL", "")
    if url:
        try:
            backend = RedisBackend(url)
            logger.info("query_cache_backend: redis")
            return backend
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis_unavailable: %s — falling back to in-memory cache", exc)
    return InMemoryBackend()


def _get_query_cache(settings) -> QueryCache | None:
    """Return the query cache, or None when caching is off."""
    global _query_cache
    if getattr(settings, "CACHE_ENABLED", False) is not True:
        return None
    if _query_cache is None:
        _query_cache = QueryCache(
            embed_fn=lambda q: get_embedder().embed_query(q),
            threshold=getattr(settings, "CACHE_SIMILARITY_THRESHOLD", 0.95),
            backend=_make_cache_backend(settings),
        )
    return _query_cache


def query_pipeline(
    question: str, top_k: int | None = None, sources: list[str] | None = None
) -> dict:
    settings = get_settings()
    if top_k is None:
        top_k = settings.TOP_K
    start = time.time()
    logger.info("query_start: question=%s", question[:100])

    # Scoped queries bypass the cache: cache keys are question-only, so a
    # cached unscoped answer would leak out-of-scope sources (and vice versa).
    cache = _get_query_cache(settings)
    if cache is not None and not sources:
        cached = cache.get(question)
        if cached is not None:
            logger.info("cache_hit: question=%s", question[:100])
            return {**cached, "cached": True}

    reranked = _retrieve_and_rerank(question, top_k, settings, sources=sources)

    retrieval_ms = (time.time() - start) * 1000
    trace_data = [
        {"content": d.page_content[:100], "score": d.metadata.get("relevance_score", 0)}
        for d in reranked
    ]
    trace_retrieval(question, trace_data, retrieval_ms)

    # LLM generation. Context is numbered so the grounded prompt's [n]
    # citations map back to the sources returned below. complete_with_model()
    # routes to the strong model and falls back to a same-provider model on
    # failure, returning the model that answered so cost is attributed correctly.
    context = format_context(reranked)
    prompt_text = select_prompt(settings.PROMPT_MODE).format(context=context, question=question)
    try:
        answer, model = complete_with_model(prompt_text)
        logger.info("llm_complete: latency_ms=%.1f", (time.time() - start) * 1000 - retrieval_ms)
    except Exception as exc:
        logger.error("llm_generation_failed: %s", exc)
        raise

    total_ms = (time.time() - start) * 1000
    logger.info("query_complete: latency_ms=%.1f", total_ms)

    usage = usage_for(prompt_text, answer, model)
    logger.info(
        "query_usage: input_tokens=%d output_tokens=%d cost_usd=%.6f",
        usage["input_tokens"], usage["output_tokens"], usage["cost_usd"],
    )

    result = {
        "answer": answer,
        "sources": _docs_to_sources(reranked),
        "latency_ms": total_ms,
        "usage": usage,
    }
    if cache is not None and not sources:
        cache.put(question, result)
    return result


async def stream_query(
    question: str, top_k: int | None = None, sources: list[str] | None = None
) -> AsyncIterator[dict]:
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

    reranked = await asyncio.to_thread(_retrieve_and_rerank, question, top_k, settings, sources)
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
