import time
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

RAG_PROMPT = """Answer the question based on the following context.

Context:
{context}

Question: {question}

Answer:"""


def ingest_pipeline(source: str) -> dict:
    settings = get_settings()
    documents = load_documents(source)
    chunks = chunk_documents(documents, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

    vs = VectorStore()
    vs.upsert(chunks)

    bm25 = BM25Store(data_dir=settings.DATA_DIR)
    bm25.add_documents(chunks)

    if settings.GRAPH_EXTRACTOR != "none":
        llm = get_llm() if settings.GRAPH_EXTRACTOR == "llm" else None
        builder = GraphBuilder(extractor_type=settings.GRAPH_EXTRACTOR, llm=llm)
        triples = builder.extract(chunks)
        gs = GraphStore(data_dir=settings.DATA_DIR)
        gs.add_triples(triples)

    return {"source": source, "chunks": len(chunks)}


def query_pipeline(question: str) -> dict:
    settings = get_settings()
    start = time.time()

    vs = VectorStore()
    bm25 = BM25Store(data_dir=settings.DATA_DIR)
    retriever = HybridRetriever(vector_store=vs, bm25_store=bm25)

    hybrid_results = retriever.retrieve(question, top_k=settings.TOP_K)

    graph_docs = []
    if settings.GRAPH_EXTRACTOR != "none":
        gs = GraphStore(data_dir=settings.DATA_DIR)
        gr = GraphRetriever(graph_store=gs)
        graph_docs = gr.retrieve(question, depth=1)

    all_docs = [doc for doc, _ in hybrid_results] + graph_docs

    reranker_model = get_reranker()
    reranker_svc = RerankerService(reranker=reranker_model)
    reranked = reranker_svc.rerank(question, all_docs, top_k=settings.RERANK_TOP_K)

    latency_ms = (time.time() - start) * 1000
    trace_data = [
        {"content": d.page_content[:100], "score": d.metadata.get("relevance_score", 0)}
        for d in reranked
    ]
    trace_retrieval(question, trace_data, latency_ms)

    context = "\n\n".join(d.page_content for d in reranked)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    llm = get_llm()
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    return {
        "answer": response.content,
        "sources": [{"content": d.page_content[:200], "metadata": d.metadata} for d in reranked],
        "latency_ms": latency_ms,
    }
