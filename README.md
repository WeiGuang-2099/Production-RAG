# Production RAG System

Production-grade Retrieval-Augmented Generation with hybrid retrieval, GraphRAG, reranking, and observability.

## Quick Start

```bash
# 1. Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# 2. Start services
docker-compose up -d

# 3. Ingest documents
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "path/to/document.pdf"}'

# 4. Query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -v

# Run with coverage
pytest --cov=app --cov-report=term-missing
```

## Evaluation

```bash
# Set up API keys in .env, then:
python evaluation/run_eval.py
```

## Architecture

- **Ingest**: Loaders (PDF/MD/Web) -> Chunker -> Embedder -> Qdrant + BM25 + Graph
- **Query**: Vector+BM25 hybrid (RRF fuse) -> GraphRAG expand -> Rerank -> LLM generate
- **Config**: All settings via `.env`, provider-agnostic factories
- **Observability**: LangSmith tracing on every retrieval

## Configuration

| Variable | Default | Description |
|---|---|---|
| LLM_PROVIDER | openai | openai / anthropic |
| LLM_MODEL | gpt-4o | Chat model name |
| EMBEDDING_PROVIDER | openai | openai / huggingface |
| EMBEDDING_MODEL | text-embedding-3-small | Embedding model |
| RERANKER_PROVIDER | cohere | cohere / none |
| GRAPH_EXTRACTOR | llm | llm / nlp / none |
| QDRANT_URL | http://localhost:6333 | Qdrant endpoint |
| COLLECTION_NAME | rag_docs | Qdrant collection |
| TOP_K | 5 | Initial retrieval count |
| RERANK_TOP_K | 3 | Post-rerank count |
| CHUNK_SIZE | 512 | Chunk token size |
| CHUNK_OVERLAP | 64 | Overlap tokens |
| LANGSMITH_API_KEY | - | Tracing API key |
| LANGSMITH_PROJECT | production-rag | LangSmith project |
| DATA_DIR | ./data | Data storage directory |

## Tech Stack

- Python 3.11+, LangChain 0.3+, FastAPI
- Qdrant (vector store), rank_bm25 (keyword), NetworkX (graph)
- Cohere Rerank, LangSmith tracing, RAGAS evaluation
- Docker Compose deployment
