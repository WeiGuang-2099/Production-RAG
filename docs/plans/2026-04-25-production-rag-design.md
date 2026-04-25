# Production-Grade RAG System Design

## Architecture: Layered Monolith

Single FastAPI service with clear module layers behind config-driven factories.
Docker Compose runs FastAPI + Qdrant.

## Directory Structure

```
production-rag/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI entrypoint
│   ├── config.py                # Pydantic Settings, all env-driven
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes_chat.py       # POST /chat
│   │   └── routes_ingest.py     # POST /ingest
│   ├── core/
│   │   ├── __init__.py
│   │   ├── factories.py         # Config-driven factory functions
│   │   └── pipeline.py          # Orchestrate retrieve->rerank->generate
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loaders.py           # PDF/Markdown/Web loaders
│   │   ├── chunkers.py          # Text splitters
│   │   └── embedder.py          # Embedding wrapper
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py      # Qdrant client
│   │   ├── bm25_store.py        # BM25 keyword index
│   │   ├── hybrid_retriever.py  # RRF fusion
│   │   └── graph_retriever.py   # NetworkX GraphRAG
│   ├── reranker/
│   │   ├── __init__.py
│   │   └── reranker.py          # Cohere / configurable reranker
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── builder.py           # Entity/relationship extraction
│   │   └── store.py             # NetworkX graph persistence
│   └── observability/
│       ├── __init__.py
│       └── tracing.py           # LangSmith integration
├── tests/
│   ├── conftest.py
│   ├── test_loaders.py
│   ├── test_chunkers.py
│   ├── test_vector_store.py
│   ├── test_bm25_store.py
│   ├── test_hybrid_retriever.py
│   ├── test_graph_retriever.py
│   ├── test_reranker.py
│   ├── test_pipeline.py
│   └── test_api.py
├── evaluation/
│   ├── eval_dataset.json        # RAGAS test dataset
│   └── run_eval.py              # RAGAS evaluation script
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── .env.example
└── README.md
```

## Data Flow

```
Ingest path:
  File/URL -> Loader -> Chunker -> Embedder -> Qdrant upsert
                                          -> BM25 index update
                                          -> Graph builder -> NetworkX .gpickle

Query path:
  /chat question -> Embed query
                  -> Vector search (Qdrant)  ─┐
                  -> BM25 search              ├──> RRF fuse
                  -> Graph expand (optional)  ─┘     │
                                                        v
                                                   Reranker (Cohere)
                                                        │
                                                        v
                                                   LLM generate answer
                                                        │
                                                        v
                                                   LangSmith trace log
                                                        │
                                                        v
                                                   Return response
```

## Configuration

All via `.env` file:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` / `anthropic` |
| `LLM_MODEL` | `gpt-4o` | Chat model name |
| `EMBEDDING_PROVIDER` | `openai` | `openai` / `huggingface` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `RERANKER_PROVIDER` | `cohere` | `cohere` / `none` |
| `RERANKER_MODEL` | `rerank-v3` | Reranker model |
| `GRAPH_EXTRACTOR` | `llm` | `llm` / `nlp` / `none` |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `COLLECTION_NAME` | `rag_docs` | Qdrant collection |
| `LANGSMITH_API_KEY` | - | Tracing |
| `CHUNK_SIZE` | `512` | Chunk token size |
| `CHUNK_OVERLAP` | `64` | Overlap tokens |
| `TOP_K` | `5` | Initial retrieval count |
| `RERANK_TOP_K` | `3` | Post-rerank count |

## Key Design Decisions

- **Factories**: `get_llm()`, `get_embedder()`, `get_reranker()` read config and return the right instance. No if/else scattered in business logic.
- **BM25**: In-memory `rank_bm25` library. Serialized alongside Qdrant state in Docker volume.
- **GraphRAG**: LLM extractor uses structured output (function calling) to pull (entity, relation, entity) triples. NLP extractor uses spaCy NER + heuristic relations. Graph stored as NetworkX DiGraph, serialized to `.gpickle`.
- **RRF**: Standard formula `score = sum(1 / (k + rank))` with `k=60`.
- **LangSmith**: Decorator-based tracing on the pipeline function. No manual span management.

## Testing Strategy

- **Unit tests**: Each module tested with mocks for external services (Qdrant, LLM, Cohere). Use `pytest` fixtures in `conftest.py`.
- **Integration tests**: `test_api.py` uses FastAPI `TestClient` to hit endpoints end-to-end with a test Qdrant instance.
- **Evaluation**: `evaluation/run_eval.py` runs RAGAS metrics against a curated `eval_dataset.json`.

## Docker Deployment

```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [qdrant]
    volumes:
      - bm25_data:/app/data
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
    volumes:
      - qdrant_data:/qdrant/storage
volumes:
  qdrant_data:
  bm25_data:
```
