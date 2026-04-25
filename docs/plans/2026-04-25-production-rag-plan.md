# Production RAG System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a production-grade RAG system with hybrid retrieval (vector + BM25 + GraphRAG), configurable LLM providers, Cohere reranking, LangSmith tracing, and Docker deployment.

**Architecture:** Layered monolith FastAPI service. Config-driven factories instantiate the right LLM/embedder/reranker based on environment variables. Data flows through: ingest (load -> chunk -> embed -> store) and query (retrieve -> fuse -> rerank -> generate). All modules are independently testable with mocked externals.

**Tech Stack:** Python 3.11+, LangChain 0.3+, FastAPI, Qdrant, NetworkX, rank_bm25, Cohere, LangSmith, RAGAS, pytest, Docker

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `app/__init__.py`
- Create: `app/main.py`
- Create: `.env.example`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "production-rag"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.6.0",
    "langchain>=0.3.0",
    "langchain-openai>=0.2.0",
    "langchain-anthropic>=0.3.0",
    "langchain-community>=0.3.0",
    "langchain-qdrant>=0.2.0",
    "qdrant-client>=1.12.0",
    "rank-bm25>=0.2.2",
    "networkx>=3.4.0",
    "cohere>=5.13.0",
    "langsmith>=0.2.0",
    "python-multipart>=0.0.12",
    "httpx>=0.28.0",
    "pypdf>=5.1.0",
    "beautifulsoup4>=4.12.0",
    "spacy>=3.8.0",
    "ragas>=0.2.0",
    "numpy>=1.26.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=6.0.0",
    "httpx>=0.28.0",
]

[build-system]
requires = ["setuptools>=75.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

**Step 2: Create .env.example**

```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_API_KEY=sk-xxx
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=sk-xxx
RERANKER_PROVIDER=cohere
RERANKER_MODEL=rerank-v3
COHERE_API_KEY=xxx
GRAPH_EXTRACTOR=llm
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=rag_docs
LANGSMITH_API_KEY=xxx
LANGSMITH_PROJECT=production-rag
LANGSMITH_TRACING=true
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K=5
RERANK_TOP_K=3
DATA_DIR=./data
```

**Step 3: Create app/__init__.py and app/main.py**

```python
# app/__init__.py - empty
```

```python
# app/main.py
from fastapi import FastAPI

app = FastAPI(title="Production RAG", version="0.1.0")


@app.get("/health")
async def health():
    return {"status": "ok"}
```

**Step 4: Create tests/__init__.py and tests/conftest.py**

```python
# tests/__init__.py - empty
```

```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)
```

**Step 5: Write and run the health check test**

Create `tests/test_main.py`:

```python
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

Run: `cd "D:/codeproject/Production RAG 系统" && python -m pytest tests/test_main.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git init
git add pyproject.toml .env.example app/ tests/
git commit -m "feat: project scaffolding with FastAPI and test setup"
```

---

### Task 2: Config Module

**Files:**
- Create: `app/config.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
import os
import pytest
from app.config import Settings


def test_default_settings():
    settings = Settings()
    assert settings.LLM_PROVIDER == "openai"
    assert settings.LLM_MODEL == "gpt-4o"
    assert settings.EMBEDDING_PROVIDER == "openai"
    assert settings.CHUNK_SIZE == 512
    assert settings.CHUNK_OVERLAP == 64
    assert settings.TOP_K == 5
    assert settings.RERANK_TOP_K == 3
    assert settings.COLLECTION_NAME == "rag_docs"


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-20250514")
    monkeypatch.setenv("CHUNK_SIZE", "1024")
    settings = Settings()
    assert settings.LLM_PROVIDER == "anthropic"
    assert settings.LLM_MODEL == "claude-sonnet-4-20250514"
    assert settings.CHUNK_SIZE == 1024
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL (module not found)

**Step 3: Write implementation**

```python
# app/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o"
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str | None = None

    # Embedding
    EMBEDDING_PROVIDER: str = "openai"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str | None = None

    # Reranker
    RERANKER_PROVIDER: str = "cohere"
    RERANKER_MODEL: str = "rerank-v3"
    COHERE_API_KEY: str = ""

    # Graph
    GRAPH_EXTRACTOR: str = "llm"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    COLLECTION_NAME: str = "rag_docs"

    # LangSmith
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "production-rag"
    LANGSMITH_TRACING: bool = False

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # Retrieval
    TOP_K: int = 5
    RERANK_TOP_K: int = 3

    # Data
    DATA_DIR: str = "./data"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


def get_settings() -> Settings:
    return Settings()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat: pydantic settings config module"
```

---

### Task 3: Factories

**Files:**
- Create: `app/core/__init__.py`
- Create: `app/core/factories.py`
- Create: `tests/test_factories.py`

**Step 1: Write the failing test**

```python
# tests/test_factories.py
import pytest
from unittest.mock import MagicMock, patch
from app.core.factories import get_llm, get_embedder, get_reranker


def test_get_llm_openai():
    with patch("app.core.factories.ChatOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock()
        with patch("app.core.factories.get_settings") as mock_settings:
            mock_settings.return_value.LLM_PROVIDER = "openai"
            mock_settings.return_value.LLM_MODEL = "gpt-4o"
            mock_settings.return_value.LLM_API_KEY = "sk-test"
            mock_settings.return_value.LLM_BASE_URL = None
            llm = get_llm()
            mock_cls.assert_called_once()


def test_get_llm_anthropic():
    with patch("app.core.factories.ChatAnthropic") as mock_cls:
        mock_cls.return_value = MagicMock()
        with patch("app.core.factories.get_settings") as mock_settings:
            mock_settings.return_value.LLM_PROVIDER = "anthropic"
            mock_settings.return_value.LLM_MODEL = "claude-sonnet-4-20250514"
            mock_settings.return_value.LLM_API_KEY = "sk-ant-test"
            mock_settings.return_value.LLM_BASE_URL = None
            llm = get_llm()
            mock_cls.assert_called_once()


def test_get_llm_unsupported():
    with patch("app.core.factories.get_settings") as mock_settings:
        mock_settings.return_value.LLM_PROVIDER = "ollama"
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_llm()


def test_get_embedder_openai():
    with patch("app.core.factories.OpenAIEmbeddings") as mock_cls:
        mock_cls.return_value = MagicMock()
        with patch("app.core.factories.get_settings") as mock_settings:
            mock_settings.return_value.EMBEDDING_PROVIDER = "openai"
            mock_settings.return_value.EMBEDDING_MODEL = "text-embedding-3-small"
            mock_settings.return_value.EMBEDDING_API_KEY = "sk-test"
            mock_settings.return_value.EMBEDDING_BASE_URL = None
            embedder = get_embedder()
            mock_cls.assert_called_once()


def test_get_reranker_cohere():
    with patch("app.core.factories.CohereRerank") as mock_cls:
        mock_cls.return_value = MagicMock()
        with patch("app.core.factories.get_settings") as mock_settings:
            mock_settings.return_value.RERANKER_PROVIDER = "cohere"
            mock_settings.return_value.RERANKER_MODEL = "rerank-v3"
            mock_settings.return_value.COHERE_API_KEY = "test-key"
            reranker = get_reranker()
            mock_cls.assert_called_once()


def test_get_reranker_none():
    with patch("app.core.factories.get_settings") as mock_settings:
        mock_settings.return_value.RERANKER_PROVIDER = "none"
        reranker = get_reranker()
        assert reranker is None
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_factories.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# app/core/__init__.py - empty
```

```python
# app/core/factories.py
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Cohere
from langchain_cohere import CohereRerank
from app.config import get_settings


def get_llm():
    settings = get_settings()
    if settings.LLM_PROVIDER == "openai":
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY or None,
            base_url=settings.LLM_BASE_URL,
        )
    elif settings.LLM_PROVIDER == "anthropic":
        return ChatAnthropic(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY or None,
            base_url=settings.LLM_BASE_URL,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")


def get_embedder():
    settings = get_settings()
    if settings.EMBEDDING_PROVIDER == "openai":
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.EMBEDDING_API_KEY or None,
            base_url=settings.EMBEDDING_BASE_URL,
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}")


def get_reranker():
    settings = get_settings()
    if settings.RERANKER_PROVIDER == "none":
        return None
    if settings.RERANKER_PROVIDER == "cohere":
        return CohereRerank(
            model=settings.RERANKER_MODEL,
            cohere_api_key=settings.COHERE_API_KEY,
        )
    raise ValueError(f"Unsupported reranker provider: {settings.RERANKER_PROVIDER}")
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_factories.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/core/ tests/test_factories.py
git commit -m "feat: config-driven factory functions for LLM, embedder, reranker"
```

---

### Task 4: Document Loaders

**Files:**
- Create: `app/ingestion/__init__.py`
- Create: `app/ingestion/loaders.py`
- Create: `tests/test_loaders.py`
- Create: `tests/fixtures/` (test files)

**Step 1: Create test fixtures**

Create `tests/fixtures/sample.pdf` (a minimal PDF -- we will mock pypdf in tests) and `tests/fixtures/sample.md`:

```markdown
# Test Document

This is a test markdown document.

## Section 1

Content for section one.
```

**Step 2: Write the failing test**

```python
# tests/test_loaders.py
import pytest
from unittest.mock import patch, MagicMock
from app.ingestion.loaders import load_pdf, load_markdown, load_webpage, load_documents


def test_load_markdown(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("# Title\n\nHello world.")
    docs = load_markdown(str(md_file))
    assert len(docs) == 1
    assert "Hello world." in docs[0].page_content
    assert docs[0].metadata["source"] == str(md_file)
    assert docs[0].metadata["file_type"] == "markdown"


def test_load_pdf(tmp_path):
    with patch("app.ingestion.loaders.PyPDFLoader") as mock_loader:
        mock_doc = MagicMock()
        mock_doc.page_content = "PDF content here."
        mock_doc.metadata = {"source": "test.pdf", "page": 0}
        mock_loader.return_value.load.return_value = [mock_doc]

        docs = load_pdf("test.pdf")
        assert len(docs) == 1
        assert docs[0].metadata["file_type"] == "pdf"


def test_load_webpage():
    with patch("app.ingestion.loaders.WebBaseLoader") as mock_loader:
        mock_doc = MagicMock()
        mock_doc.page_content = "Web content."
        mock_doc.metadata = {"source": "https://example.com"}
        mock_loader.return_value.load.return_value = [mock_doc]

        docs = load_webpage("https://example.com")
        assert len(docs) == 1
        assert docs[0].metadata["file_type"] == "web"


def test_load_documents_routing(tmp_path):
    md_file = tmp_path / "test.md"
    md_file.write_text("Content")
    docs = load_documents(str(md_file))
    assert len(docs) == 1


def test_load_documents_unsupported():
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_documents("file.xyz")
```

**Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_loaders.py -v`
Expected: FAIL

**Step 4: Write implementation**

```python
# app/ingestion/__init__.py - empty
```

```python
# app/ingestion/loaders.py
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, WebBaseLoader
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["file_type"] = "pdf"
    return docs


def load_markdown(file_path: str) -> list[Document]:
    loader = UnstructuredMarkdownLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = file_path
        doc.metadata["file_type"] = "markdown"
    return docs


def load_webpage(url: str) -> list[Document]:
    loader = WebBaseLoader(url)
    docs = loader.load()
    for doc in docs:
        doc.metadata["file_type"] = "web"
    return docs


def load_documents(source: str) -> list[Document]:
    path = Path(source)
    if path.exists():
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return load_pdf(source)
        elif suffix in (".md", ".markdown"):
            return load_markdown(source)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    elif source.startswith(("http://", "https://")):
        return load_webpage(source)
    else:
        raise ValueError(f"Unsupported source: {source}")
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_loaders.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add app/ingestion/ tests/test_loaders.py tests/fixtures/
git commit -m "feat: document loaders for PDF, Markdown, and web pages"
```

---

### Task 5: Chunkers

**Files:**
- Create: `app/ingestion/chunkers.py`
- Create: `tests/test_chunkers.py`

**Step 1: Write the failing test**

```python
# tests/test_chunkers.py
from app.ingestion.chunkers import chunk_documents
from langchain_core.documents import Document


def test_chunk_documents():
    docs = [Document(page_content="Word " * 200, metadata={"source": "test.txt"})]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all("source" in c.metadata for c in chunks)
    assert all(c.metadata["chunk_index"] is not None for c in chunks)


def test_chunk_documents_empty():
    chunks = chunk_documents([])
    assert chunks == []


def test_chunk_preserves_metadata():
    docs = [Document(page_content="Short text.", metadata={"source": "a.txt", "custom": "val"})]
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].metadata["source"] == "a.txt"
    assert chunks[0].metadata["custom"] == "val"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_chunkers.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# app/ingestion/chunkers.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_chunkers.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/ingestion/chunkers.py tests/test_chunkers.py
git commit -m "feat: recursive character text splitter for document chunking"
```

---

### Task 6: Embedder Wrapper

**Files:**
- Create: `app/ingestion/embedder.py`
- Create: `tests/test_embedder.py`

**Step 1: Write the failing test**

```python
# tests/test_embedder.py
import pytest
from unittest.mock import MagicMock, patch
from app.ingestion.embedder import Embedder
from langchain_core.documents import Document


def test_embed_documents():
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    embedder = Embedder(mock_embeddings)
    texts = ["hello", "world"]
    vectors = embedder.embed_documents(texts)
    assert len(vectors) == 2
    mock_embeddings.embed_documents.assert_called_once_with(texts)


def test_embed_query():
    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

    embedder = Embedder(mock_embeddings)
    vector = embedder.embed_query("test query")
    assert len(vector) == 3
    mock_embeddings.embed_query.assert_called_once_with("test query")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_embedder.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# app/ingestion/embedder.py
from langchain_core.embeddings import Embeddings


class Embedder:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        return self.embeddings.embed_query(query)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_embedder.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/ingestion/embedder.py tests/test_embedder.py
git commit -m "feat: embedder wrapper for document and query embedding"
```

---

### Task 7: Vector Store (Qdrant)

**Files:**
- Create: `app/retrieval/__init__.py`
- Create: `app/retrieval/vector_store.py`
- Create: `tests/test_vector_store.py`

**Step 1: Write the failing test**

```python
# tests/test_vector_store.py
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from app.retrieval.vector_store import VectorStore
from langchain_core.documents import Document


@pytest.fixture
def mock_qdrant():
    with patch("app.retrieval.vector_store.QdrantVectorStore") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.from_texts.return_value = mock_instance
        mock_cls.return_value = mock_instance
        yield mock_instance, mock_cls


def test_upsert_documents(mock_qdrant):
    mock_instance, mock_cls = mock_qdrant
    mock_instance.add_documents.return_value = ["id1", "id2"]

    with patch("app.retrieval.vector_store.get_embedder") as mock_emb:
        mock_emb.return_value = MagicMock()
        with patch("app.retrieval.vector_store.get_settings") as mock_s:
            mock_s.return_value.QDRANT_URL = "http://localhost:6333"
            mock_s.return_value.COLLECTION_NAME = "test_col"
            mock_s.return_value.QDRANT_API_KEY = None

            vs = VectorStore()
            docs = [
                Document(page_content="hello", metadata={"source": "a"}),
                Document(page_content="world", metadata={"source": "b"}),
            ]
            ids = vs.upsert(docs)
            assert len(ids) == 2


def test_search(mock_qdrant):
    mock_instance, mock_cls = mock_qdrant
    mock_doc = Document(page_content="result", metadata={"score": 0.9})
    mock_instance.similarity_search_with_score.return_value = [(mock_doc, 0.9)]

    with patch("app.retrieval.vector_store.get_embedder") as mock_emb:
        mock_emb.return_value = MagicMock()
        with patch("app.retrieval.vector_store.get_settings") as mock_s:
            mock_s.return_value.QDRANT_URL = "http://localhost:6333"
            mock_s.return_value.COLLECTION_NAME = "test_col"
            mock_s.return_value.QDRANT_API_KEY = None

            vs = VectorStore()
            results = vs.search("test query", top_k=5)
            assert len(results) == 1
            assert results[0][1] == 0.9
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_vector_store.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# app/retrieval/__init__.py - empty
```

```python
# app/retrieval/vector_store.py
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from app.config import get_settings
from app.core.factories import get_embedder


class VectorStore:
    def __init__(self):
        settings = get_settings()
        self.embedder = get_embedder()
        self.collection_name = settings.COLLECTION_NAME
        self.url = settings.QDRANT_URL
        self.api_key = settings.QDRANT_API_KEY
        self._store: QdrantVectorStore | None = None

    def _get_store(self) -> QdrantVectorStore:
        if self._store is None:
            self._store = QdrantVectorStore.from_existing_collection(
                embedding=self.embedder,
                url=self.url,
                api_key=self.api_key,
                collection_name=self.collection_name,
            )
        return self._store

    def upsert(self, documents: list[Document]) -> list[str]:
        store = self._get_store()
        return store.add_documents(documents)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        store = self._get_store()
        return store.similarity_search_with_score(query, k=top_k)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_vector_store.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/retrieval/vector_store.py tests/test_vector_store.py
git commit -m "feat: Qdrant vector store with upsert and search"
```

---

### Task 8: BM25 Store

**Files:**
- Create: `app/retrieval/bm25_store.py`
- Create: `tests/test_bm25_store.py`

**Step 1: Write the failing test**

```python
# tests/test_bm25_store.py
import pytest
from app.retrieval.bm25_store import BM25Store
from langchain_core.documents import Document


def test_add_and_search(tmp_path):
    store = BM25Store(data_dir=str(tmp_path))
    docs = [
        Document(page_content="Machine learning is a subset of artificial intelligence.", metadata={"source": "a"}),
        Document(page_content="Python is a popular programming language.", metadata={"source": "b"}),
        Document(page_content="Neural networks are used in deep learning.", metadata={"source": "c"}),
    ]
    store.add_documents(docs)

    results = store.search("machine learning", top_k=2)
    assert len(results) == 2
    assert results[0][0].page_content == docs[0].page_content
    assert results[0][1] > 0


def test_search_empty(tmp_path):
    store = BM25Store(data_dir=str(tmp_path))
    results = store.search("test", top_k=5)
    assert results == []


def test_save_and_load(tmp_path):
    store1 = BM25Store(data_dir=str(tmp_path))
    docs = [Document(page_content="test document content", metadata={"source": "a"})]
    store1.add_documents(docs)

    store2 = BM25Store(data_dir=str(tmp_path))
    results = store2.search("test", top_k=5)
    assert len(results) == 1
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bm25_store.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# app/retrieval/bm25_store.py
import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Store:
    def __init__(self, data_dir: str = "./data"):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._documents: list[Document] = []
        self._bm25: BM25Okapi | None = None
        self._load()

    def add_documents(self, documents: list[Document]) -> None:
        self._documents.extend(documents)
        tokenized = [_tokenize(doc.page_content) for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized)
        self._save()

    def search(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        if self._bm25 is None or not self._documents:
            return []
        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._documents[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    def _save(self) -> None:
        if self._bm25 is None:
            return
        with open(self._data_dir / "bm25_index.pkl", "wb") as f:
            pickle.dump({"documents": self._documents, "bm25": self._bm25}, f)

    def _load(self) -> None:
        path = self._data_dir / "bm25_index.pkl"
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._documents = data["documents"]
            self._bm25 = data["bm25"]
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bm25_store.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/retrieval/bm25_store.py tests/test_bm25_store.py
git commit -m "feat: BM25 keyword store with persistence"
```

---

### Task 9: Hybrid Retriever (RRF Fusion)

**Files:**
- Create: `app/retrieval/hybrid_retriever.py`
- Create: `tests/test_hybrid_retriever.py`

**Step 1: Write the failing test**

```python
# tests/test_hybrid_retriever.py
import pytest
from unittest.mock import MagicMock
from app.retrieval.hybrid_retriever import HybridRetriever, rrf_fuse
from langchain_core.documents import Document


def test_rrf_fuse_basic():
    doc_a = Document(page_content="doc A")
    doc_b = Document(page_content="doc B")
    doc_c = Document(page_content="doc C")

    ranked_lists = [
        [(doc_a, 0.9), (doc_b, 0.8)],
        [(doc_b, 0.95), (doc_c, 0.7)],
    ]
    results = rrf_fuse(ranked_lists, k=60)
    assert len(results) == 3
    # doc_b appears in both lists so should rank highest
    assert results[0][0].page_content == "doc B"


def test_rrf_fuse_single_list():
    doc_a = Document(page_content="doc A")
    results = rrf_fuse([[(doc_a, 0.9)]], k=60)
    assert len(results) == 1


def test_hybrid_retriever():
    mock_vector = MagicMock()
    mock_vector.search.return_value = [
        (Document(page_content="vec result 1"), 0.9),
        (Document(page_content="vec result 2"), 0.7),
    ]

    mock_bm25 = MagicMock()
    mock_bm25.search.return_value = [
        (Document(page_content="vec result 1"), 2.5),
        (Document(page_content="bm25 result"), 1.8),
    ]

    retriever = HybridRetriever(vector_store=mock_vector, bm25_store=mock_bm25)
    results = retriever.retrieve("test query", top_k=3)
    assert len(results) <= 3
    assert len(results) > 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hybrid_retriever.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# app/retrieval/hybrid_retriever.py
from langchain_core.documents import Document


def rrf_fuse(
    ranked_lists: list[list[tuple[Document, float]]],
    k: int = 60,
) -> list[tuple[Document, float]]:
    scores: dict[str, tuple[Document, float]] = {}

    for ranked in ranked_lists:
        for rank, (doc, _original_score) in enumerate(ranked):
            content = doc.page_content
            rrf_score = 1.0 / (k + rank + 1)
            if content in scores:
                existing_doc, existing_score = scores[content]
                scores[content] = (existing_doc, existing_score + rrf_score)
            else:
                scores[content] = (doc, rrf_score)

    sorted_results = sorted(scores.values(), key=lambda x: x[1], reverse=True)
    return sorted_results


class HybridRetriever:
    def __init__(self, vector_store, bm25_store):
        self.vector_store = vector_store
        self.bm25_store = bm25_store

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        vector_results = self.vector_store.search(query, top_k=top_k)
        bm25_results = self.bm25_store.search(query, top_k=top_k)
        fused = rrf_fuse([vector_results, bm25_results])
        return fused[:top_k]
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hybrid_retriever.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/retrieval/hybrid_retriever.py tests/test_hybrid_retriever.py
git commit -m "feat: hybrid retriever with RRF fusion of vector and BM25"
```

---

### Task 10: Graph Module (Builder + Store)

**Files:**
- Create: `app/graph/__init__.py`
- Create: `app/graph/builder.py`
- Create: `app/graph/store.py`
- Create: `tests/test_graph.py`

**Step 1: Write the failing test**

```python
# tests/test_graph.py
import pytest
from unittest.mock import MagicMock, patch
from app.graph.builder import GraphBuilder
from app.graph.store import GraphStore
from langchain_core.documents import Document


def test_llm_extractor():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content='[{"head": "Python", "relation": "is_a", "tail": "programming language"}]'
    )

    builder = GraphBuilder(extractor_type="llm", llm=mock_llm)
    docs = [Document(page_content="Python is a programming language.")]
    triples = builder.extract(docs)
    assert len(triples) >= 1
    assert triples[0]["head"] == "Python"
    assert triples[0]["relation"] == "is_a"


def test_nlp_extractor():
    with patch("app.graph.builder.spacy") as mock_spacy:
        mock_nlp = MagicMock()
        mock_ent1 = MagicMock(text="Python", label_="LANGUAGE")
        mock_ent2 = MagicMock(text="Google", label_="ORG")
        mock_doc = MagicMock(ents=[mock_ent1, mock_ent2], sents=[MagicMock(text="Python was created at Google.")])
        mock_nlp.pipe.return_value = [mock_doc]
        mock_spacy.load.return_value = mock_nlp

        builder = GraphBuilder(extractor_type="nlp")
        docs = [Document(page_content="Python was created at Google.")]
        triples = builder.extract(docs)
        assert len(triples) >= 1


def test_graph_store_add_and_query(tmp_path):
    store = GraphStore(data_dir=str(tmp_path))
    store.add_triples([
        {"head": "A", "relation": "connects", "tail": "B"},
        {"head": "B", "relation": "connects", "tail": "C"},
        {"head": "X", "relation": "connects", "tail": "Y"},
    ])

    neighbors = store.get_neighbors("A", depth=2)
    names = [n for n, _ in neighbors]
    assert "B" in names
    assert "C" in names


def test_graph_store_persistence(tmp_path):
    store1 = GraphStore(data_dir=str(tmp_path))
    store1.add_triples([{"head": "A", "relation": "r", "tail": "B"}])

    store2 = GraphStore(data_dir=str(tmp_path))
    neighbors = store2.get_neighbors("A", depth=1)
    assert any(n == "B" for n, _ in neighbors)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# app/graph/__init__.py - empty
```

```python
# app/graph/builder.py
import json
import re
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel


EXTRACTION_PROMPT = """Extract entity-relationship triples from the following text.
Return ONLY a JSON array of objects with keys: head, relation, tail.

Text: {text}

JSON:"""


class GraphBuilder:
    def __init__(self, extractor_type: str = "llm", llm: BaseChatModel | None = None):
        self.extractor_type = extractor_type
        self.llm = llm

    def extract(self, documents: list[Document]) -> list[dict]:
        if self.extractor_type == "llm":
            return self._extract_llm(documents)
        elif self.extractor_type == "nlp":
            return self._extract_nlp(documents)
        return []

    def _extract_llm(self, documents: list[Document]) -> list[dict]:
        if not self.llm:
            return []
        all_triples = []
        for doc in documents:
            prompt = EXTRACTION_PROMPT.format(text=doc.page_content)
            response = self.llm.invoke(prompt)
            triples = self._parse_response(response.content)
            all_triples.extend(triples)
        return all_triples

    def _extract_nlp(self, documents: list[Document]) -> list[dict]:
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            nlp = spacy.blank("en")

        triples = []
        for doc in documents:
            spacy_doc = nlp(doc.page_content)
            entities = [(ent.text, ent.label_) for ent in spacy_doc.ents]
            for i, (text_a, _) in enumerate(entities):
                for text_b, _ in entities[i + 1:]:
                    triples.append({"head": text_a, "relation": "related_to", "tail": text_b})
        return triples

    def _parse_response(self, content: str) -> list[dict]:
        try:
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return []
```

```python
# app/graph/store.py
import pickle
from pathlib import Path
import networkx as nx


class GraphStore:
    def __init__(self, data_dir: str = "./data"):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self.graph = nx.DiGraph()
        self._load()

    def add_triples(self, triples: list[dict]) -> None:
        for triple in triples:
            self.graph.add_edge(
                triple["head"],
                triple["tail"],
                relation=triple["relation"],
            )
        self._save()

    def get_neighbors(self, entity: str, depth: int = 1) -> list[tuple[str, str]]:
        if entity not in self.graph:
            return []
        visited = set()
        queue = [(entity, 0)]
        result = []
        while queue:
            node, d = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            if d > 0:
                for _, neighbor, data in self.graph.edges(node, data=True):
                    result.append((neighbor, data.get("relation", "")))
                for pred, _, data in self.graph.in_edges(node, data=True):
                    result.append((pred, data.get("relation", "")))
            if d < depth:
                for neighbor in self.graph.successors(node):
                    if neighbor not in visited:
                        queue.append((neighbor, d + 1))
                for pred in self.graph.predecessors(node):
                    if pred not in visited:
                        queue.append((pred, d + 1))
        return result

    def _save(self) -> None:
        with open(self._data_dir / "knowledge_graph.gpickle", "wb") as f:
            pickle.dump(self.graph, f)

    def _load(self) -> None:
        path = self._data_dir / "knowledge_graph.gpickle"
        if path.exists():
            with open(path, "rb") as f:
                self.graph = pickle.load(f)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_graph.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/graph/ tests/test_graph.py
git commit -m "feat: graph builder (LLM/NLP) and NetworkX graph store"
```

---

### Task 11: Graph Retriever

**Files:**
- Create: `app/retrieval/graph_retriever.py`
- Create: `tests/test_graph_retriever.py`

**Step 1: Write the failing test**

```python
# tests/test_graph_retriever.py
import pytest
from unittest.mock import MagicMock
from app.retrieval.graph_retriever import GraphRetriever
from langchain_core.documents import Document


def test_graph_retriever():
    mock_graph_store = MagicMock()
    mock_graph_store.get_neighbors.return_value = [
        ("machine learning", "is_subset_of"),
        ("neural networks", "uses"),
    ]

    retriever = GraphRetriever(graph_store=mock_graph_store)
    docs = retriever.retrieve("artificial intelligence")
    assert len(docs) > 0
    assert any("machine learning" in d.page_content for d in docs)


def test_graph_retriever_no_neighbors():
    mock_graph_store = MagicMock()
    mock_graph_store.get_neighbors.return_value = []

    retriever = GraphRetriever(graph_store=mock_graph_store)
    docs = retriever.retrieve("unknown entity")
    assert docs == []
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_graph_retriever.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# app/retrieval/graph_retriever.py
from langchain_core.documents import Document
from app.graph.store import GraphStore


class GraphRetriever:
    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store

    def retrieve(self, query: str, depth: int = 1) -> list[Document]:
        neighbors = self.graph_store.get_neighbors(query, depth=depth)
        if not neighbors:
            return []
        documents = []
        for entity, relation in neighbors:
            documents.append(
                Document(
                    page_content=f"{query} --[{relation}]--> {entity}",
                    metadata={"source": "graph", "entity": entity, "relation": relation},
                )
            )
        return documents
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_graph_retriever.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/retrieval/graph_retriever.py tests/test_graph_retriever.py
git commit -m "feat: graph retriever for entity-expanded context"
```

---

### Task 12: Reranker

**Files:**
- Create: `app/reranker/reranker.py`
- Create: `tests/test_reranker.py`

**Step 1: Write the failing test**

```python
# tests/test_reranker.py
import pytest
from unittest.mock import MagicMock
from app.reranker.reranker import RerankerService
from langchain_core.documents import Document


def test_rerank_with_cohere():
    mock_cohere = MagicMock()
    mock_result_1 = MagicMock(index=0, relevance_score=0.95)
    mock_result_2 = MagicMock(index=1, relevance_score=0.7)
    mock_cohere.rerank.return_value = MagicMock(results=[mock_result_1, mock_result_2])

    service = RerankerService(reranker=mock_cohere)
    docs = [
        Document(page_content="relevant doc"),
        Document(page_content="less relevant doc"),
    ]
    results = service.rerank("test query", docs, top_k=2)
    assert len(results) == 2
    assert results[0].metadata["relevance_score"] == 0.95


def test_rerank_no_reranker():
    service = RerankerService(reranker=None)
    docs = [Document(page_content="doc1"), Document(page_content="doc2")]
    results = service.rerank("test query", docs, top_k=2)
    assert len(results) == 2
    assert results[0].page_content == "doc1"


def test_rerank_fewer_than_top_k():
    service = RerankerService(reranker=None)
    docs = [Document(page_content="doc1")]
    results = service.rerank("test query", docs, top_k=5)
    assert len(results) == 1
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reranker.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# app/reranker/__init__.py - empty
```

```python
# app/reranker/reranker.py
from langchain_core.documents import Document
from langchain_cohere import CohereRerank


class RerankerService:
    def __init__(self, reranker: CohereRerank | None = None):
        self.reranker = reranker

    def rerank(self, query: str, documents: list[Document], top_k: int = 3) -> list[Document]:
        if self.reranker is None:
            return documents[:top_k]

        results = self.reranker.rerank(
            documents=[doc.page_content for doc in documents],
            query=query,
            top_n=top_k,
        )
        reranked = []
        for result in results.results:
            doc = documents[result.index]
            doc.metadata["relevance_score"] = result.relevance_score
            reranked.append(doc)
        return reranked
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reranker.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/reranker/ tests/test_reranker.py
git commit -m "feat: reranker service with Cohere and passthrough modes"
```

---

### Task 13: Observability (LangSmith Tracing)

**Files:**
- Create: `app/observability/__init__.py`
- Create: `app/observability/tracing.py`
- Create: `tests/test_tracing.py`

**Step 1: Write the failing test**

```python
# tests/test_tracing.py
import pytest
from unittest.mock import patch, MagicMock
from app.observability.tracing import setup_tracing, trace_retrieval


def test_setup_tracing_enabled():
    with patch("app.observability.tracing.get_settings") as mock_s:
        mock_s.return_value.LANGSMITH_TRACING = True
        mock_s.return_value.LANGSMITH_API_KEY = "test-key"
        mock_s.return_value.LANGSMITH_PROJECT = "test-project"
        with patch.dict("os.environ", {}) as mock_env:
            setup_tracing()
            assert mock_env.get("LANGSMITH_TRACING") == "true"
            assert mock_env.get("LANGSMITH_API_KEY") == "test-key"


def test_setup_tracing_disabled():
    with patch("app.observability.tracing.get_settings") as mock_s:
        mock_s.return_value.LANGSMITH_TRACING = False
        mock_s.return_value.LANGSMITH_API_KEY = ""
        with patch.dict("os.environ", {}) as mock_env:
            setup_tracing()
            assert mock_env.get("LANGSMITH_TRACING", "false") != "true"


def test_trace_retrieval():
    with patch("app.observability.tracing.get_settings") as mock_s:
        mock_s.return_value.LANGSMITH_TRACING = True
        with patch("app.observability.tracing.traceable") as mock_traceable:
            mock_traceable.side_effect = lambda f: f
            result = trace_retrieval("test query", [{"doc": "result", "score": 0.9}], 0.5)
            assert result["query"] == "test query"
            assert result["hit_count"] == 1
            assert result["latency_ms"] == 500
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tracing.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# app/observability/__init__.py - empty
```

```python
# app/observability/tracing.py
import os
import time
from langsmith import traceable
from app.config import get_settings


def setup_tracing() -> None:
    settings = get_settings()
    if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    else:
        os.environ.pop("LANGSMITH_TRACING", None)


@traceable(name="retrieval", run_type="retriever")
def trace_retrieval(query: str, results: list[dict], latency_ms: float) -> dict:
    return {
        "query": query,
        "hit_count": len(results),
        "latency_ms": latency_ms,
        "results": results,
    }
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tracing.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/observability/ tests/test_tracing.py
git commit -m "feat: LangSmith tracing setup and retrieval trace decorator"
```

---

### Task 14: Pipeline Orchestration

**Files:**
- Create: `app/core/pipeline.py`
- Create: `tests/test_pipeline.py`

**Step 1: Write the failing test**

```python
# tests/test_pipeline.py
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
         patch("app.core.pipeline.get_settings") as mock_s:

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

        mock_gb = MagicMock()
        mock_gb_cls.return_value = mock_gb
        mock_gb.extract.return_value = [{"head": "A", "relation": "r", "tail": "B"}]

        mock_gs = MagicMock()
        mock_gs_cls.return_value = mock_gs

        from app.core.pipeline import ingest_pipeline
        result = ingest_pipeline("test.md")
        assert result["chunks"] == 1


def test_query_pipeline():
    with patch("app.core.pipeline.get_llm") as mock_llm_f, \
         patch("app.core.pipeline.get_embedder") as mock_emb, \
         patch("app.core.pipeline.get_reranker") as mock_rr_f, \
         patch("app.core.pipeline.VectorStore") as mock_vs_cls, \
         patch("app.core.pipeline.BM25Store") as mock_bm25_cls, \
         patch("app.core.pipeline.HybridRetriever") as mock_hr_cls, \
         patch("app.core.pipeline.GraphRetriever") as mock_gr_cls, \
         patch("app.core.pipeline.GraphStore") as mock_gs_cls, \
         patch("app.core.pipeline.RerankerService") as mock_rs_cls, \
         patch("app.core.pipeline.get_settings") as mock_s:

        mock_s.return_value.TOP_K = 5
        mock_s.return_value.RERANK_TOP_K = 3

        mock_llm = MagicMock()
        mock_llm_f.return_value = mock_llm
        mock_llm.invoke.return_value = MagicMock(content="Generated answer")

        mock_rr = MagicMock()
        mock_rr_f.return_value = mock_rr

        doc = Document(page_content="context doc")
        mock_hr = MagicMock()
        mock_hr_cls.return_value = mock_hr
        mock_hr.retrieve.return_value = [(doc, 0.9)]

        mock_gr = MagicMock()
        mock_gr_cls.return_value = mock_gr
        mock_gr.retrieve.return_value = []

        mock_rs = MagicMock()
        mock_rs_cls.return_value = mock_rs
        mock_rs.rerank.return_value = [doc]

        from app.core.pipeline import query_pipeline
        result = query_pipeline("What is AI?")
        assert result["answer"] == "Generated answer"
        assert len(result["sources"]) > 0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# app/core/pipeline.py
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
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/core/pipeline.py tests/test_pipeline.py
git commit -m "feat: ingest and query pipeline orchestration"
```

---

### Task 15: API Routes

**Files:**
- Create: `app/api/__init__.py`
- Create: `app/api/routes_ingest.py`
- Create: `app/api/routes_chat.py`
- Modify: `app/main.py`
- Create: `tests/test_api.py`

**Step 1: Write the failing test**

```python
# tests/test_api.py
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


def test_ingest_endpoint(client):
    with patch("app.api.routes_ingest.ingest_pipeline") as mock_ingest:
        mock_ingest.return_value = {"source": "test.pdf", "chunks": 5}
        response = client.post("/ingest", json={"source": "test.pdf"})
        assert response.status_code == 200
        data = response.json()
        assert data["chunks"] == 5


def test_chat_endpoint(client):
    with patch("app.api.routes_chat.query_pipeline") as mock_query:
        mock_query.return_value = {
            "answer": "AI is artificial intelligence.",
            "sources": [{"content": "context", "metadata": {}}],
            "latency_ms": 100.0,
        }
        response = client.post("/chat", json={"question": "What is AI?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["sources"]) == 1


def test_chat_missing_question(client):
    response = client.post("/chat", json={})
    assert response.status_code == 422


def test_ingest_missing_source(client):
    response = client.post("/ingest", json={})
    assert response.status_code == 422
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# app/api/__init__.py - empty
```

```python
# app/api/routes_ingest.py
from fastapi import APIRouter
from pydantic import BaseModel
from app.core.pipeline import ingest_pipeline

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestRequest(BaseModel):
    source: str


class IngestResponse(BaseModel):
    source: str
    chunks: int


@router.post("", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    result = ingest_pipeline(request.source)
    return IngestResponse(**result)
```

```python
# app/api/routes_chat.py
from fastapi import APIRouter
from pydantic import BaseModel
from app.core.pipeline import query_pipeline

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    content: str
    metadata: dict


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    latency_ms: float


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = query_pipeline(request.question)
    return ChatResponse(**result)
```

Update `app/main.py`:

```python
from fastapi import FastAPI
from app.api.routes_chat import router as chat_router
from app.api.routes_ingest import router as ingest_router
from app.observability.tracing import setup_tracing

setup_tracing()

app = FastAPI(title="Production RAG", version="0.1.0")
app.include_router(ingest_router)
app.include_router(chat_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/api/ app/main.py tests/test_api.py
git commit -m "feat: FastAPI /chat and /ingest endpoints with request models"
```

---

### Task 16: RAGAS Evaluation

**Files:**
- Create: `evaluation/eval_dataset.json`
- Create: `evaluation/run_eval.py`

**Step 1: Create eval dataset**

```json
[
  {
    "question": "What is machine learning?",
    "ground_truth": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "contexts": ["Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."],
    "answer": ""
  },
  {
    "question": "What is a neural network?",
    "ground_truth": "A neural network is a computing system inspired by biological neural networks.",
    "contexts": ["Neural networks are computing systems vaguely inspired by biological neural networks that constitute animal brains."],
    "answer": ""
  },
  {
    "question": "What is Python used for?",
    "ground_truth": "Python is used for web development, data analysis, AI, and scripting.",
    "contexts": ["Python is a high-level programming language used for web development, data analysis, artificial intelligence, and scripting."],
    "answer": ""
  }
]
```

**Step 2: Create eval runner**

```python
# evaluation/run_eval.py
"""Run RAGAS evaluation metrics against the RAG system."""
import json
import sys
from pathlib import Path

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.pipeline import query_pipeline


def load_dataset(path: str = "evaluation/eval_dataset.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_evaluation():
    dataset = load_dataset()
    samples = []

    for item in dataset:
        result = query_pipeline(item["question"])
        sample = SingleTurnSample(
            user_input=item["question"],
            response=result["answer"],
            reference=item["ground_truth"],
            retrieved_contexts=[s["content"] for s in result["sources"]],
        )
        samples.append(sample)

    eval_dataset = EvaluationDataset(samples=samples)
    results = evaluate(eval_dataset, metrics=[faithfulness, answer_relevancy, context_recall])
    print(results)
    return results


if __name__ == "__main__":
    run_evaluation()
```

**Step 3: Commit**

```bash
git add evaluation/
git commit -m "feat: RAGAS evaluation dataset and runner script"
```

---

### Task 17: Docker Setup

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`

**Step 1: Create Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY . .
RUN python -m spacy download en_core_web_sm || true

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 2: Create docker-compose.yml**

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      qdrant:
        condition: service_started
    volumes:
      - rag_data:/app/data
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

volumes:
  qdrant_data:
  rag_data:
```

**Step 3: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat: Docker and docker-compose for one-command deployment"
```

---

### Task 18: README

**Files:**
- Create: `README.md`

**Step 1: Create README**

```markdown
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
| EMBEDDING_PROVIDER | openai | openai / huggingface |
| RERANKER_PROVIDER | cohere | cohere / none |
| GRAPH_EXTRACTOR | llm | llm / nlp / none |
| TOP_K | 5 | Initial retrieval count |
| RERANK_TOP_K | 3 | Post-rerank count |
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with quick start, development, and configuration"
```

---

### Task 19: Final Validation

**Step 1: Run full test suite**

Run: `pytest -v --tb=short`
Expected: All tests PASS

**Step 2: Verify directory structure**

Run: `find . -type f -not -path './.git/*' | sort`
Expected: Matches design document structure

**Step 3: Final commit (if any fixes needed)**

```bash
git add -A
git commit -m "chore: final cleanup and validation"
```
