from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["file_type"] = "pdf"
    return docs


def load_markdown(file_path: str) -> list[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return [Document(page_content=content, metadata={"source": file_path, "file_type": "markdown"})]


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
