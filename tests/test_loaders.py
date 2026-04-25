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
    with pytest.raises(ValueError, match="Unsupported"):
        load_documents("file.xyz")
