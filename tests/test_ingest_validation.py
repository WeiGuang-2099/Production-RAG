from unittest.mock import MagicMock

import pytest

from app.ingestion.validation import validate_source


def _settings(tmp_path):
    s = MagicMock()
    s.DATA_DIR = str(tmp_path)
    s.MAX_FILE_SIZE_MB = 100
    return s


def test_url_passthrough(tmp_path):
    assert validate_source("https://example.com/x.pdf", _settings(tmp_path)) == "https://example.com/x.pdf"


def test_rejects_path_outside_data_dir(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    evil = tmp_path / "data-evil"
    evil.mkdir()
    f = evil / "doc.md"
    f.write_text("x")
    s = MagicMock()
    s.DATA_DIR = str(data)
    s.MAX_FILE_SIZE_MB = 100
    with pytest.raises(ValueError, match="within DATA_DIR"):
        validate_source(str(f), s)


def test_rejects_bad_extension(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("x")
    with pytest.raises(ValueError, match="Unsupported file type"):
        validate_source(str(f), _settings(tmp_path))


def test_accepts_valid_file(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("x")
    assert validate_source(str(f), _settings(tmp_path)) == str(f)
