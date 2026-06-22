from unittest.mock import patch


def _settings(mock_s, data_dir):
    mock_s.return_value.DATA_DIR = str(data_dir)
    mock_s.return_value.MAX_FILE_SIZE_MB = 100


def test_upload_ingests_markdown(client, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with patch("app.api.routes_ingest.get_settings") as mock_s, \
         patch("app.api.routes_ingest.ingest_pipeline") as mock_ingest:
        _settings(mock_s, data_dir)
        mock_ingest.return_value = {"source": "x", "chunks": 3, "status": "ingested"}
        resp = client.post("/ingest/upload", files={"file": ("doc.md", b"# hi", "text/markdown")})
        assert resp.status_code == 200
        assert resp.json()["chunks"] == 3
        # file was saved under DATA_DIR and passed to the pipeline
        saved = mock_ingest.call_args[0][0]
        assert str(data_dir) in saved
        assert (data_dir / "doc.md").exists()


def test_upload_rejects_bad_suffix(client, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with patch("app.api.routes_ingest.get_settings") as mock_s, \
         patch("app.api.routes_ingest.ingest_pipeline") as mock_ingest:
        _settings(mock_s, data_dir)
        resp = client.post("/ingest/upload", files={"file": ("evil.exe", b"MZ", "application/octet-stream")})
        assert resp.status_code == 400
        mock_ingest.assert_not_called()


def test_upload_rejects_oversize(client, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with patch("app.api.routes_ingest.get_settings") as mock_s, \
         patch("app.api.routes_ingest.ingest_pipeline") as mock_ingest:
        mock_s.return_value.DATA_DIR = str(data_dir)
        mock_s.return_value.MAX_FILE_SIZE_MB = 0  # everything is oversize
        resp = client.post("/ingest/upload", files={"file": ("doc.md", b"abc", "text/markdown")})
        assert resp.status_code == 413
        mock_ingest.assert_not_called()


def test_upload_sanitizes_filename(client, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with patch("app.api.routes_ingest.get_settings") as mock_s, \
         patch("app.api.routes_ingest.ingest_pipeline") as mock_ingest:
        _settings(mock_s, data_dir)
        mock_ingest.return_value = {"source": "x", "chunks": 1, "status": "ingested"}
        resp = client.post("/ingest/upload", files={"file": ("../../evil.md", b"x", "text/markdown")})
        assert resp.status_code == 200
        # saved as basename inside DATA_DIR, no traversal
        assert (data_dir / "evil.md").exists()
        assert not (tmp_path / "evil.md").exists()
