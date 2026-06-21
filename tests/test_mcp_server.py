def test_mcp_server_module_exposes_app():
    import app.mcp_server as srv
    assert srv.mcp is not None
    assert callable(srv.main)
