import pytest
import importlib

@pytest.mark.unit
def test_core_imports():
    """Test that core modules can be imported."""
    assert importlib.import_module("src.core")
    assert importlib.import_module("src.ingestion")
    assert importlib.import_module("src.libs")
    assert importlib.import_module("src.observability")
    assert importlib.import_module("src.mcp_server")

@pytest.mark.unit
def test_main_import():
    """Test that main module can be imported."""
    assert importlib.import_module("src.main")
