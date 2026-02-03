import pytest
from src.main import main

def test_main_import():
    """Test that main module can be imported"""
    assert main is not None

def test_main_execution(capsys):
    """Test main execution output"""
    main()
    captured = capsys.readouterr()
    assert "Modular RAG MCP Server Initialized" in captured.out
