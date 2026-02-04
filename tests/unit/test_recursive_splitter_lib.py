import pytest
import importlib.util
from unittest.mock import MagicMock

if importlib.util.find_spec("langchain_text_splitters") is None:
    pytest.skip("langchain_text_splitters is not installed", allow_module_level=True)

from src.libs.splitter.recursive_splitter import RecursiveSplitter
from src.libs.splitter.splitter_factory import SplitterFactory
from src.core.settings import Settings, IngestionSettings, SplitterSettings

@pytest.fixture
def mock_settings():
    settings = MagicMock(spec=Settings)
    settings.ingestion = MagicMock(spec=IngestionSettings)
    settings.ingestion.splitter = MagicMock(spec=SplitterSettings)
    settings.ingestion.splitter.provider = "recursive"
    settings.ingestion.splitter.chunk_size = 100
    settings.ingestion.splitter.chunk_overlap = 20
    return settings

def test_recursive_splitter_initialization(mock_settings):
    splitter = RecursiveSplitter(mock_settings)
    assert splitter.chunk_size == 100
    assert splitter.chunk_overlap == 20

def test_recursive_splitter_factory_integration(mock_settings):
    """Verify that the factory can create the recursive splitter"""
    # Ensure it's registered (should be by default in module scope)
    assert "recursive" in SplitterFactory._registry
    
    splitter = SplitterFactory.create(mock_settings)
    assert isinstance(splitter, RecursiveSplitter)
    assert splitter.chunk_size == 100

def test_recursive_splitter_split_text(mock_settings):
    splitter = RecursiveSplitter(mock_settings)
    text = "This is a long text " * 10  # 200 chars
    chunks = splitter.split_text(text)
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 100

def test_recursive_splitter_markdown_handling(mock_settings):
    """Verify that it handles markdown headers reasonably well (via standard separators)"""
    mock_settings.ingestion.splitter.chunk_size = 50
    splitter = RecursiveSplitter(mock_settings)
    
    text = "# Header\n\nBody text that is long enough to be split."
    chunks = splitter.split_text(text)
    
    # Just verify it splits and returns list of strings
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert "# Header" in chunks[0]

def test_recursive_splitter_empty_text(mock_settings):
    splitter = RecursiveSplitter(mock_settings)
    chunks = splitter.split_text("")
    assert chunks == []
