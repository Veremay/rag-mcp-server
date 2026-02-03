import pytest
from typing import List, Optional, Any
from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import SplitterFactory

# Mock classes to simulate Settings structure without full initialization
class MockSplitterSettings:
    def __init__(self, provider: str):
        self.provider = provider
        self.chunk_size = 1000
        self.chunk_overlap = 200

class MockIngestionSettings:
    def __init__(self, provider: str):
        self.splitter = MockSplitterSettings(provider)

class MockSettings:
    def __init__(self, provider: str = "fake"):
        self.ingestion = MockIngestionSettings(provider)

class FakeSplitter(BaseSplitter):
    def __init__(self, settings: Any):
        self.settings = settings
        
    def split_text(self, text: str, trace: Optional[Any] = None, **kwargs: Any) -> List[str]:
        return [text]

def test_factory_registration():
    # Clear registry to avoid side effects (though strictly shouldn't affect if keys differ)
    SplitterFactory._registry.clear()
    
    SplitterFactory.register("fake", FakeSplitter)
    assert "fake" in SplitterFactory._registry
    assert SplitterFactory._registry["fake"] == FakeSplitter

def test_factory_create_success():
    SplitterFactory.register("fake", FakeSplitter)
    settings = MockSettings(provider="fake")
    
    splitter = SplitterFactory.create(settings)
    
    assert isinstance(splitter, FakeSplitter)
    # Check if settings were passed correctly
    assert splitter.settings == settings

def test_factory_unknown_provider():
    SplitterFactory._registry.clear()
    settings = MockSettings(provider="unknown_provider")
    
    with pytest.raises(ValueError, match="Unknown splitter provider"):
        SplitterFactory.create(settings)

def test_factory_case_insensitivity():
    SplitterFactory.register("fake", FakeSplitter)
    # Should work even if config is uppercase "FAKE"
    settings = MockSettings(provider="FAKE")
    
    splitter = SplitterFactory.create(settings)
    assert isinstance(splitter, FakeSplitter)
