from typing import Any, List, Optional

import pytest

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

    def split_text(
        self, text: str, trace: Optional[Any] = None, **kwargs: Any
    ) -> List[str]:
        return [text]


@pytest.fixture(autouse=True)
def restore_registry():
    """Save and restore the registry state."""
    original_registry = SplitterFactory._registry.copy()
    yield
    SplitterFactory._registry = original_registry


def test_factory_registration():
    # We can add to the registry without clearing it
    # Or if we want to test empty, we can mock it
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
    # Don't clear, just use a definitely unknown name
    settings = MockSettings(provider="unknown_provider_xyz")

    with pytest.raises(ValueError, match="Unknown splitter provider"):
        SplitterFactory.create(settings)


def test_factory_case_insensitivity():
    SplitterFactory.register("fake", FakeSplitter)
    # Should work even if config is uppercase "FAKE"
    settings = MockSettings(provider="FAKE")

    splitter = SplitterFactory.create(settings)
    assert isinstance(splitter, FakeSplitter)
