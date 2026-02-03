import pytest
from typing import List, Any
from unittest.mock import MagicMock
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.core.settings import Settings, EmbeddingSettings

# --- Fake Implementation for Testing ---
class FakeEmbedding(BaseEmbedding):
    def __init__(self, settings: Settings):
        self.settings = settings

    def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        # Return a fixed mock vector for each text
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def aembed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

# --- Fixtures ---
@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the factory registry before each test."""
    # Store original registry
    original_registry = EmbeddingFactory._registry.copy()
    EmbeddingFactory._registry.clear()
    yield
    # Restore original registry
    EmbeddingFactory._registry = original_registry

@pytest.fixture
def mock_settings():
    settings = MagicMock(spec=Settings)
    settings.embedding = MagicMock(spec=EmbeddingSettings)
    return settings

# --- Tests ---

def test_factory_registration():
    """Test that providers can be registered."""
    EmbeddingFactory.register("fake", FakeEmbedding)
    assert "fake" in EmbeddingFactory._registry
    assert EmbeddingFactory._registry["fake"] == FakeEmbedding

def test_factory_create_success(mock_settings):
    """Test creating an instance for a registered provider."""
    EmbeddingFactory.register("fake", FakeEmbedding)
    mock_settings.embedding.provider = "fake"
    
    instance = EmbeddingFactory.create(mock_settings)
    
    assert isinstance(instance, FakeEmbedding)
    assert instance.settings == mock_settings

def test_factory_create_unknown_provider(mock_settings):
    """Test that unknown providers raise a ValueError."""
    mock_settings.embedding.provider = "unknown_provider"
    
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        EmbeddingFactory.create(mock_settings)

def test_factory_case_insensitive(mock_settings):
    """Test that provider names are case-insensitive."""
    EmbeddingFactory.register("fake", FakeEmbedding)
    mock_settings.embedding.provider = "FAKE"
    
    instance = EmbeddingFactory.create(mock_settings)
    assert isinstance(instance, FakeEmbedding)

def test_base_embedding_interface():
    """Test that BaseEmbedding cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseEmbedding()
