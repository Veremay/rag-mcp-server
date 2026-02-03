import pytest
from unittest.mock import MagicMock
from src.core.settings import Settings, VectorStoreSettings
from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

class MockVectorStore(BaseVectorStore):
    """Mock implementation for contract testing."""
    def __init__(self):
        self.store = {}

    def upsert(self, records, trace=None):
        for record in records:
            self.store[record.id] = record

    def query(self, vector, top_k, filters=None, trace=None):
        # Dummy implementation: return all values
        return list(self.store.values())[:top_k]

@pytest.fixture
def mock_settings():
    settings = MagicMock(spec=Settings)
    settings.vector_store = MagicMock(spec=VectorStoreSettings)
    return settings

def test_factory_unsupported_backend(mock_settings):
    """Test that factory raises ValueError for unknown backend."""
    mock_settings.vector_store.backend = "unknown_db"
    
    with pytest.raises(ValueError, match="Unsupported vector store backend"):
        VectorStoreFactory.create(mock_settings)

def test_factory_chroma_pending(mock_settings):
    """Test that factory raises NotImplementedError for chroma (pending B7.6)."""
    mock_settings.vector_store.backend = "chroma"
    
    with pytest.raises(NotImplementedError, match="ChromaStore implementation is pending"):
        VectorStoreFactory.create(mock_settings)

def test_vector_record_structure():
    """Test VectorRecord dataclass structure."""
    record = VectorRecord(
        id="123",
        embedding=[0.1, 0.2, 0.3],
        content="test content",
        metadata={"source": "doc1"}
    )
    
    assert record.id == "123"
    assert len(record.embedding) == 3
    assert record.metadata["source"] == "doc1"

def test_base_vector_store_contract():
    """Test that a concrete implementation works as expected."""
    store = MockVectorStore()
    
    record = VectorRecord(
        id="1",
        embedding=[0.1, 0.1],
        content="test",
        metadata={}
    )
    
    store.upsert([record])
    results = store.query([0.1, 0.1], top_k=1)
    
    assert len(results) == 1
    assert results[0].content == "test"
