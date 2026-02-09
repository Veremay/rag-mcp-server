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

from unittest.mock import MagicMock, patch

# ... (existing imports)

def test_factory_creates_chroma(mock_settings, monkeypatch):
    """Test that factory creates ChromaStore when backend is chroma."""
    monkeypatch.setenv("FORCE_CHROMA", "1")
    mock_settings.vector_store.backend = "chroma"
    mock_settings.vector_store.persist_path = "/tmp/test_chroma"
    mock_settings.vector_store.collection_name = "test"
    
    with patch("src.libs.vector_store.chroma_store.ChromaStore") as MockChromaStore:
        store = VectorStoreFactory.create(mock_settings)
        
        MockChromaStore.assert_called_once_with(mock_settings)
        assert store == MockChromaStore.return_value

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
