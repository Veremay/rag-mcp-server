import os
import shutil
import tempfile
from dataclasses import dataclass

import pytest

pytest.importorskip("chromadb")

from src.libs.vector_store.base_vector_store import VectorRecord
from src.libs.vector_store.chroma_store import ChromaStore


# Mocking parts of settings to avoid full object construction overhead/complexity
@dataclass
class MockVectorStoreSettings:
    backend: str
    persist_path: str
    collection_name: str = "test_collection"


@dataclass
class MockSettings:
    vector_store: MockVectorStoreSettings


@pytest.fixture
def temp_chroma_settings():
    """Create settings with a temporary directory for ChromaDB."""
    temp_dir = tempfile.mkdtemp()
    settings = MockSettings(
        vector_store=MockVectorStoreSettings(
            backend="chroma", persist_path=temp_dir, collection_name="test_collection"
        )
    )
    yield settings
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.mark.integration
def test_chroma_store_roundtrip(temp_chroma_settings):
    """Test full roundtrip: upsert -> query."""
    store = ChromaStore(temp_chroma_settings)

    # 1. Create test records
    # Note: Dimension must match what Chroma expects if it enforces it,
    # but for default cosine it usually adapts to the first insertion.
    records = [
        VectorRecord(
            id="1",
            embedding=[0.1, 0.2, 0.3],
            content="Hello world",
            metadata={"source": "test", "type": "A"},
        ),
        VectorRecord(
            id="2",
            embedding=[0.9, 0.8, 0.7],
            content="Another document",
            metadata={"source": "test", "type": "B"},
        ),
    ]

    # 2. Upsert
    store.upsert(records)

    # 3. Query exact match
    results = store.query(vector=[0.1, 0.2, 0.3], top_k=1)

    assert len(results) == 1
    assert results[0].id == "1"
    assert results[0].content == "Hello world"
    assert results[0].metadata["source"] == "test"

    results_no_filter = store.query(
        vector=[0.1, 0.2, 0.3],
        top_k=2,
        filters={},
    )
    assert len(results_no_filter) >= 1

    # 4. Query with filter
    results_filtered = store.query(
        vector=[0.1, 0.2, 0.3], top_k=2, filters={"type": "A"}
    )
    assert len(results_filtered) == 1
    assert results_filtered[0].id == "1"

    # 5. Query with non-matching filter
    results_empty = store.query(
        vector=[0.1, 0.2, 0.3], top_k=2, filters={"source": "other"}
    )
    assert len(results_empty) == 0


@pytest.mark.integration
def test_chroma_store_idempotency(temp_chroma_settings):
    """Test that upserting the same record twice doesn't duplicate it."""
    store = ChromaStore(temp_chroma_settings)

    record = VectorRecord(
        id="1",
        embedding=[0.5, 0.5, 0.5],
        content="Idempotency test",
        metadata={"version": 1},
    )

    # First upsert
    store.upsert([record])

    # Verify count
    results = store.query(vector=[0.5, 0.5, 0.5], top_k=10)
    assert len(results) == 1
    assert results[0].metadata["version"] == 1

    # Update record
    record.metadata["version"] = 2
    record.content = "Updated content"

    # Second upsert (should update)
    store.upsert([record])

    # Verify update
    results = store.query(vector=[0.5, 0.5, 0.5], top_k=10)
    assert len(results) == 1
    assert results[0].metadata["version"] == 2
    assert results[0].content == "Updated content"
