import json
from unittest.mock import MagicMock, Mock

import pytest

from src.ingestion.document_manager import DocumentManager
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.loader.file_integrity import FileIntegrityRegistry
from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord


@pytest.fixture
def mock_vector_store():
    store = Mock(spec=BaseVectorStore)
    store.collection_name = "test_coll"
    return store

@pytest.fixture
def mock_bm25():
    return Mock(spec=BM25Indexer)

@pytest.fixture
def mock_image_storage():
    return Mock(spec=ImageStorage)

@pytest.fixture
def mock_file_integrity():
    return Mock(spec=FileIntegrityRegistry)

@pytest.fixture
def document_manager(mock_vector_store, mock_bm25, mock_image_storage, mock_file_integrity):
    return DocumentManager(
        vector_store=mock_vector_store,
        bm25_indexer=mock_bm25,
        image_storage=mock_image_storage,
        file_integrity=mock_file_integrity
    )

def test_list_documents(document_manager, mock_vector_store):
    # Setup mock data
    mock_vector_store.get_records_by_metadata.return_value = [
        VectorRecord(id="1", content="a", embedding=[], metadata={"source_path": "file1.txt"}),
        VectorRecord(id="2", content="b", embedding=[], metadata={"source_path": "file1.txt", "images": '[{"image_id": "img1"}]'}),
        VectorRecord(id="3", content="c", embedding=[], metadata={"source_path": "file2.txt"}),
    ]

    docs = document_manager.list_documents(collection="test_coll")
    
    assert len(docs) == 2
    
    doc1 = next(d for d in docs if d.source_path == "file1.txt")
    assert doc1.chunk_count == 2
    assert doc1.image_count == 1
    
    doc2 = next(d for d in docs if d.source_path == "file2.txt")
    assert doc2.chunk_count == 1
    assert doc2.image_count == 0

def test_delete_document_success(document_manager, mock_vector_store, mock_bm25, mock_image_storage, mock_file_integrity):
    source = "file1.txt"
    mock_vector_store.get_records_by_metadata.return_value = [
        VectorRecord(id="1", content="a", embedding=[], metadata={"source_path": source, "images": '[{"image_id": "img1"}]'}),
    ]
    mock_image_storage.delete.return_value = True
    mock_file_integrity.remove_record.return_value = True

    result = document_manager.delete_document(source, "test_coll")

    assert result.success is True
    assert result.deleted_chunks == 1
    assert result.deleted_images == 1
    
    mock_image_storage.delete.assert_called_with(collection="test_coll", image_id="img1")
    mock_bm25.remove_document.assert_called_with(collection="test_coll", chunk_ids=["1"])
    mock_vector_store.delete_by_metadata.assert_called()
    mock_file_integrity.remove_record.assert_called_with(source)

def test_delete_document_not_found(document_manager, mock_vector_store, mock_file_integrity):
    mock_vector_store.get_records_by_metadata.return_value = []
    mock_file_integrity.remove_record.return_value = False
    
    result = document_manager.delete_document("missing.txt", "test_coll")
    
    assert result.success is False
    assert result.deleted_chunks == 0

def test_get_document_detail(document_manager, mock_vector_store):
    source = "file1.txt"
    mock_vector_store.get_records_by_metadata.return_value = [
        VectorRecord(id="1", content="content1", embedding=[], metadata={"source_path": source}),
    ]
    
    detail = document_manager.get_document_detail(source)
    
    assert detail is not None
    assert detail.source_path == source
    assert len(detail.chunks) == 1
    assert detail.chunks[0].content == "content1"

def test_get_collection_stats(document_manager, mock_vector_store):
    mock_vector_store.get_records_by_metadata.return_value = [
        VectorRecord(id="1", content="a", embedding=[], metadata={"source_path": "file1.txt"}),
    ]
    
    stats = document_manager.get_collection_stats("test_coll")
    
    assert stats.total_documents == 1
    assert stats.total_chunks == 1
