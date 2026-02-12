import pytest
from unittest.mock import Mock, call
from pathlib import Path
from src.ingestion.pipeline import IngestionPipeline
from src.core.settings import Settings
from src.ingestion.models import Document, Chunk

@pytest.fixture
def mock_settings():
    settings = Mock(spec=Settings)
    # Setup minimal settings structure for pipeline init
    settings.ingestion = Mock()
    settings.ingestion.splitter = Mock()
    settings.ingestion.splitter.provider = "test_splitter"
    settings.embedding = Mock()
    settings.embedding.model = "test_model"
    settings.vector_store = Mock()
    settings.vector_store.backend = "test_backend"
    return settings

@pytest.fixture
def mock_loader():
    loader = Mock()
    loader.load.return_value = Document(text="test text", id="doc1", metadata={})
    return loader

@pytest.fixture
def mock_splitter():
    splitter = Mock()
    # Mock splitting into 2 chunks
    splitter.split_text.return_value = ["chunk1", "chunk2"]
    return splitter

def test_pipeline_calls_on_progress_callback(mock_settings, mock_loader, tmp_path):
    # Setup dependencies
    mock_integrity = Mock()
    mock_integrity.compute_sha256.return_value = "hash123"
    mock_integrity.should_skip.return_value = False
    
    # Create dummy file
    test_file = tmp_path / "test.txt"
    test_file.write_text("dummy")
    
    # Mock progress callback
    on_progress = Mock()
    
    pipeline = IngestionPipeline(
        mock_settings,
        integrity=mock_integrity,
        loader=mock_loader,
        transforms=[],  # Empty transforms for simplicity
        dense_encoder=Mock(),
        sparse_encoder=Mock(),
        batch_processor=Mock(),
        vector_store=Mock(),
        bm25_indexer=Mock(),
        image_storage=Mock()
    )
    
    # Mock split method to avoid factory dependency
    pipeline.split = Mock()
    pipeline.split.return_value.chunks = [
        Chunk(text="c1", doc_id="d1", metadata={"chunk_index": 0}),
        Chunk(text="c2", doc_id="d1", metadata={"chunk_index": 1})
    ]
    
    # Mock internal methods to avoid complex dependencies
    pipeline._encode = Mock()
    pipeline._encode.return_value.dense_vectors = [[0.1], [0.2]]
    pipeline._encode.return_value.sparse_vectors = [{}, {}]
    
    pipeline._upsert = Mock()
    pipeline._upsert.return_value.records = [Mock(id="c1"), Mock(id="c2")]
    
    pipeline._build_bm25 = Mock()
    # Mock return value of bm25 build
    mock_bm25 = Mock()
    mock_bm25.terms = ["term1", "term2"]
    mock_bm25.postings = {"term1": [1], "term2": [2]}
    pipeline._build_bm25.return_value = mock_bm25
    
    pipeline._store_images = Mock()
    
    # Run ingest
    pipeline.ingest(
        collection="test_coll",
        file_path=test_file,
        on_progress=on_progress
    )
    
    # Verify calls
    # We expect calls for: start, hash_calculated, loaded, split, transformed, encoded, upserted, bm25_built, complete
    assert on_progress.call_count >= 8
    
    # call_args_list entries are (args, kwargs) tuples
    # args[0] is stage name, args[1] is data dict
    calls = [call.args[0] for call in on_progress.call_args_list]
    expected_stages = [
        "start", 
        "hash_calculated", 
        "loaded", 
        "split", 
        "transformed", 
        "encoded", 
        "upserted", 
        "bm25_built",
        "complete"
    ]
    
    for stage in expected_stages:
        assert stage in calls, f"Missing progress callback for stage: {stage}"
        
    # Verify call arguments structure
    # Check "split" call specifically
    split_call = next(call for call in on_progress.call_args_list if call.args[0] == "split")
    assert "chunks" in split_call.args[1]
    assert len(split_call.args[1]["chunks"]) == 2

def test_pipeline_on_progress_none_safe(mock_settings, mock_loader, tmp_path):
    """Verify pipeline runs fine when on_progress is None"""
    mock_integrity = Mock()
    mock_integrity.compute_sha256.return_value = "hash123"
    mock_integrity.should_skip.return_value = False
    
    test_file = tmp_path / "test.txt"
    test_file.write_text("dummy")
    
    pipeline = IngestionPipeline(
        mock_settings,
        integrity=mock_integrity,
        loader=mock_loader
    )
    
    # Mock out heavy lifting
    pipeline.split = Mock()
    pipeline.split.return_value.chunks = []
    pipeline._encode = Mock()
    pipeline._encode.return_value.dense_vectors = []
    pipeline._encode.return_value.sparse_vectors = []
    pipeline._upsert = Mock()
    pipeline._upsert.return_value.records = []
    pipeline._build_bm25 = Mock()
    mock_bm25 = Mock()
    mock_bm25.terms = []
    mock_bm25.postings = []
    pipeline._build_bm25.return_value = mock_bm25
    pipeline._store_images = Mock()
    
    # Should not raise error
    pipeline.ingest(
        collection="test_coll",
        file_path=test_file,
        on_progress=None
    )
