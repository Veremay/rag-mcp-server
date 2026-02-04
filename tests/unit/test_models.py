import pytest
from src.ingestion.models import Document, Chunk

class TestDocument:
    def test_document_initialization(self):
        doc = Document(text="Hello world", metadata={"source": "test"})
        assert doc.text == "Hello world"
        assert doc.metadata == {"source": "test"}
        assert doc.id is not None
        assert isinstance(doc.id, str)

    def test_document_defaults(self):
        doc = Document(text="Just text")
        assert doc.metadata == {}
        assert doc.id is not None

    def test_document_serialization(self):
        doc = Document(id="123", text="Serialize me", metadata={"key": "value"})
        data = doc.model_dump()
        assert data["id"] == "123"
        assert data["text"] == "Serialize me"
        assert data["metadata"] == {"key": "value"}
        
        json_str = doc.model_dump_json()
        assert "Serialize me" in json_str

class TestChunk:
    def test_chunk_initialization(self):
        chunk = Chunk(
            text="Chunk text",
            doc_id="doc_1",
            start_char_idx=0,
            end_char_idx=10
        )
        assert chunk.text == "Chunk text"
        assert chunk.doc_id == "doc_1"
        assert chunk.start_char_idx == 0
        assert chunk.end_char_idx == 10
        assert chunk.id is not None

    def test_chunk_defaults(self):
        chunk = Chunk(text="Standalone chunk")
        assert chunk.doc_id is None
        assert chunk.metadata == {}

    def test_chunk_serialization(self):
        chunk = Chunk(
            id="chunk_1",
            text="Content",
            doc_id="doc_1",
            metadata={"score": 0.9}
        )
        data = chunk.model_dump()
        assert data["id"] == "chunk_1"
        assert data["doc_id"] == "doc_1"
        assert data["metadata"] == {"score": 0.9}
