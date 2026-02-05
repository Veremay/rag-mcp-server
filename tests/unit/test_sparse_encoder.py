from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.models import Chunk


def test_sparse_encoder_outputs_term_weights():
    encoder = SparseEncoder()
    chunks = [Chunk(text="Hello, hello world!"), Chunk(text="world")]

    outputs = encoder.encode(chunks)

    assert len(outputs) == 2
    assert outputs[0]["hello"] == 2.0
    assert outputs[0]["world"] == 1.0
    assert outputs[1]["world"] == 1.0


def test_sparse_encoder_empty_text_has_clear_behavior():
    encoder = SparseEncoder()
    chunks = [Chunk(text=""), Chunk(text="   ")]

    outputs = encoder.encode(chunks)

    assert outputs == [{}, {}]


def test_sparse_encoder_empty_chunks_returns_empty_list():
    encoder = SparseEncoder()
    assert encoder.encode([]) == []
