from unittest.mock import MagicMock

import pytest

from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.models import Chunk


def test_batch_processor_splits_into_stable_batches_and_preserves_order():
    chunks = [Chunk(text=t) for t in ["a", "b", "c", "d", "e"]]

    dense = MagicMock()
    sparse = MagicMock()

    def dense_encode(batch, trace=None):
        return [[ord(c.text)] for c in batch]

    def sparse_encode(batch, trace=None):
        return [{"term": c.text} for c in batch]

    dense.encode.side_effect = dense_encode
    sparse.encode.side_effect = sparse_encode

    processor = BatchProcessor(batch_size=2)
    result = processor.process(chunks, dense_encoder=dense, sparse_encoder=sparse)

    assert [m.size for m in result.batches] == [2, 2, 1]
    assert result.dense_vectors == [[97], [98], [99], [100], [101]]
    assert result.sparse_vectors == [
        {"term": "a"},
        {"term": "b"},
        {"term": "c"},
        {"term": "d"},
        {"term": "e"},
    ]

    assert dense.encode.call_count == 3
    assert sparse.encode.call_count == 3
    assert [call.args[0] for call in dense.encode.call_args_list] == [
        chunks[0:2],
        chunks[2:4],
        chunks[4:5],
    ]


def test_batch_processor_empty_input_returns_empty():
    dense = MagicMock()
    sparse = MagicMock()

    processor = BatchProcessor(batch_size=2)
    result = processor.process([], dense_encoder=dense, sparse_encoder=sparse)

    assert result.dense_vectors == []
    assert result.sparse_vectors == []
    assert result.batches == []
    dense.encode.assert_not_called()
    sparse.encode.assert_not_called()


def test_batch_processor_rejects_non_positive_batch_size():
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        BatchProcessor(batch_size=0)
