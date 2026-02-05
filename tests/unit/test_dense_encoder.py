from unittest.mock import MagicMock

import pytest

from src.core.settings import Settings
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.models import Chunk
from src.libs.embedding.base_embedding import BaseEmbedding


def test_dense_encoder_outputs_count_and_dimension():
    settings = MagicMock(spec=Settings)
    embedding = MagicMock(spec=BaseEmbedding)
    embedding.embed.return_value = [[0.0, 1.0], [2.0, 3.0]]

    encoder = DenseEncoder(settings, embedding=embedding)
    chunks = [Chunk(text="a"), Chunk(text="b")]

    vectors = encoder.encode(chunks)

    assert len(vectors) == 2
    assert all(len(v) == 2 for v in vectors)
    assert embedding.embed.call_args.args[0] == ["a", "b"]


def test_dense_encoder_empty_input():
    settings = MagicMock(spec=Settings)
    embedding = MagicMock(spec=BaseEmbedding)

    encoder = DenseEncoder(settings, embedding=embedding)
    vectors = encoder.encode([])

    assert vectors == []
    embedding.embed.assert_not_called()


def test_dense_encoder_raises_on_count_mismatch():
    settings = MagicMock(spec=Settings)
    embedding = MagicMock(spec=BaseEmbedding)
    embedding.embed.return_value = [[0.0, 1.0]]

    encoder = DenseEncoder(settings, embedding=embedding)
    chunks = [Chunk(text="a"), Chunk(text="b")]

    with pytest.raises(ValueError, match="vector count mismatch"):
        encoder.encode(chunks)


def test_dense_encoder_raises_on_dimension_mismatch():
    settings = MagicMock(spec=Settings)
    embedding = MagicMock(spec=BaseEmbedding)
    embedding.embed.return_value = [[0.0, 1.0], [2.0]]

    encoder = DenseEncoder(settings, embedding=embedding)
    chunks = [Chunk(text="a"), Chunk(text="b")]

    with pytest.raises(ValueError, match="vector dimension mismatch"):
        encoder.encode(chunks)
