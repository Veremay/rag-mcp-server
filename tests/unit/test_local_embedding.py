from unittest.mock import MagicMock

import pytest

from src.libs.embedding.local_embedding import LocalEmbedding


def test_local_embedding_initialization():
    """Test that LocalEmbedding initializes with correct parameters."""
    embedder = LocalEmbedding(model="test-model", dimension=10)
    assert embedder.model == "test-model"
    assert embedder.dimension == 10

    # Test defaults
    embedder_default = LocalEmbedding()
    assert embedder_default.dimension == 1536


def test_local_embedding_embed():
    """Test synchronous embedding generation."""
    embedder = LocalEmbedding(dimension=10)
    texts = ["hello", "world"]
    embeddings = embedder.embed(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 10
    assert len(embeddings[1]) == 10
    # Check values are floats
    assert isinstance(embeddings[0][0], float)


@pytest.mark.asyncio
async def test_local_embedding_aembed():
    """Test asynchronous embedding generation."""
    embedder = LocalEmbedding(dimension=10)
    texts = ["async", "test"]
    embeddings = await embedder.aembed(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 10


def test_local_embedding_deterministic():
    """Test that embeddings are deterministic for the same input."""
    embedder = LocalEmbedding(dimension=10)
    text = "consistent text"
    emb1 = embedder.embed([text])[0]
    emb2 = embedder.embed([text])[0]

    assert emb1 == emb2

    # Different text should produce different embedding (with high probability)
    text2 = "different text"
    emb3 = embedder.embed([text2])[0]
    assert emb1 != emb3


def test_local_embedding_empty_input():
    """Test behavior with empty input list."""
    embedder = LocalEmbedding()
    embeddings = embedder.embed([])
    assert embeddings == []
