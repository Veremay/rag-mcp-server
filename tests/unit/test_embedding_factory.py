from unittest.mock import MagicMock

import pytest

from src.core.settings import EmbeddingSettings, Settings
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.local_embedding import LocalEmbedding
from src.libs.embedding.openai_embedding import OpenAIEmbedding


@pytest.fixture
def mock_settings():
    settings = MagicMock(spec=Settings)
    settings.embedding = MagicMock(spec=EmbeddingSettings)
    # Default values
    settings.embedding.base_url = None
    settings.embedding.api_key = "test-key"
    return settings


def test_factory_create_openai(mock_settings):
    """Test creating OpenAI embedding instance."""
    mock_settings.embedding.provider = "openai"
    mock_settings.embedding.model = "text-embedding-3-small"

    instance = EmbeddingFactory.create(mock_settings)

    assert isinstance(instance, OpenAIEmbedding)
    assert instance.model == "text-embedding-3-small"


def test_factory_create_openai_with_base_url(mock_settings):
    mock_settings.embedding.provider = "openai"
    mock_settings.embedding.model = "text-embedding-v4"
    mock_settings.embedding.api_key = "test-key"
    mock_settings.embedding.base_url = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    from unittest.mock import patch

    with patch("src.libs.embedding.embedding_factory.OpenAIEmbedding") as MockEmbedding:
        EmbeddingFactory.create(mock_settings)
        MockEmbedding.assert_called_once_with(
            api_key="test-key",
            model="text-embedding-v4",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )


def test_factory_create_openai_missing_key(mock_settings):
    """Test OpenAI provider raises error without api_key."""
    mock_settings.embedding.provider = "openai"
    mock_settings.embedding.api_key = None

    with pytest.raises(ValueError, match="requires api_key"):
        EmbeddingFactory.create(mock_settings)


def test_factory_create_local(mock_settings):
    """Test creating Local embedding instance."""
    mock_settings.embedding.provider = "local"
    mock_settings.embedding.model = "local-model"

    instance = EmbeddingFactory.create(mock_settings)

    assert isinstance(instance, LocalEmbedding)
    assert instance.model == "local-model"


def test_factory_create_unknown_provider(mock_settings):
    """Test that unknown providers raise a ValueError."""
    mock_settings.embedding.provider = "unknown_provider"

    with pytest.raises(ValueError, match="Unknown embedding provider"):
        EmbeddingFactory.create(mock_settings)


def test_factory_case_insensitive(mock_settings):
    """Test that provider names are case-insensitive."""
    mock_settings.embedding.provider = "LOCAL"
    mock_settings.embedding.model = "local-model"

    instance = EmbeddingFactory.create(mock_settings)
    assert isinstance(instance, LocalEmbedding)


def test_base_embedding_interface():
    """Test that BaseEmbedding cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseEmbedding()
