from unittest.mock import MagicMock, patch

import pytest

from src.core.settings import EmbeddingSettings, Settings
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.openai_embedding import OpenAIEmbedding


@pytest.fixture
def mock_openai_client():
    with patch("src.libs.embedding.openai_embedding.openai.OpenAI") as mock:
        yield mock


@pytest.fixture
def mock_settings():
    settings = MagicMock(spec=Settings)
    settings.embedding = MagicMock(spec=EmbeddingSettings)
    settings.embedding.provider = "openai"
    settings.embedding.model = "text-embedding-3-small"
    settings.embedding.api_key = "sk-test"
    settings.embedding.base_url = None
    return settings


class TestOpenAIEmbedding:
    def test_init(self, mock_openai_client):
        """Test initialization of OpenAIEmbedding"""
        embedding = OpenAIEmbedding(api_key="sk-test", model="text-embedding-3-small")

        mock_openai_client.assert_called_once()
        assert embedding.model == "text-embedding-3-small"

    def test_embed_success(self, mock_openai_client):
        """Test successful embedding generation"""
        # Setup mock response
        mock_instance = mock_openai_client.return_value
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_instance.embeddings.create.return_value = mock_response

        embedding = OpenAIEmbedding(api_key="sk-test")
        result = embedding.embed(["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

        mock_instance.embeddings.create.assert_called_once_with(
            input=["hello", "world"], model="text-embedding-3-small"
        )

    def test_embed_empty_input(self, mock_openai_client):
        """Test embedding with empty input list"""
        embedding = OpenAIEmbedding(api_key="sk-test")
        result = embedding.embed([])
        assert result == []
        mock_openai_client.return_value.embeddings.create.assert_not_called()

    def test_embed_api_error(self, mock_openai_client):
        """Test handling of API errors"""
        import openai

        mock_instance = mock_openai_client.return_value
        mock_instance.embeddings.create.side_effect = openai.APIConnectionError(
            message="Connection failed", request=MagicMock()
        )

        embedding = OpenAIEmbedding(api_key="sk-test")

        with pytest.raises(RuntimeError) as exc:
            embedding.embed(["test"])
        assert "Failed to connect" in str(exc.value)


class TestEmbeddingFactory:
    def test_create_openai_provider(self, mock_settings):
        """Test creating OpenAI provider via factory"""
        mock_settings.embedding.provider = "openai"
        mock_settings.embedding.api_key = "sk-test"

        with patch("src.libs.embedding.embedding_factory.OpenAIEmbedding") as MockClass:
            EmbeddingFactory.create(mock_settings)
            MockClass.assert_called_once_with(
                api_key="sk-test", model="text-embedding-3-small"
            )

    def test_create_unknown_provider(self, mock_settings):
        """Test error for unknown provider"""
        mock_settings.embedding.provider = "unknown"

        with pytest.raises(ValueError) as exc:
            EmbeddingFactory.create(mock_settings)
        assert "Unknown embedding provider" in str(exc.value)

    def test_create_openai_missing_key(self, mock_settings):
        """Test error when API key is missing for OpenAI"""
        mock_settings.embedding.provider = "openai"
        mock_settings.embedding.api_key = None

        with pytest.raises(ValueError) as exc:
            EmbeddingFactory.create(mock_settings)
        assert "requires api_key" in str(exc.value)
