from unittest.mock import MagicMock

import pytest

from src.core.settings import LLMSettings, Settings
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.ollama_llm import OllamaLLM


class TestLLMFactoryOllama:

    def test_create_ollama_default(self):
        """Test creating Ollama LLM with minimal settings."""
        settings = MagicMock(spec=Settings)
        settings.llm = LLMSettings(provider="ollama", model="llama3")

        llm = LLMFactory.create(settings)

        assert isinstance(llm, OllamaLLM)
        assert llm.model == "llama3"
        # Check defaults - OpenAI client usually normalizes URL
        # We check if the default base_url we set in OllamaLLM is used
        assert "localhost:11434" in str(llm.client.base_url)

    def test_create_ollama_custom(self):
        """Test creating Ollama LLM with custom settings."""
        settings = MagicMock(spec=Settings)
        settings.llm = LLMSettings(
            provider="ollama",
            model="mistral",
            base_url="http://custom:1234/v1",
            api_key="secret",
        )

        llm = LLMFactory.create(settings)

        assert isinstance(llm, OllamaLLM)
        assert llm.model == "mistral"
        assert "custom:1234" in str(llm.client.base_url)
        assert llm.client.api_key == "secret"
