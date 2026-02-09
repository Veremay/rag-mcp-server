from unittest.mock import Mock

import pytest

from src.core.settings import LLMSettings, Settings
from src.libs.llm.base_llm import BaseLLM
from src.libs.llm.llm_factory import LLMFactory


# Mock Implementations for Testing
class MockOpenAILLM(BaseLLM):
    def __init__(self, settings):
        self.model = settings.model

    def chat(self, messages, **kwargs):
        return "mock_openai_response"


class MockAzureLLM(BaseLLM):
    def __init__(self, settings):
        self.model = settings.model
        self.endpoint = settings.azure_endpoint

    def chat(self, messages, **kwargs):
        return "mock_azure_response"


@pytest.fixture
def mock_settings():
    settings = Mock(spec=Settings)
    settings.llm = Mock(spec=LLMSettings)
    # Default valid settings
    settings.llm.provider = "mock_openai"
    settings.llm.model = "gpt-4-test"
    settings.llm.azure_endpoint = None
    return settings


def test_factory_registration():
    # Clear registry for isolation
    LLMFactory._registry = {}

    LLMFactory.register("mock_openai", MockOpenAILLM)
    assert "mock_openai" in LLMFactory._registry
    assert LLMFactory._registry["mock_openai"] == MockOpenAILLM


def test_factory_create_success(mock_settings):
    # Setup registry
    LLMFactory._registry = {}
    LLMFactory.register("mock_openai", MockOpenAILLM)

    # Test creation
    llm = LLMFactory.create(mock_settings)
    assert isinstance(llm, MockOpenAILLM)
    assert llm.model == "gpt-4-test"


def test_factory_create_unknown_provider(mock_settings):
    # Setup registry
    LLMFactory._registry = {}

    mock_settings.llm.provider = "unknown_provider"

    with pytest.raises(ValueError) as excinfo:
        LLMFactory.create(mock_settings)
    assert "Unknown LLM provider" in str(excinfo.value)


def test_factory_switching_providers(mock_settings):
    # Setup registry
    LLMFactory._registry = {}
    LLMFactory.register("mock_openai", MockOpenAILLM)
    LLMFactory.register("mock_azure", MockAzureLLM)

    # Create OpenAI
    mock_settings.llm.provider = "mock_openai"
    llm1 = LLMFactory.create(mock_settings)
    assert isinstance(llm1, MockOpenAILLM)

    # Create Azure
    mock_settings.llm.provider = "mock_azure"
    mock_settings.llm.azure_endpoint = "https://test.azure.com"
    llm2 = LLMFactory.create(mock_settings)
    assert isinstance(llm2, MockAzureLLM)
    assert llm2.endpoint == "https://test.azure.com"
