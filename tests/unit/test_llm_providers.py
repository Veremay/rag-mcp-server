from unittest.mock import MagicMock, patch

import pytest

from src.core.settings import LLMSettings, Settings
from src.libs.llm.azure_llm import AzureOpenAILLM
from src.libs.llm.deepseek_llm import DeepSeekLLM
from src.libs.llm.llm_factory import LLMFactory
from src.libs.llm.openai_llm import OpenAILLM

# --- Fixtures ---


@pytest.fixture
def mock_openai_client():
    with patch("src.libs.llm.openai_llm.openai.OpenAI") as mock:
        yield mock


@pytest.fixture
def mock_azure_client():
    with patch("src.libs.llm.azure_llm.openai.AzureOpenAI") as mock:
        yield mock


@pytest.fixture
def settings():
    s = MagicMock(spec=Settings)
    s.llm = MagicMock(spec=LLMSettings)
    # Default values
    s.llm.api_key = "test-key"
    s.llm.model = "test-model"
    s.llm.base_url = None
    s.llm.azure_endpoint = None
    return s


# --- Tests ---


def test_create_openai_llm(settings, mock_openai_client):
    settings.llm.provider = "openai"
    llm = LLMFactory.create(settings)

    assert isinstance(llm, OpenAILLM)
    mock_openai_client.assert_called_once()
    args, kwargs = mock_openai_client.call_args
    assert kwargs["api_key"] == "test-key"


def test_create_azure_llm(settings, mock_azure_client):
    settings.llm.provider = "azure"
    settings.llm.azure_endpoint = "https://test.azure.com"
    llm = LLMFactory.create(settings)

    assert isinstance(llm, AzureOpenAILLM)
    mock_azure_client.assert_called_once()
    args, kwargs = mock_azure_client.call_args
    assert kwargs["azure_endpoint"] == "https://test.azure.com"


def test_create_deepseek_llm(settings, mock_openai_client):
    settings.llm.provider = "deepseek"
    llm = LLMFactory.create(settings)

    assert isinstance(llm, DeepSeekLLM)
    # DeepSeek inherits from OpenAILLM, so it uses openai.OpenAI
    mock_openai_client.assert_called_once()
    args, kwargs = mock_openai_client.call_args
    assert kwargs["base_url"] == "https://api.deepseek.com"


def test_openai_chat_calls_client(settings, mock_openai_client):
    settings.llm.provider = "openai"
    llm = LLMFactory.create(settings)

    # Mock response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Hello world"
    llm.client.chat.completions.create.return_value = mock_response

    response = llm.chat([{"role": "user", "content": "Hi"}])

    assert response == "Hello world"
    llm.client.chat.completions.create.assert_called_once()


def test_factory_missing_api_key_openai(settings):
    settings.llm.provider = "openai"
    settings.llm.api_key = None

    with pytest.raises(ValueError, match="OpenAI provider requires api_key"):
        LLMFactory.create(settings)


def test_factory_unknown_provider(settings):
    settings.llm.provider = "unknown"

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        LLMFactory.create(settings)
