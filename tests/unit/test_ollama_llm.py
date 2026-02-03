import pytest
from unittest.mock import MagicMock, patch
from src.libs.llm.ollama_llm import OllamaLLM
import openai

class TestOllamaLLM:
    
    @patch("src.libs.llm.ollama_llm.openai.OpenAI")
    def test_initialization_defaults(self, mock_openai):
        """Test initialization with default values."""
        llm = OllamaLLM(model="llama3")
        
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["base_url"] == "http://localhost:11434/v1"
        assert call_kwargs["api_key"] == "ollama"
        assert llm.model == "llama3"

    @patch("src.libs.llm.ollama_llm.openai.OpenAI")
    def test_initialization_custom(self, mock_openai):
        """Test initialization with custom values."""
        llm = OllamaLLM(
            model="mistral",
            base_url="http://custom-host:11434/v1",
            api_key="custom-key"
        )
        
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["base_url"] == "http://custom-host:11434/v1"
        assert call_kwargs["api_key"] == "custom-key"
        assert llm.model == "mistral"

    @patch("src.libs.llm.ollama_llm.openai.OpenAI")
    def test_chat_success(self, mock_openai):
        """Test successful chat completion."""
        # Setup mock response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello from Ollama"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        llm = OllamaLLM(model="llama3")
        response = llm.chat([{"role": "user", "content": "Hi"}])
        
        assert response == "Hello from Ollama"
        mock_client.chat.completions.create.assert_called_once_with(
            model="llama3",
            messages=[{"role": "user", "content": "Hi"}]
        )

    @patch("src.libs.llm.ollama_llm.openai.OpenAI")
    def test_chat_api_error(self, mock_openai):
        """Test handling of API errors."""
        mock_client = MagicMock()
        # Mocking an APIError requires args usually, but we can just use the class
        mock_client.chat.completions.create.side_effect = openai.APIError("Ollama Error", request=MagicMock(), body={})
        mock_openai.return_value = mock_client
        
        llm = OllamaLLM(model="llama3")
        
        with pytest.raises(RuntimeError) as exc:
            llm.chat([{"role": "user", "content": "Hi"}])
        
        assert "Ollama API Error" in str(exc.value)

    @patch("src.libs.llm.ollama_llm.openai.OpenAI")
    def test_chat_connection_error(self, mock_openai):
        """Test handling of connection errors."""
        mock_client = MagicMock()
        mock_client.base_url = "http://localhost:11434/v1"
        # APIConnectionError needs message and request
        mock_client.chat.completions.create.side_effect = openai.APIConnectionError(message="Connection refused", request=MagicMock())
        mock_openai.return_value = mock_client
        
        llm = OllamaLLM(model="llama3")
        
        with pytest.raises(RuntimeError) as exc:
            llm.chat([{"role": "user", "content": "Hi"}])
        
        assert "Failed to connect to Ollama" in str(exc.value)
        assert "http://localhost:11434/v1" in str(exc.value)
