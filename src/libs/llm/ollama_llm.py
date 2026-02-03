from typing import List, Dict, Any, Optional
import openai
from src.libs.llm.base_llm import BaseLLM

class OllamaLLM(BaseLLM):
    """
    Ollama LLM implementation using OpenAI-compatible API.
    
    This class connects to a local or remote Ollama instance using the OpenAI SDK,
    leveraging Ollama's OpenAI-compatible endpoint support.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",  # Ollama usually doesn't require an API key, but client needs one
        **kwargs
    ):
        """
        Initialize Ollama client.

        Args:
            model: Model name (e.g., "llama3").
            base_url: Base URL for Ollama API. Defaults to http://localhost:11434/v1.
            api_key: API key. Defaults to "ollama" as it's often ignored by Ollama but required by SDK.
            **kwargs: Additional arguments passed to openai.OpenAI.
        """
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )
        self.model = model

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Send chat request to Ollama.

        Args:
            messages: List of message dictionaries.
            **kwargs: Additional arguments (temperature, etc.).

        Returns:
            Assistant response content.

        Raises:
            RuntimeError: If API call fails or connection refused.
        """
        try:
            # Filter out None values from kwargs
            params = {k: v for k, v in kwargs.items() if v is not None}
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages, # type: ignore
                **params
            )
            return response.choices[0].message.content or ""
            
        except openai.APIConnectionError as e:
            # Provide a more helpful error message for connection issues
            # We explicitly mention the URL to help debugging
            raise RuntimeError(
                f"Failed to connect to Ollama at {self.client.base_url}. "
                "Is Ollama running? (Try 'ollama serve' or check your base_url)"
            ) from e
            
        except openai.APIError as e:
            raise RuntimeError(f"Ollama API Error: {e}") from e
