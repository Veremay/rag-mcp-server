from typing import Any, Dict, List, Optional

import openai

from src.libs.llm.base_llm import BaseLLM


class OpenAILLM(BaseLLM):
    """OpenAI-compatible LLM implementation."""

    def __init__(
        self, api_key: str, model: str, base_url: Optional[str] = None, **kwargs
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key.
            model: Model name (e.g., "gpt-4o").
            base_url: Optional base URL for compatible APIs.
            **kwargs: Additional arguments passed to openai.OpenAI.
        """
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, **kwargs)
        self.model = model

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Send chat request to OpenAI.

        Args:
            messages: List of message dictionaries.
            **kwargs: Additional arguments (temperature, etc.).

        Returns:
            Assistant response content.

        Raises:
            RuntimeError: If API call fails.
        """
        try:
            # Filter out None values from kwargs to allow defaults
            params = {k: v for k, v in kwargs.items() if v is not None}

            response = self.client.chat.completions.create(
                model=self.model, messages=messages, **params  # type: ignore
            )
            return response.choices[0].message.content or ""
        except openai.APIError as e:
            raise RuntimeError(f"OpenAI API Error: {e}") from e
