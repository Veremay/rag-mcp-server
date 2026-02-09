from typing import Any, Dict, List, Optional

import openai

from src.libs.llm.base_llm import BaseLLM


class AzureOpenAILLM(BaseLLM):
    """Azure OpenAI LLM implementation."""

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        model: str,
        api_version: str = "2023-05-15",
        **kwargs,
    ):
        """
        Initialize Azure OpenAI client.

        Args:
            api_key: Azure API key.
            azure_endpoint: Azure endpoint URL.
            model: Deployment name.
            api_version: API version (default: 2023-05-15).
            **kwargs: Additional arguments.
        """
        self.client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            **kwargs,
        )
        self.model = model

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Send chat request to Azure OpenAI.
        """
        try:
            params = {k: v for k, v in kwargs.items() if v is not None}

            response = self.client.chat.completions.create(
                model=self.model, messages=messages, **params  # type: ignore
            )
            return response.choices[0].message.content or ""
        except openai.APIError as e:
            raise RuntimeError(f"Azure OpenAI API Error: {e}") from e
