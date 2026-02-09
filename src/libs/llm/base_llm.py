from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.

    This class defines the interface that all LLM implementations must follow,
    ensuring pluggability across different providers (Azure, OpenAI, Ollama, etc.).
    """

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Send a chat completion request to the LLM.

        Args:
            messages: A list of message dictionaries, e.g.,
                     [{"role": "user", "content": "Hello"}]
            **kwargs: Additional provider-specific arguments (temperature, max_tokens, etc.)

        Returns:
            The content of the assistant's response as a string.
        """
        pass

    async def achat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Async version of chat completion.

        Args:
            messages: A list of message dictionaries
            **kwargs: Additional provider-specific arguments

        Returns:
            The content of the assistant's response as a string.
        """
        # Default implementation falls back to sync call if not overridden
        # In a real async implementation, this should be non-blocking
        return self.chat(messages, **kwargs)
