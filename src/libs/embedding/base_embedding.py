from abc import ABC, abstractmethod
from typing import List, Optional, Any

class BaseEmbedding(ABC):
    """
    Abstract base class for Embedding providers.
    All embedding implementations must inherit from this class.
    """

    @abstractmethod
    def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.
            **kwargs: Additional provider-specific arguments.

        Returns:
            List of embedding vectors (list of floats).
        """
        pass

    @abstractmethod
    async def aembed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """
        Asynchronously generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.
            **kwargs: Additional provider-specific arguments.

        Returns:
            List of embedding vectors (list of floats).
        """
        pass
