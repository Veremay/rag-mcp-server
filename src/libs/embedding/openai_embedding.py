from typing import Any, List, Optional

import openai

from src.libs.embedding.base_embedding import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI Embedding implementation using the official OpenAI Python SDK.
    Supports standard OpenAI API and compatible endpoints.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize the OpenAI Embedding provider.

        Args:
            api_key: OpenAI API key.
            model: Model name to use for embeddings.
            base_url: Optional custom base URL (for compatible APIs).
            **kwargs: Additional arguments passed to the OpenAI client.
        """
        self.model = model
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, **kwargs)
        self.aclient = openai.AsyncOpenAI(api_key=api_key, base_url=base_url, **kwargs)

    def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI API.

        Args:
            texts: List of strings to embed.
            **kwargs: Additional arguments passed to client.embeddings.create.

        Returns:
            List of embedding vectors.

        Raises:
            RuntimeError: If the API call fails.
        """
        if not texts:
            return []

        # Remove empty strings to avoid API errors if necessary,
        # but spec says "Empty input... have clear behavior".
        # OpenAI handles empty strings by returning error or embedding depending on version.
        # We will let the API decide or handle specific cases if we see failures.
        # Actually, for robustness, we should probably allow empty strings if the user sends them,
        # but usually embedding an empty string is useless.
        # Let's just pass through for now and handle errors.

        try:
            # Merge kwargs with defaults if needed
            response = self.client.embeddings.create(
                input=texts, model=self.model, **kwargs
            )
            # OpenAI response.data is a list of objects sorted by index
            # Ensure we return in the same order
            return [data.embedding for data in response.data]
        except openai.APIConnectionError as e:
            raise RuntimeError(f"Failed to connect to OpenAI Embedding API: {e}") from e
        except openai.APIStatusError as e:
            raise RuntimeError(f"OpenAI Embedding API returned error: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error during embedding generation: {e}"
            ) from e

    async def aembed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """
        Asynchronously generate embeddings for a list of texts.
        """
        if not texts:
            return []

        try:
            response = await self.aclient.embeddings.create(
                input=texts, model=self.model, **kwargs
            )
            return [data.embedding for data in response.data]
        except openai.APIConnectionError as e:
            raise RuntimeError(f"Failed to connect to OpenAI Embedding API: {e}") from e
        except openai.APIStatusError as e:
            raise RuntimeError(f"OpenAI Embedding API returned error: {e}") from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error during embedding generation: {e}"
            ) from e
