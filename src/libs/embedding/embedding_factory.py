from typing import Dict, Type, Any
from src.core.settings import Settings
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.openai_embedding import OpenAIEmbedding
from src.libs.embedding.local_embedding import LocalEmbedding

class EmbeddingFactory:
    """
    Factory for creating Embedding instances based on configuration.
    """
    
    @staticmethod
    def create(settings: Settings) -> BaseEmbedding:
        """
        Create an Embedding instance based on the provided settings.

        Args:
            settings: The global application settings.

        Returns:
            An instance of BaseEmbedding.

        Raises:
            ValueError: If the provider is not supported or missing config.
        """
        provider = settings.embedding.provider.lower()
        
        if provider == "openai":
            if not settings.embedding.api_key:
                raise ValueError("OpenAI embedding provider requires api_key")
            
            kwargs = {}
            if settings.embedding.base_url:
                kwargs["base_url"] = settings.embedding.base_url
                
            return OpenAIEmbedding(
                api_key=settings.embedding.api_key,
                model=settings.embedding.model,
                **kwargs
            )
            
        elif provider == "local":
            return LocalEmbedding(
                model=settings.embedding.model
            )
            
        else:
            raise ValueError(f"Unknown embedding provider: '{provider}'")
