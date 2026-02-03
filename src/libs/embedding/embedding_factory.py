from typing import Dict, Type, Any
from src.core.settings import Settings
from src.libs.embedding.base_embedding import BaseEmbedding

class EmbeddingFactory:
    """
    Factory for creating Embedding instances based on configuration.
    Supports dynamic registration of providers.
    """
    _registry: Dict[str, Type[BaseEmbedding]] = {}

    @classmethod
    def register(cls, provider: str, embedding_cls: Type[BaseEmbedding]) -> None:
        """
        Register a new embedding provider class.

        Args:
            provider: The provider name (e.g., "openai", "azure").
            embedding_cls: The class implementing BaseEmbedding.
        """
        cls._registry[provider.lower()] = embedding_cls

    @classmethod
    def create(cls, settings: Settings) -> BaseEmbedding:
        """
        Create an Embedding instance based on the provided settings.

        Args:
            settings: The global application settings.

        Returns:
            An instance of BaseEmbedding.

        Raises:
            ValueError: If the provider is not registered.
        """
        provider = settings.embedding.provider.lower()
        
        if provider not in cls._registry:
            raise ValueError(
                f"Unknown embedding provider: '{provider}'. "
                f"Available providers: {list(cls._registry.keys())}"
            )
        
        embedding_cls = cls._registry[provider]
        
        # Here we would typically pass relevant config to the constructor
        # For now, we pass the whole settings object or specific config
        # Assuming constructors take settings or **kwargs
        return embedding_cls(settings)
