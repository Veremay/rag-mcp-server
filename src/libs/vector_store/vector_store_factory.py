from src.core.settings import Settings
from src.libs.vector_store.base_vector_store import BaseVectorStore

class VectorStoreFactory:
    """Factory for creating vector store instances based on configuration."""

    @staticmethod
    def create(settings: Settings) -> BaseVectorStore:
        """
        Create a vector store instance.

        Args:
            settings: Global settings object containing vector_store configuration.

        Returns:
            An instance of BaseVectorStore.

        Raises:
            ValueError: If the configured backend is not supported.
        """
        backend = settings.vector_store.backend.lower()

        if backend == "chroma":
            # TODO: Implement ChromaStore in B7.6
            raise NotImplementedError("ChromaStore implementation is pending (scheduled for B7.6)")
        
        # Extension point for other backends (e.g., qdrant, pinecone)
        
        raise ValueError(f"Unsupported vector store backend: {backend}")
