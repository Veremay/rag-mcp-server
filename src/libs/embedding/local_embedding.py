import random
from typing import List, Any
from src.libs.embedding.base_embedding import BaseEmbedding

class LocalEmbedding(BaseEmbedding):
    """
    Local embedding implementation.
    
    Currently serves as a placeholder/adapter for B7.4 task.
    Generates fake deterministic vectors for testing purposes to ensure pipeline connectivity.
    Future versions can integrate BGE/Ollama/SentenceTransformers.
    """
    
    def __init__(self, model: str = "text-embedding-3-small", dimension: int = 1536):
        """
        Initialize LocalEmbedding.
        
        Args:
            model: Model name (unused in fake mode, but kept for interface compatibility).
            dimension: Dimension of the fake embedding vectors. Defaults to 1536 (OpenAI small).
        """
        self.model = model
        self.dimension = dimension

    def _generate_fake_vector(self, text: str) -> List[float]:
        """Generate a deterministic fake vector based on text content."""
        # Use a deterministic seed based on text hash
        seed = hash(text)
        rng = random.Random(seed)
        return [rng.random() for _ in range(self.dimension)]

    def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """
        Generate fake embeddings for a list of texts.
        
        Args:
            texts: List of strings to embed.
            **kwargs: Ignored in fake mode.
            
        Returns:
            List of embedding vectors.
        """
        return [self._generate_fake_vector(text) for text in texts]

    async def aembed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """
        Asynchronously generate fake embeddings for a list of texts.
        
        Args:
            texts: List of strings to embed.
            **kwargs: Ignored in fake mode.
            
        Returns:
            List of embedding vectors.
        """
        # CPU-bound fake generation doesn't really benefit from async, but satisfies interface
        return self.embed(texts, **kwargs)
