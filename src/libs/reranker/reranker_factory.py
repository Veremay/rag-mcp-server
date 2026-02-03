from typing import List, Any, Optional
from src.core.settings import Settings
from src.libs.reranker.base_reranker import BaseReranker

class NoneReranker(BaseReranker):
    """
    No-op reranker that returns candidates in their original order.
    Used as a fallback or when reranking is disabled.
    """
    def rerank(
        self, 
        query: str, 
        candidates: List[Any], 
        top_k: Optional[int] = None,
        trace: Optional[Any] = None
    ) -> List[Any]:
        """Return candidates as-is, optionally truncated to top_k."""
        if top_k is not None:
            return candidates[:top_k]
        return candidates

class RerankerFactory:
    """Factory for creating Reranker instances based on configuration."""

    @staticmethod
    def create(settings: Settings) -> BaseReranker:
        """
        Create a Reranker instance.

        Args:
            settings: Global settings object containing rerank configuration.

        Returns:
            An instance of BaseReranker.

        Raises:
            ValueError: If the configured backend is not supported.
        """
        backend = settings.rerank.backend.lower()

        if backend == "none":
            return NoneReranker()
        
        # Future implementations:
        # if backend == "cross-encoder": ...
        # if backend == "llm": ...
        
        raise ValueError(f"Unsupported reranker backend: {backend}")
