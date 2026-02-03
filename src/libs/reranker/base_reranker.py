from abc import ABC, abstractmethod
from typing import List, Any, Optional

class BaseReranker(ABC):
    """Abstract base class for Reranker implementations."""

    @abstractmethod
    def rerank(
        self, 
        query: str, 
        candidates: List[Any], 
        top_k: Optional[int] = None,
        trace: Optional[Any] = None
    ) -> List[Any]:
        """
        Rerank a list of candidates based on their relevance to the query.

        Args:
            query: The search query string.
            candidates: List of candidate objects (e.g., VectorRecord or chunks) to rerank.
            top_k: Optional number of top results to return. If None, return all.
            trace: Optional trace context.

        Returns:
            List of candidates sorted by relevance (most relevant first).
        """
        pass
