from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseEvaluator(ABC):
    """Abstract base class for Evaluator implementations."""

    @abstractmethod
    def evaluate(
        self, 
        query: str, 
        retrieved_ids: List[str], 
        golden_ids: List[str],
        trace: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Evaluate retrieval results against golden standard.

        Args:
            query: The search query.
            retrieved_ids: List of document IDs retrieved by the system.
            golden_ids: List of expected document IDs (ground truth).
            trace: Optional trace context.

        Returns:
            Dictionary of metric names and their values (e.g., {"hit_rate": 1.0}).
        """
        pass
