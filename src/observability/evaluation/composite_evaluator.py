from typing import Any, Dict, List, Optional

from src.libs.evaluator.base_evaluator import BaseEvaluator


class CompositeEvaluator(BaseEvaluator):
    """
    Evaluator that combines multiple evaluators.
    Executes all provided evaluators and merges their metrics.
    """

    def __init__(self, evaluators: List[BaseEvaluator]):
        """
        Initialize with a list of evaluators.

        Args:
            evaluators: List of BaseEvaluator instances to combine.
        """
        self.evaluators = evaluators

    def evaluate(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Run all evaluators and merge their results.

        Args:
            query: The search query.
            retrieved_ids: List of document IDs retrieved by the system.
            golden_ids: List of expected document IDs (ground truth).
            trace: Optional trace context.
            **kwargs: Additional arguments for specific evaluators.

        Returns:
            Combined dictionary of metrics from all evaluators.
        """
        all_metrics = {}
        for evaluator in self.evaluators:
            metrics = evaluator.evaluate(
                query, retrieved_ids, golden_ids, trace, **kwargs
            )
            all_metrics.update(metrics)
        return all_metrics
