from typing import List

from src.core.settings import Settings
from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.observability.evaluation.ragas_evaluator import RagasEvaluator
from src.observability.evaluation.composite_evaluator import CompositeEvaluator


class EvaluatorFactory:
    """Factory for creating Evaluator instances based on configuration."""

    @staticmethod
    def create(settings: Settings) -> BaseEvaluator:
        """
        Create an Evaluator instance.

        Supports 'custom', 'ragas', or a combination of both.
        If multiple backends are specified, returns a CompositeEvaluator.

        Args:
            settings: Global settings object.

        Returns:
            An instance of BaseEvaluator (could be CompositeEvaluator).

        Raises:
            ValueError: If no supported backend is configured.
        """
        backends = [b.lower() for b in settings.evaluation.backends]
        evaluators: List[BaseEvaluator] = []

        if "custom" in backends:
            evaluators.append(CustomEvaluator())

        if "ragas" in backends:
            evaluators.append(RagasEvaluator(metrics=settings.evaluation.metrics))

        if not evaluators:
            raise ValueError(f"No supported evaluator backend found in: {backends}")

        if len(evaluators) == 1:
            return evaluators[0]

        return CompositeEvaluator(evaluators)
