from typing import List

from src.core.settings import Settings
from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.libs.evaluator.ragas_evaluator import RagasEvaluator


class EvaluatorFactory:
    """Factory for creating Evaluator instances based on configuration."""

    @staticmethod
    def create(settings: Settings) -> BaseEvaluator:
        """
        Create an Evaluator instance.

        For now, this supports creating a 'custom' evaluator or 'ragas' evaluator.
        If multiple backends are specified, it prioritizes 'custom', then 'ragas'.

        Args:
            settings: Global settings object.

        Returns:
            An instance of BaseEvaluator.

        Raises:
            ValueError: If no supported backend is configured.
        """
        backends = [b.lower() for b in settings.evaluation.backends]

        if "custom" in backends:
            return CustomEvaluator()

        if "ragas" in backends:
            return RagasEvaluator(metrics=settings.evaluation.metrics)

        raise ValueError(f"No supported evaluator backend found in: {backends}")
