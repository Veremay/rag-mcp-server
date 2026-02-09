from typing import List

from src.core.settings import Settings
from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator


class EvaluatorFactory:
    """Factory for creating Evaluator instances based on configuration."""

    @staticmethod
    def create(settings: Settings) -> BaseEvaluator:
        """
        Create an Evaluator instance.

        For now, this supports creating a 'custom' evaluator.
        If multiple backends are specified, it prioritizes 'custom' or the first supported one.

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

        # Future:
        # if "ragas" in backends: return RagasEvaluator(...)

        raise ValueError(f"No supported evaluator backend found in: {backends}")
