from unittest.mock import MagicMock

import pytest

from src.core.settings import EvaluationSettings, Settings
from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.libs.evaluator.evaluator_factory import EvaluatorFactory


def test_custom_evaluator_metrics():
    """Test Hit Rate and MRR calculation logic."""
    evaluator = CustomEvaluator()
    query = "test"

    # Case 1: Perfect match at rank 1
    # golden: A, retrieved: A, B, C
    metrics = evaluator.evaluate(query, ["A", "B", "C"], ["A"])
    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 1.0

    # Case 2: Match at rank 2
    # golden: A, retrieved: B, A, C
    metrics = evaluator.evaluate(query, ["B", "A", "C"], ["A"])
    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 0.5  # 1/2

    # Case 3: No match
    # golden: A, retrieved: B, C, D
    metrics = evaluator.evaluate(query, ["B", "C", "D"], ["A"])
    assert metrics["hit_rate"] == 0.0
    assert metrics["mrr"] == 0.0

    # Case 4: Multiple goldens
    # golden: A, Z. retrieved: B, A, C
    # First match is A at rank 2
    metrics = evaluator.evaluate(query, ["B", "A", "C"], ["A", "Z"])
    assert metrics["hit_rate"] == 1.0
    assert metrics["mrr"] == 0.5

    # Case 5: Empty golden (should be 0)
    metrics = evaluator.evaluate(query, ["A"], [])
    assert metrics["hit_rate"] == 0.0
    assert metrics["mrr"] == 0.0


def test_factory_creates_custom_evaluator():
    """Test factory creates CustomEvaluator when configured."""
    settings = MagicMock(spec=Settings)
    settings.evaluation = MagicMock(spec=EvaluationSettings)
    settings.evaluation.backends = ["custom"]

    evaluator = EvaluatorFactory.create(settings)
    assert isinstance(evaluator, CustomEvaluator)
    assert isinstance(evaluator, BaseEvaluator)


def test_factory_creates_custom_evaluator_mixed():
    """Test factory prioritizes custom if present in list."""
    settings = MagicMock(spec=Settings)
    settings.evaluation = MagicMock(spec=EvaluationSettings)
    settings.evaluation.backends = ["ragas", "custom"]

    evaluator = EvaluatorFactory.create(settings)
    assert isinstance(evaluator, CustomEvaluator)


def test_factory_raises_error_on_unsupported():
    """Test factory raises error if no supported backend found."""
    settings = MagicMock(spec=Settings)
    settings.evaluation = MagicMock(spec=EvaluationSettings)
    settings.evaluation.backends = ["unknown_backend"]

    with pytest.raises(ValueError, match="No supported evaluator backend"):
        EvaluatorFactory.create(settings)
