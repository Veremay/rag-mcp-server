import pytest
from unittest.mock import MagicMock, patch, Mock
from typing import Dict, Any, List, Optional

from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.observability.evaluation.composite_evaluator import CompositeEvaluator
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.libs.evaluator.custom_evaluator import CustomEvaluator
# We don't import RagasEvaluator directly to avoid hard dependency in test file if we patch it
# from src.observability.evaluation.ragas_evaluator import RagasEvaluator

class MockEvaluator(BaseEvaluator):
    def __init__(self, metrics: Dict[str, float]):
        self.metrics = metrics
        self.call_args = []

    def evaluate(
        self,
        query: str,
        retrieved_ids: List[str],
        golden_ids: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        self.call_args.append((query, retrieved_ids, golden_ids, trace, kwargs))
        return self.metrics

class TestCompositeEvaluator:
    def test_evaluate_merges_metrics(self):
        eval1 = MockEvaluator({"m1": 0.1, "m2": 0.2})
        eval2 = MockEvaluator({"m3": 0.3})
        composite = CompositeEvaluator([eval1, eval2])
        
        results = composite.evaluate("q", ["1"], ["1"])
        
        assert results == {"m1": 0.1, "m2": 0.2, "m3": 0.3}
        assert len(eval1.call_args) == 1
        assert len(eval2.call_args) == 1

    def test_evaluate_overwrites_metrics(self):
        # Last one wins
        eval1 = MockEvaluator({"score": 0.5})
        eval2 = MockEvaluator({"score": 0.9})
        composite = CompositeEvaluator([eval1, eval2])
        
        results = composite.evaluate("q", ["1"], ["1"])
        assert results == {"score": 0.9}

class TestEvaluatorFactoryComposite:
    def test_create_single_custom(self):
        settings = Mock()
        settings.evaluation.backends = ["custom"]
        
        evaluator = EvaluatorFactory.create(settings)
        assert isinstance(evaluator, CustomEvaluator)

    def test_create_single_ragas(self):
        settings = Mock()
        settings.evaluation.backends = ["ragas"]
        settings.evaluation.metrics = ["faithfulness"]
        
        with patch("src.libs.evaluator.evaluator_factory.RagasEvaluator") as mock_ragas_cls:
            mock_instance = Mock()
            mock_ragas_cls.return_value = mock_instance
            
            evaluator = EvaluatorFactory.create(settings)
            
            assert evaluator == mock_instance
            mock_ragas_cls.assert_called_once_with(metrics=["faithfulness"])

    def test_create_composite(self):
        settings = Mock()
        settings.evaluation.backends = ["custom", "ragas"]
        settings.evaluation.metrics = ["faithfulness"]
        
        with patch("src.libs.evaluator.evaluator_factory.RagasEvaluator") as mock_ragas_cls:
            mock_ragas_instance = Mock()
            mock_ragas_cls.return_value = mock_ragas_instance
            
            evaluator = EvaluatorFactory.create(settings)
            
            assert isinstance(evaluator, CompositeEvaluator)
            assert len(evaluator.evaluators) == 2
            assert isinstance(evaluator.evaluators[0], CustomEvaluator)
            assert evaluator.evaluators[1] == mock_ragas_instance
