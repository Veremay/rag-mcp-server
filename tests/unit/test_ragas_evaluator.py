import pytest
from unittest.mock import MagicMock, patch
from src.libs.evaluator.ragas_evaluator import RagasEvaluator

# Create mock metric objects with a 'name' attribute
def create_mock_metric(name):
    m = MagicMock()
    m.name = name
    return m

class TestRagasEvaluator:
    
    @patch("src.libs.evaluator.ragas_evaluator.RAGAS_AVAILABLE", True)
    @patch("src.libs.evaluator.ragas_evaluator.context_recall", create_mock_metric("context_recall"))
    @patch("src.libs.evaluator.ragas_evaluator.context_precision", create_mock_metric("context_precision"))
    @patch("src.libs.evaluator.ragas_evaluator.faithfulness", create_mock_metric("faithfulness"))
    @patch("src.libs.evaluator.ragas_evaluator.answer_relevancy", create_mock_metric("answer_relevancy"))
    def test_initialization_defaults(self):
        evaluator = RagasEvaluator()
        # Should have all 4 supported metrics
        assert len(evaluator.selected_metrics) == 4

    @patch("src.libs.evaluator.ragas_evaluator.RAGAS_AVAILABLE", True)
    @patch("src.libs.evaluator.ragas_evaluator.context_recall", create_mock_metric("context_recall"))
    @patch("src.libs.evaluator.ragas_evaluator.context_precision", create_mock_metric("context_precision"))
    @patch("src.libs.evaluator.ragas_evaluator.faithfulness", create_mock_metric("faithfulness"))
    @patch("src.libs.evaluator.ragas_evaluator.answer_relevancy", create_mock_metric("answer_relevancy"))
    def test_initialization_filtering(self):
        evaluator = RagasEvaluator(metrics=["context_recall", "invalid_metric"])
        assert len(evaluator.selected_metrics) == 1
        # The metric object name might be 'context_recall'
        assert evaluator.selected_metrics[0].name == "context_recall"

    @patch("src.libs.evaluator.ragas_evaluator.RAGAS_AVAILABLE", False)
    def test_initialization_missing_dependency(self):
        with pytest.raises(ImportError):
            RagasEvaluator()

    @patch("src.libs.evaluator.ragas_evaluator.RAGAS_AVAILABLE", True)
    @patch("src.libs.evaluator.ragas_evaluator.context_recall", create_mock_metric("context_recall"))
    @patch("src.libs.evaluator.ragas_evaluator.context_precision", create_mock_metric("context_precision"))
    @patch("src.libs.evaluator.ragas_evaluator.faithfulness", create_mock_metric("faithfulness"))
    @patch("src.libs.evaluator.ragas_evaluator.answer_relevancy", create_mock_metric("answer_relevancy"))
    def test_evaluate_missing_kwargs(self, caplog):
        evaluator = RagasEvaluator()
        result = evaluator.evaluate(
            query="test query", 
            retrieved_ids=["doc1"], 
            golden_ids=["doc1"]
        )
        assert result == {}
        # Expect generic warning when no metrics can be run due to missing data
        assert "No applicable metrics for the provided data" in caplog.text

    @patch("src.libs.evaluator.ragas_evaluator.RAGAS_AVAILABLE", True)
    @patch("src.libs.evaluator.ragas_evaluator.ragas_evaluate")
    @patch("src.libs.evaluator.ragas_evaluator.Dataset")
    @patch("src.libs.evaluator.ragas_evaluator.context_recall", create_mock_metric("context_recall"))
    @patch("src.libs.evaluator.ragas_evaluator.context_precision", create_mock_metric("context_precision"))
    @patch("src.libs.evaluator.ragas_evaluator.faithfulness", create_mock_metric("faithfulness"))
    @patch("src.libs.evaluator.ragas_evaluator.answer_relevancy", create_mock_metric("answer_relevancy"))
    def test_evaluate_success(self, mock_dataset, mock_evaluate):
        # Setup mock return
        mock_evaluate.return_value = {"context_recall": 0.9}
        
        evaluator = RagasEvaluator(metrics=["context_recall"])
        
        # Execute
        result = evaluator.evaluate(
            query="test query",
            retrieved_ids=["doc1"],
            golden_ids=["doc1"],
            retrieved_texts=["content of doc1"],
            golden_answer="expected answer"
        )
        
        # Verify
        assert result == {"context_recall": 0.9}
        mock_dataset.from_dict.assert_called_once()
        
        # Check data passed to Dataset
        call_args = mock_dataset.from_dict.call_args
        data = call_args[0][0]
        assert data["question"] == ["test query"]
        assert data["contexts"] == [["content of doc1"]]
        assert data["ground_truth"] == ["expected answer"]
        assert "answer" not in data # generated_answer not provided

    @patch("src.libs.evaluator.ragas_evaluator.RAGAS_AVAILABLE", True)
    @patch("src.libs.evaluator.ragas_evaluator.ragas_evaluate")
    @patch("src.libs.evaluator.ragas_evaluator.Dataset")
    @patch("src.libs.evaluator.ragas_evaluator.context_recall", create_mock_metric("context_recall"))
    @patch("src.libs.evaluator.ragas_evaluator.context_precision", create_mock_metric("context_precision"))
    @patch("src.libs.evaluator.ragas_evaluator.faithfulness", create_mock_metric("faithfulness"))
    @patch("src.libs.evaluator.ragas_evaluator.answer_relevancy", create_mock_metric("answer_relevancy"))
    def test_evaluate_with_generated_answer(self, mock_dataset, mock_evaluate):
        mock_evaluate.return_value = {"faithfulness": 0.8}
        
        evaluator = RagasEvaluator(metrics=["faithfulness"])
        
        result = evaluator.evaluate(
            query="test query",
            retrieved_ids=["doc1"],
            golden_ids=["doc1"],
            retrieved_texts=["content"],
            generated_answer="generated answer"
        )
        
        assert result == {"faithfulness": 0.8}
        
        data = mock_dataset.from_dict.call_args[0][0]
        assert data["answer"] == ["generated answer"]
