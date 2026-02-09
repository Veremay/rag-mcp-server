import sys
from unittest.mock import MagicMock, patch

import pytest

from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker


class TestCrossEncoderReranker:

    @pytest.fixture
    def mock_scorer(self):
        scorer = MagicMock()
        return scorer

    def test_initialization(self, mock_scorer):
        reranker = CrossEncoderReranker(model_name="test-model", scorer=mock_scorer)
        assert reranker.scorer == mock_scorer
        assert reranker.model_name == "test-model"

    def test_rerank_sorting(self, mock_scorer):
        """Test that candidates are correctly sorted by score."""
        reranker = CrossEncoderReranker(model_name="test-model", scorer=mock_scorer)
        candidates = ["doc1", "doc2", "doc3"]
        query = "test query"

        # Mock scores: doc2 (0.9) > doc1 (0.5) > doc3 (0.1)
        # pairs order in code: [[query, doc1], [query, doc2], [query, doc3]]
        mock_scorer.predict.return_value = [0.5, 0.9, 0.1]

        results = reranker.rerank(query, candidates)

        # Verify predict called with correct pairs
        expected_pairs = [[query, "doc1"], [query, "doc2"], [query, "doc3"]]
        mock_scorer.predict.assert_called_once_with(expected_pairs)

        # Verify order
        assert results == ["doc2", "doc1", "doc3"]

    def test_rerank_top_k(self, mock_scorer):
        """Test top_k truncation."""
        reranker = CrossEncoderReranker(model_name="test-model", scorer=mock_scorer)
        candidates = ["doc1", "doc2", "doc3"]
        mock_scorer.predict.return_value = [0.5, 0.9, 0.1]

        results = reranker.rerank("q", candidates, top_k=2)
        assert len(results) == 2
        assert results == ["doc2", "doc1"]

    def test_rerank_complex_objects(self, mock_scorer):
        """Test reranking with object candidates (checking text extraction)."""
        reranker = CrossEncoderReranker(model_name="test-model", scorer=mock_scorer)

        class Doc:
            def __init__(self, text):
                self.text = text

        c1 = Doc("text1")
        c2 = {"text": "text2"}
        c3 = "text3"
        candidates = [c1, c2, c3]

        mock_scorer.predict.return_value = [0.1, 0.2, 0.3]  # c3 > c2 > c1

        results = reranker.rerank("q", candidates)

        # Verify text extraction
        expected_pairs = [["q", "text1"], ["q", "text2"], ["q", "text3"]]
        mock_scorer.predict.assert_called_with(expected_pairs)

        assert results == [c3, c2, c1]

    def test_rerank_failure_propagation(self, mock_scorer):
        """Test that exceptions are propagated (for fallback signal)."""
        reranker = CrossEncoderReranker(model_name="test-model", scorer=mock_scorer)
        mock_scorer.predict.side_effect = Exception("Model error")

        with pytest.raises(Exception, match="Model error"):
            reranker.rerank("q", ["doc1"])

    def test_empty_candidates(self, mock_scorer):
        reranker = CrossEncoderReranker(model_name="test-model", scorer=mock_scorer)
        assert reranker.rerank("q", []) == []

    def test_initialization_load_model(self):
        """Test initialization attempts to import sentence_transformers."""
        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
            # Mock the class inside the module
            mock_module = sys.modules["sentence_transformers"]
            mock_cls = MagicMock()
            mock_module.CrossEncoder = mock_cls

            reranker = CrossEncoderReranker(model_name="test-model")
            mock_cls.assert_called_once_with("test-model")
            assert reranker.scorer is not None

    def test_missing_dependency(self):
        """Test graceful degradation when dependency is missing."""
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            # Simulate ImportError
            with patch("builtins.__import__", side_effect=ImportError):
                # Note: Testing import errors is tricky with patch,
                # but logic is: init shouldn't crash, but rerank should raise RuntimeError
                reranker = CrossEncoderReranker(model_name="test-model")
                assert reranker.scorer is None

                with pytest.raises(RuntimeError, match="not initialized"):
                    reranker.rerank("q", ["doc"])
