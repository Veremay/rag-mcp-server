from unittest.mock import MagicMock, patch

import pytest

from src.core.settings import RerankSettings, Settings
from src.libs.reranker.base_reranker import BaseReranker
from src.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from src.libs.reranker.llm_reranker import LLMReranker
from src.libs.reranker.reranker_factory import NoneReranker, RerankerFactory


@pytest.fixture
def mock_settings():
    settings = MagicMock(spec=Settings)
    settings.rerank = MagicMock(spec=RerankSettings)
    return settings


def test_none_reranker_behavior():
    """Test that NoneReranker returns candidates unchanged."""
    reranker = NoneReranker()
    candidates = ["doc1", "doc2", "doc3"]
    query = "test query"

    # Without top_k
    results = reranker.rerank(query, candidates)
    assert results == candidates

    # With top_k
    results_top_2 = reranker.rerank(query, candidates, top_k=2)
    assert results_top_2 == ["doc1", "doc2"]
    assert len(results_top_2) == 2


def test_factory_create_none_reranker(mock_settings):
    """Test factory creates NoneReranker when backend is 'none'."""
    mock_settings.rerank.backend = "none"
    reranker = RerankerFactory.create(mock_settings)
    assert isinstance(reranker, NoneReranker)
    assert isinstance(reranker, BaseReranker)


def test_factory_unsupported_backend(mock_settings):
    """Test factory raises ValueError for unknown backend."""
    mock_settings.rerank.backend = "unknown_bert"
    with pytest.raises(ValueError, match="Unsupported reranker backend"):
        RerankerFactory.create(mock_settings)


def test_factory_case_insensitive(mock_settings):
    """Test factory handles backend case insensitively."""
    mock_settings.rerank.backend = "NONE"
    reranker = RerankerFactory.create(mock_settings)
    assert isinstance(reranker, NoneReranker)


@patch("src.libs.reranker.reranker_factory.LLMFactory")
def test_factory_create_llm_reranker(mock_llm_factory, mock_settings):
    """Test factory creates LLMReranker when backend is 'llm'."""
    mock_settings.rerank.backend = "llm"
    # Mock LLM creation
    mock_llm = MagicMock()
    mock_llm_factory.create.return_value = mock_llm

    reranker = RerankerFactory.create(mock_settings)

    assert isinstance(reranker, LLMReranker)
    assert reranker.llm == mock_llm
    mock_llm_factory.create.assert_called_once_with(mock_settings)


@patch("src.libs.reranker.cross_encoder_reranker.CrossEncoderReranker._load_model")
def test_factory_create_cross_encoder_reranker(mock_load_model, mock_settings):
    """Test factory creates CrossEncoderReranker when backend is 'cross-encoder'."""
    mock_settings.rerank.backend = "cross-encoder"
    mock_settings.rerank.model = "test/model"

    reranker = RerankerFactory.create(mock_settings)

    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model_name == "test/model"


def test_none_reranker_boundaries():
    """Test NoneReranker with boundary conditions."""
    reranker = NoneReranker()
    query = "test"
    
    # 1. Empty candidates
    assert reranker.rerank(query, []) == []
    
    # 2. Candidates fewer than top_k
    candidates = ["a", "b"]
    assert len(reranker.rerank(query, candidates, top_k=10)) == 2
    
    # 3. Candidates equal to top_k
    assert len(reranker.rerank(query, candidates, top_k=2)) == 2
    
    # 4. Top_k=0 (should return empty list)
    assert reranker.rerank(query, candidates, top_k=0) == []
    
    # 5. Very large input (should work fast for NoneReranker)
    large_candidates = ["doc"] * 100
    assert len(reranker.rerank(query, large_candidates, top_k=50)) == 50
