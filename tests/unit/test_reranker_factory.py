import pytest
from unittest.mock import MagicMock, patch
from src.core.settings import Settings, RerankSettings
from src.libs.reranker.base_reranker import BaseReranker
from src.libs.reranker.reranker_factory import RerankerFactory, NoneReranker
from src.libs.reranker.llm_reranker import LLMReranker

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
