import pytest
from unittest.mock import MagicMock
from src.libs.reranker.llm_reranker import LLMReranker
from src.libs.llm.base_llm import BaseLLM

class Candidate:
    def __init__(self, text):
        self.text = text
    def __repr__(self):
        return f"Candidate({self.text})"

@pytest.fixture
def mock_llm():
    return MagicMock(spec=BaseLLM)

@pytest.fixture
def reranker(mock_llm, tmp_path):
    prompt_file = tmp_path / "rerank.txt"
    prompt_file.write_text("Query: {query}\nCandidates:\n{candidates}", encoding="utf-8")
    return LLMReranker(mock_llm, prompt_path=str(prompt_file))

def test_rerank_success(reranker, mock_llm):
    candidates = [Candidate("apple"), Candidate("banana"), Candidate("orange")]
    # LLM returns indices: 1 (banana), 2 (orange), 0 (apple)
    mock_llm.chat.return_value = "[1, 2, 0]"
    
    results = reranker.rerank("fruit", candidates)
    
    assert len(results) == 3
    assert results[0].text == "banana"
    assert results[1].text == "orange"
    assert results[2].text == "apple"
    
    # Verify LLM called with correct prompt
    args, _ = mock_llm.chat.call_args
    messages = args[0]
    content = messages[0]["content"]
    assert "Query: fruit" in content
    assert "[0] apple" in content
    assert "[1] banana" in content

def test_rerank_with_top_k(reranker, mock_llm):
    candidates = [Candidate("apple"), Candidate("banana"), Candidate("orange")]
    mock_llm.chat.return_value = "[1, 2, 0]"
    
    results = reranker.rerank("fruit", candidates, top_k=2)
    
    assert len(results) == 2
    assert results[0].text == "banana"
    assert results[1].text == "orange"

def test_rerank_fallback_on_llm_error(reranker, mock_llm):
    candidates = [Candidate("apple"), Candidate("banana")]
    mock_llm.chat.side_effect = Exception("API Error")
    
    results = reranker.rerank("fruit", candidates)
    
    # Should return original order
    assert len(results) == 2
    assert results[0].text == "apple"
    assert results[1].text == "banana"

def test_rerank_fallback_on_parse_error(reranker, mock_llm):
    candidates = [Candidate("apple"), Candidate("banana")]
    mock_llm.chat.return_value = "I cannot do that."
    
    results = reranker.rerank("fruit", candidates)
    
    # Should return original order
    assert len(results) == 2
    assert results[0].text == "apple"
    assert results[1].text == "banana"

def test_rerank_partial_indices(reranker, mock_llm):
    # LLM returns only one index
    candidates = [Candidate("apple"), Candidate("banana"), Candidate("orange")]
    mock_llm.chat.return_value = "[2]"
    
    results = reranker.rerank("fruit", candidates)
    
    # [2] first, then others (0, 1) appended
    assert len(results) == 3
    assert results[0].text == "orange"
    # Order of others depends on implementation logic. 
    # Current logic: for i in range(len(candidates)): if i not in seen_indices: append
    # So it should be 0, 1
    assert results[1].text == "apple"
    assert results[2].text == "banana"

def test_json_parsing_resilience(reranker, mock_llm):
    candidates = [Candidate("a"), Candidate("b")]
    # Markdown code block
    mock_llm.chat.return_value = "Here is the result:\n```json\n[1, 0]\n```"
    
    results = reranker.rerank("q", candidates)
    
    assert results[0].text == "b"
    assert results[1].text == "a"
