import pytest

from src.core.query_engine.hybrid_search import HybridSearchHit
from src.core.response.response_builder import ResponseBuilder
from src.libs.vector_store.base_vector_store import VectorRecord


@pytest.mark.unit
def test_response_builder_empty_hits_returns_friendly_message() -> None:
    out = ResponseBuilder().build([], query="q")
    assert out["content"][0]["type"] == "text"
    assert "未找到相关文档" in out["content"][0]["text"]
    assert out["structuredContent"]["citations"] == []


@pytest.mark.unit
def test_response_builder_builds_markdown_and_citations() -> None:
    record = VectorRecord(
        id="c1",
        embedding=[0.0],
        content="Hello world " * 20,
        metadata={"source_path": "/tmp/a.pdf", "section_path": "S1"},
    )
    hits = [
        HybridSearchHit(
            chunk_id="c1",
            score=0.9,
            record=record,
            dense_rank=1,
            sparse_rank=2,
        )
    ]
    out = ResponseBuilder().build(hits, query="hello")
    assert out["content"][0]["type"] == "text"
    text = out["content"][0]["text"]
    assert "[1]" in text
    citations = out["structuredContent"]["citations"]
    assert len(citations) == 1
    assert citations[0]["source"] == "/tmp/a.pdf"
    assert citations[0]["chunk_id"] == "c1"
