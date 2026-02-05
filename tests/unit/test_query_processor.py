from src.core.query_engine.query_processor import QueryProcessor


def test_query_processor_extracts_keywords():
    qp = QueryProcessor()
    result = qp.process("Explain Hybrid Search and RRF Fusion in RAG")
    assert isinstance(result.keywords, list)
    assert len(result.keywords) > 0
    assert "hybrid" in result.keywords
    assert "rrf" in result.keywords
    assert isinstance(result.filters, dict)


def test_query_processor_parses_filters_and_keeps_keywords():
    qp = QueryProcessor()
    result = qp.process("collection:demo doc_type=pdf 什么是 BM25 检索")
    assert result.filters.get("collection") == "demo"
    assert result.filters.get("doc_type") == "pdf"
    assert len(result.keywords) > 0
    assert "bm25" in result.keywords


def test_query_processor_empty_query():
    qp = QueryProcessor()
    result = qp.process("")
    assert result.keywords == []
    assert result.filters == {}
