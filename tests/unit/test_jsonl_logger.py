import json
import logging
from pathlib import Path

import pytest

from src.core.settings import ObservabilitySettings, Settings
from src.core.trace.trace_context import TraceContext
from src.observability.logger import write_trace, get_trace_logger


def _settings(*, enabled: bool, log_file: str) -> Settings:
    from src.core.settings import (
        EmbeddingSettings,
        EvaluationSettings,
        IngestionSettings,
        LLMSettings,
        RerankSettings,
        RetrievalSettings,
        SplitterSettings,
        TransformSettings,
        VectorStoreSettings,
        VisionLLMSettings,
    )

    return Settings(
        llm=LLMSettings(provider="ollama", model="x", api_key=None, base_url=None),
        embedding=EmbeddingSettings(provider="local", model="fake"),
        vision_llm=VisionLLMSettings(provider="ollama", model="x"),
        vector_store=VectorStoreSettings(
            backend="jsonl", persist_path="data/db/vector"
        ),
        ingestion=IngestionSettings(
            splitter=SplitterSettings(provider="recursive"),
            transform=TransformSettings(),
        ),
        retrieval=RetrievalSettings(
            sparse_backend="bm25",
            fusion_algorithm="rrf",
            top_k_dense=2,
            top_k_sparse=2,
            top_k_final=3,
        ),
        rerank=RerankSettings(backend="none", model="x", top_m=5),
        evaluation=EvaluationSettings(backends=["custom"], golden_test_set=""),
        observability=ObservabilitySettings(
            enabled=enabled, log_file=log_file, dashboard_port=0
        ),
    )


@pytest.fixture(autouse=True)
def reset_trace_logger():
    """Reset trace logger handlers before and after each test."""
    logger = get_trace_logger()
    logger.handlers = []
    yield
    logger.handlers = []


@pytest.mark.unit
def test_jsonl_logger_writes_one_valid_json_line(tmp_path: Path) -> None:
    log_path = tmp_path / "traces.jsonl"
    settings = _settings(enabled=True, log_file=str(log_path))

    trace = TraceContext(trace_id="t1")
    trace.record_stage("s1", duration_ms=1.0, data={"k": "v"})
    trace.finish()

    write_trace(trace.to_dict(), settings=settings)

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["trace_id"] == "t1"
    assert isinstance(payload["stages"], list)


@pytest.mark.unit
def test_jsonl_logger_disabled_returns_none(tmp_path: Path) -> None:
    log_path = tmp_path / "traces.jsonl"
    settings = _settings(enabled=False, log_file=str(log_path))
    
    trace = TraceContext(trace_id="t1")
    trace.finish()
    
    write_trace(trace.to_dict(), settings=settings)
    
    assert not log_path.exists()
