import json

import pytest

from src.core.trace.trace_context import TraceContext


@pytest.mark.unit
def test_record_stage_appends_stage() -> None:
    trace = TraceContext(trace_id="t1")
    trace.record_stage("dense", start_ms=10.0, end_ms=25.0, data={"k": "v"})
    trace.record_stage("fusion", duration_ms=3.5, metrics={"top_k": 10.0})

    assert trace.trace_id == "t1"
    assert len(trace.stages) == 2
    assert trace.stages[0].name == "dense"
    assert trace.stages[1].name == "fusion"


@pytest.mark.unit
def test_finish_returns_json_serializable_payload() -> None:
    trace = TraceContext(trace_id="t2", trace_type="test")
    # Simulate a non-serializable object in data to test fallback
    trace.record_stage("s1", data={"path": object()})

    trace.finish()
    assert trace.finished_ms is not None

    payload = trace.to_dict()
    assert payload["trace_id"] == "t2"
    assert payload["trace_type"] == "test"
    assert isinstance(payload["duration_ms"], float)
    assert isinstance(payload["total_elapsed_ms"], float)
    assert payload["duration_ms"] == payload["total_elapsed_ms"]
    assert isinstance(payload["stages"], list)

    s = json.dumps(payload, ensure_ascii=False)
    assert isinstance(s, str) and s


@pytest.mark.unit
def test_elapsed_ms_method() -> None:
    trace = TraceContext()
    trace.record_stage("s1", start_ms=100.0, end_ms=200.0)
    trace.record_stage("s2", duration_ms=50.0)
    
    # Test stage duration
    assert trace.elapsed_ms("s1") == 100.0
    assert trace.elapsed_ms("s2") == 50.0
    assert trace.elapsed_ms("non_existent") == 0.0
    
    # Test total duration
    total = trace.elapsed_ms()
    assert isinstance(total, float)
    assert total >= 0.0


@pytest.mark.unit
def test_trace_context_enhanced_fields() -> None:
    # 验证 trace_type 默认值
    trace = TraceContext()
    assert trace.trace_type == "query"

    # 验证 trace_type 自定义
    trace_ingest = TraceContext(trace_type="ingestion")
    assert trace_ingest.trace_type == "ingestion"

    # 验证 elapsed_ms
    assert trace.elapsed_ms() >= 0.0


@pytest.mark.unit
def test_record_stage_requires_non_empty_name() -> None:
    trace = TraceContext(trace_id="t3")
    with pytest.raises(ValueError):
        trace.record_stage("  ")
