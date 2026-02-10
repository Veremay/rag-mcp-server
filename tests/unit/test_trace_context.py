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
    trace = TraceContext(trace_id="t2")
    trace.record_stage("s1", data={"path": object()})

    payload = trace.finish()
    assert payload["trace_id"] == "t2"
    assert isinstance(payload["duration_ms"], float)
    assert isinstance(payload["stages"], list)

    s = json.dumps(payload, ensure_ascii=False)
    assert isinstance(s, str) and s


@pytest.mark.unit
def test_record_stage_requires_non_empty_name() -> None:
    trace = TraceContext(trace_id="t3")
    with pytest.raises(ValueError):
        trace.record_stage("  ")
