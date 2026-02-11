from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

JsonValue = Any
JsonDict = Dict[str, JsonValue]


def new_trace_id() -> str:
    return uuid.uuid4().hex


def _now_ms() -> float:
    return time.time() * 1000.0


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


@dataclass(frozen=True)
class TraceStage:
    name: str
    start_ms: Optional[float] = None
    end_ms: Optional[float] = None
    duration_ms: Optional[float] = None
    data: JsonDict = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "name": self.name,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "data": _json_safe(self.data),
            "metrics": _json_safe(self.metrics),
        }
        return payload


@dataclass
class TraceContext:
    trace_id: str = field(default_factory=new_trace_id)
    trace_type: str = "query"
    started_ms: float = field(default_factory=_now_ms)
    finished_ms: Optional[float] = None
    stages: List[TraceStage] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def elapsed_ms(self, stage_name: Optional[str] = None) -> float:
        """
        Get elapsed time in milliseconds.
        If stage_name is provided, returns duration of that stage (if found).
        Otherwise returns total trace duration.
        """
        if stage_name:
            for s in self.stages:
                if s.name == stage_name:
                    if s.duration_ms is not None:
                        return float(s.duration_ms)
                    if s.start_ms is not None and s.end_ms is not None:
                        return float(s.end_ms) - float(s.start_ms)
            return 0.0

        end = self.finished_ms if self.finished_ms is not None else _now_ms()
        return float(end) - float(self.started_ms)

    def record_stage(
        self,
        name: str,
        *,
        start_ms: Optional[float] = None,
        end_ms: Optional[float] = None,
        duration_ms: Optional[float] = None,
        data: Optional[JsonDict] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        stage_name = (name or "").strip()
        if not stage_name:
            raise ValueError("stage name must be a non-empty string")

        computed_duration: Optional[float] = duration_ms
        if computed_duration is None and start_ms is not None and end_ms is not None:
            computed_duration = float(end_ms) - float(start_ms)

        stage = TraceStage(
            name=stage_name,
            start_ms=float(start_ms) if start_ms is not None else None,
            end_ms=float(end_ms) if end_ms is not None else None,
            duration_ms=(
                float(computed_duration) if computed_duration is not None else None
            ),
            data=dict(data) if data is not None else {},
            metrics=dict(metrics) if metrics is not None else {},
        )
        self.stages.append(stage)

    def add_metric(self, key: str, value: float) -> None:
        k = (key or "").strip()
        if not k:
            raise ValueError("metric key must be a non-empty string")
        self.metrics[k] = float(value)

    def finish(self) -> None:
        """Mark the trace as finished and calculate total duration."""
        if self.finished_ms is None:
            self.finished_ms = _now_ms()

    def to_dict(self) -> JsonDict:
        """Serialize trace context to a JSON-compatible dictionary."""
        duration_ms = self.elapsed_ms()
        payload: JsonDict = {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_ms": float(self.started_ms),
            "finished_ms": float(self.finished_ms) if self.finished_ms else None,
            "duration_ms": float(duration_ms),
            "total_elapsed_ms": float(duration_ms),
            "stages": [s.to_dict() for s in list(self.stages)],
            "metrics": _json_safe(self.metrics),
        }
        # Validate JSON serializability
        try:
            json.dumps(payload, ensure_ascii=False)
        except (TypeError, ValueError):
            # Fallback for non-serializable data
            payload["metrics"] = str(self.metrics)
            for s in payload["stages"]: # type: ignore
                s["data"] = str(s["data"]) # type: ignore
        return payload
