from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from src.core.settings import Settings
from src.core.trace.trace_collector import TraceCollector
from src.core.trace.trace_context import TraceContext


@dataclass
class JsonlLogger:
    collector: TraceCollector

    def log_trace(self, trace: TraceContext) -> None:
        self.collector.write(trace)


def create_jsonl_logger(settings: Settings) -> Optional[JsonlLogger]:
    obs = getattr(settings, "observability", None)
    if obs is None or not bool(getattr(obs, "enabled", False)):
        return None

    log_file: Union[str, Path] = str(getattr(obs, "log_file", "") or "")
    if not str(log_file).strip():
        return None

    return JsonlLogger(collector=TraceCollector(log_file=log_file))


def log_trace(trace: TraceContext, *, settings: Settings) -> None:
    logger = create_jsonl_logger(settings)
    if logger is None:
        return
    logger.log_trace(trace)
