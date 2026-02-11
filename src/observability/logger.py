from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.settings import Settings

class JSONFormatter(logging.Formatter):
    """Custom logging formatter that outputs JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        # The message is expected to be a dictionary
        if isinstance(record.msg, dict):
            return json.dumps(record.msg, ensure_ascii=False)
        # Fallback for string messages
        return json.dumps({
            "message": super().format(record),
            "level": record.levelname,
            "timestamp": self.formatTime(record)
        }, ensure_ascii=False)

def get_trace_logger(name: str = "trace_logger") -> logging.Logger:
    """Get the logger configured for trace output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger

def _configure_logger(logger: logging.Logger, log_file: Path) -> None:
    """Internal helper to add file handler to logger if not present."""
    # Check if we already have a file handler for this path
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(str(log_file), encoding="utf-8")
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

def write_trace(trace_dict: Dict[str, Any], settings: Optional[Settings] = None) -> None:
    """
    Write a trace dictionary to the configured JSONL log file.
    
    Args:
        trace_dict: The dictionary representation of the trace.
        settings: Application settings containing observability config.
    """
    if settings is None:
        return

    obs = getattr(settings, "observability", None)
    if obs is None or not bool(getattr(obs, "enabled", False)):
        return

    log_file_str = str(getattr(obs, "log_file", "") or "").strip()
    if not log_file_str:
        return

    logger = get_trace_logger()
    _configure_logger(logger, Path(log_file_str))
    
    # Log the dictionary directly. The JSONFormatter will handle it.
    logger.info(trace_dict)
