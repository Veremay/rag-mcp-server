from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from src.core.trace.trace_context import TraceContext


@dataclass
class TraceCollector:
    log_file: Optional[Union[str, Path]] = None

    def collect(self, trace: TraceContext) -> None:
        if self.log_file is None:
            return

        path = Path(self.log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        trace.finish()
        payload = trace.to_dict()
        line = json.dumps(payload, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
