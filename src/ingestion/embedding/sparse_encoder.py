from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from src.ingestion.models import Chunk


class SparseEncoder:
    def __init__(self, token_pattern: str = r"\w+"):
        self._token_re = re.compile(token_pattern, flags=re.UNICODE)

    def encode(
        self, chunks: List[Chunk], trace: Optional[Any] = None
    ) -> List[Dict[str, float]]:
        if not chunks:
            return []

        outputs: List[Dict[str, float]] = []
        for chunk in chunks:
            tokens = [t.lower() for t in self._token_re.findall(chunk.text or "")]
            counts = Counter(tokens)
            outputs.append({term: float(freq) for term, freq in counts.items()})

        if len(outputs) != len(chunks):
            raise ValueError(
                f"SparseEncoder output count mismatch: chunks={len(chunks)} outputs={len(outputs)}"
            )

        return outputs

    async def aencode(
        self, chunks: List[Chunk], trace: Optional[Any] = None
    ) -> List[Dict[str, float]]:
        return self.encode(chunks, trace=trace)
