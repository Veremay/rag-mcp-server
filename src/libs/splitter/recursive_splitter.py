from __future__ import annotations

from typing import Any, List, Optional

from src.libs.splitter.base_splitter import BaseSplitter, TraceContext

_LCRecursiveSplitter: Any
try:
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter as _LCRecursiveSplitter,
    )
except ModuleNotFoundError:
    _LCRecursiveSplitter = None


class RecursiveSplitter(BaseSplitter):
    def __init__(self, settings: Any):
        splitter_config = settings.ingestion.splitter
        self.chunk_size = splitter_config.chunk_size
        self.chunk_overlap = splitter_config.chunk_overlap
        self._splitter = None
        if _LCRecursiveSplitter is not None:
            self._splitter = _LCRecursiveSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )

    def split_text(
        self, text: str, trace: Optional[TraceContext] = None, **kwargs: Any
    ) -> List[str]:
        if self._splitter is None:
            return self._fallback_split(text)
        return self._splitter.split_text(text)

    def _fallback_split(self, text: str) -> List[str]:
        if not text:
            return []

        chunk_size = int(self.chunk_size) if int(self.chunk_size) > 0 else 1
        overlap = int(self.chunk_overlap)
        if overlap < 0:
            overlap = 0
        if overlap >= chunk_size:
            overlap = max(chunk_size - 1, 0)

        step = max(chunk_size - overlap, 1)
        chunks: List[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks
