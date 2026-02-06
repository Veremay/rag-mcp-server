from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import Settings
from src.ingestion.storage.bm25_indexer import BM25Hit, BM25Indexer


@dataclass(frozen=True)
class SparseHit:
    chunk_id: str
    score: float


class SparseRetriever:
    def __init__(
        self,
        settings: Settings,
        *,
        indexer: Optional[BM25Indexer] = None,
        base_dir: str | Path = "data/db/bm25",
    ) -> None:
        self._settings = settings
        self._indexer = indexer or BM25Indexer(base_dir=base_dir)

    def retrieve(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None,
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[SparseHit]:
        normalized_query = (query or "").strip()
        if not normalized_query:
            return []

        effective_top_k = (
            int(top_k)
            if top_k is not None
            else int(self._settings.retrieval.top_k_sparse)
        )
        if effective_top_k <= 0:
            return []

        resolved_collection = (
            (collection or "").strip()
            or (str((filters or {}).get("collection", "")).strip())
            or str(self._settings.vector_store.collection_name)
        )

        hits: List[BM25Hit] = self._indexer.search(
            collection=resolved_collection,
            query=normalized_query,
            top_k=effective_top_k,
        )
        return [SparseHit(chunk_id=h.chunk_id, score=float(h.score)) for h in hits]
