from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Dict, List, Optional, Sequence

from src.core.query_engine.dense_retriever import DenseHit
from src.core.query_engine.sparse_retriever import SparseHit
from src.libs.vector_store.base_vector_store import VectorRecord


@dataclass(frozen=True)
class FusionHit:
    chunk_id: str
    score: float
    record: Optional[VectorRecord]
    dense_rank: Optional[int]
    sparse_rank: Optional[int]


class RRFFusion:
    def __init__(self, *, k: int = 60) -> None:
        self._k = int(k)
        if self._k < 0:
            raise ValueError("RRF parameter k must be >= 0")

    def fuse(
        self,
        dense_hits: Sequence[DenseHit],
        sparse_hits: Sequence[SparseHit],
        *,
        top_k: Optional[int] = None,
    ) -> List[FusionHit]:
        effective_top_k = None if top_k is None else int(top_k)
        if effective_top_k is not None and effective_top_k <= 0:
            return []

        dense_ranks: Dict[str, int] = {}
        dense_records: Dict[str, VectorRecord] = {}
        for idx, hit in enumerate(dense_hits, start=1):
            chunk_id = str(hit.record.id)
            if chunk_id in dense_ranks:
                continue
            dense_ranks[chunk_id] = idx
            dense_records[chunk_id] = hit.record

        sparse_ranks: Dict[str, int] = {}
        for idx, sparse_hit in enumerate(sparse_hits, start=1):
            chunk_id = str(sparse_hit.chunk_id)
            if chunk_id in sparse_ranks:
                continue
            sparse_ranks[chunk_id] = idx

        fused_scores: Dict[str, float] = {}

        def add_score(chunk_id: str, rank: int) -> None:
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (
                self._k + rank
            )

        for chunk_id, rank in dense_ranks.items():
            add_score(chunk_id, rank)
        for chunk_id, rank in sparse_ranks.items():
            add_score(chunk_id, rank)

        hits: List[FusionHit] = []
        for chunk_id, score in fused_scores.items():
            hits.append(
                FusionHit(
                    chunk_id=chunk_id,
                    score=float(score),
                    record=dense_records.get(chunk_id),
                    dense_rank=dense_ranks.get(chunk_id),
                    sparse_rank=sparse_ranks.get(chunk_id),
                )
            )

        def best_rank(h: FusionHit) -> float:
            d = inf if h.dense_rank is None else float(h.dense_rank)
            s = inf if h.sparse_rank is None else float(h.sparse_rank)
            return min(d, s)

        hits.sort(key=lambda h: (-h.score, best_rank(h), h.chunk_id))
        return hits if effective_top_k is None else hits[:effective_top_k]


def rrf_fuse(
    dense_hits: Sequence[DenseHit],
    sparse_hits: Sequence[SparseHit],
    *,
    k: int = 60,
    top_k: Optional[int] = None,
) -> List[FusionHit]:
    return RRFFusion(k=k).fuse(dense_hits, sparse_hits, top_k=top_k)
