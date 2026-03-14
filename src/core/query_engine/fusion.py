"""
融合模块：将稠密检索与稀疏检索的命中列表用 RRF 合并为单一排序列表。

RRF（Reciprocal Rank Fusion）不依赖原始分数尺度，只根据排名加权求和，
避免稠密/稀疏分数不可比的问题；k 为平滑常数，常用 60 以平衡高低排名差异。
"""
from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Dict, List, Optional, Sequence

from src.core.query_engine.dense_retriever import DenseHit
from src.core.query_engine.sparse_retriever import SparseHit
from src.libs.vector_store.base_vector_store import VectorRecord


@dataclass(frozen=True)
class FusionHit:
    """
    融合后的一条命中：chunk_id、RRF 分数、可选的 record（来自稠密侧）、
    dense_rank/sparse_rank 便于调试或展示双路排名。
    """
    chunk_id: str
    score: float
    record: Optional[VectorRecord]
    dense_rank: Optional[int]
    sparse_rank: Optional[int]


class RRFFusion:
    """
    RRF 融合器：按 1/(k+rank) 对稠密与稀疏的排名加权求和，再按分数降序、排名、chunk_id 排序。

    k 越大高低排名差异越小，默认 60 与常见文献一致；只出现在一侧的 chunk 也会得到
    对应侧的贡献，保证稠密/稀疏互补时不会漏掉单路高分项。
    """

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
        """
        对稠密与稀疏命中做 RRF 融合，返回按融合分数排序的 FusionHit 列表。

        先按 chunk_id 收集两边的排名与稠密侧 record，再按 RRF 公式累加分数；
        排序时用 (-score, best_rank, chunk_id) 保证分数优先、同分时排名靠前优先、再按 id 稳定序。
        """
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
    """便捷函数：用默认 k 创建 RRFFusion 并执行一次融合，供无需复用实例的调用方使用。"""
    return RRFFusion(k=k).fuse(dense_hits, sparse_hits, top_k=top_k)
