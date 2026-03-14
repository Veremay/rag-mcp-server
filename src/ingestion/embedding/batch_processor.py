"""
批量编码处理器：按 batch_size 将 chunks 分批交给稠密/稀疏编码器并合并结果。

每批记录耗时与条数便于 trace；最后校验 dense/sparse 输出条数与 chunks 一致，
避免编码器漏条或多条导致与向量库/BM25 错位。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.ingestion.models import Chunk


@dataclass(frozen=True)
class BatchMetrics:
    """单批统计：批次序号、条数、耗时(ms)，供 trace 或监控。"""
    batch_index: int
    size: int
    duration_ms: float


@dataclass(frozen=True)
class BatchProcessResult:
    """批量编码结果：与 chunks 同序的稠密/稀疏向量列表及每批的 metrics。"""
    dense_vectors: List[List[float]]
    sparse_vectors: List[Dict[str, float]]
    batches: List[BatchMetrics]


class BatchProcessor:
    """
    按 batch_size 切分 chunks，每批同时调用 dense_encoder 与 sparse_encoder，
    合并后校验条数一致，支持同步 process 与异步 aprocess。
    """

    def __init__(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self._batch_size = batch_size

    def process(
        self,
        chunks: List[Chunk],
        dense_encoder: Any,
        sparse_encoder: Any,
        trace: Optional[Any] = None,
    ) -> BatchProcessResult:
        """同步批量编码：逐批调用 encode，合并后校验条数并返回。"""
        if not chunks:
            return BatchProcessResult(dense_vectors=[], sparse_vectors=[], batches=[])

        dense_vectors: List[List[float]] = []
        sparse_vectors: List[Dict[str, float]] = []
        batches: List[BatchMetrics] = []

        for batch_index, start in enumerate(range(0, len(chunks), self._batch_size)):
            batch = chunks[start : start + self._batch_size]
            t0 = time.perf_counter()

            dense_batch = dense_encoder.encode(batch, trace=trace)
            sparse_batch = sparse_encoder.encode(batch, trace=trace)

            duration_ms = (time.perf_counter() - t0) * 1000.0
            batches.append(
                BatchMetrics(
                    batch_index=batch_index, size=len(batch), duration_ms=duration_ms
                )
            )

            dense_vectors.extend(dense_batch)
            sparse_vectors.extend(sparse_batch)

        if len(dense_vectors) != len(chunks):
            raise ValueError(
                f"BatchProcessor dense output count mismatch: chunks={len(chunks)} vectors={len(dense_vectors)}"
            )
        if len(sparse_vectors) != len(chunks):
            raise ValueError(
                f"BatchProcessor sparse output count mismatch: chunks={len(chunks)} vectors={len(sparse_vectors)}"
            )

        return BatchProcessResult(
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            batches=batches,
        )

    async def aprocess(
        self,
        chunks: List[Chunk],
        dense_encoder: Any,
        sparse_encoder: Any,
        trace: Optional[Any] = None,
    ) -> BatchProcessResult:
        """异步批量编码：逐批调用 aencode，逻辑与 process 一致。"""
        if not chunks:
            return BatchProcessResult(dense_vectors=[], sparse_vectors=[], batches=[])

        dense_vectors: List[List[float]] = []
        sparse_vectors: List[Dict[str, float]] = []
        batches: List[BatchMetrics] = []

        for batch_index, start in enumerate(range(0, len(chunks), self._batch_size)):
            batch = chunks[start : start + self._batch_size]
            t0 = time.perf_counter()

            dense_batch = await dense_encoder.aencode(batch, trace=trace)
            sparse_batch = await sparse_encoder.aencode(batch, trace=trace)

            duration_ms = (time.perf_counter() - t0) * 1000.0
            batches.append(
                BatchMetrics(
                    batch_index=batch_index, size=len(batch), duration_ms=duration_ms
                )
            )

            dense_vectors.extend(dense_batch)
            sparse_vectors.extend(sparse_batch)

        if len(dense_vectors) != len(chunks):
            raise ValueError(
                f"BatchProcessor dense output count mismatch: chunks={len(chunks)} vectors={len(dense_vectors)}"
            )
        if len(sparse_vectors) != len(chunks):
            raise ValueError(
                f"BatchProcessor sparse output count mismatch: chunks={len(chunks)} vectors={len(sparse_vectors)}"
            )

        return BatchProcessResult(
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            batches=batches,
        )
