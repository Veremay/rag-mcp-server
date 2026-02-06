from typing import List

from src.core.query_engine.dense_retriever import DenseHit
from src.core.query_engine.fusion import RRFFusion
from src.core.query_engine.sparse_retriever import SparseHit
from src.libs.vector_store.base_vector_store import VectorRecord


def _dense(ids: List[str]) -> List[DenseHit]:
    hits: List[DenseHit] = []
    for chunk_id in ids:
        hits.append(
            DenseHit(
                record=VectorRecord(
                    id=chunk_id, embedding=[0.0], content=chunk_id, metadata={}
                ),
                score=1.0,
            )
        )
    return hits


def _sparse(ids: List[str]) -> List[SparseHit]:
    return [SparseHit(chunk_id=chunk_id, score=1.0) for chunk_id in ids]


def test_rrf_fusion_is_deterministic_and_tie_breaks_by_best_rank() -> None:
    fusion = RRFFusion(k=60)

    dense = _dense(["a", "b", "c"])
    sparse = _sparse(["c", "b", "d"])
    fused = fusion.fuse(dense, sparse)

    assert [h.chunk_id for h in fused] == ["c", "b", "a", "d"]


def test_rrf_fusion_respects_top_k() -> None:
    fusion = RRFFusion(k=60)
    fused = fusion.fuse(
        _dense(["a", "b", "c"]),
        _sparse(["c", "b", "d"]),
        top_k=2,
    )
    assert [h.chunk_id for h in fused] == ["c", "b"]


def test_rrf_fusion_handles_empty_inputs() -> None:
    fusion = RRFFusion(k=60)
    assert fusion.fuse([], []) == []
