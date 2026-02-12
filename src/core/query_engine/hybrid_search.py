from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.core.query_engine.dense_retriever import DenseHit, DenseRetriever
from src.core.query_engine.fusion import FusionHit, RRFFusion
from src.core.query_engine.query_processor import QueryProcessor
from src.core.query_engine.sparse_retriever import SparseHit, SparseRetriever
from src.core.settings import Settings
from src.libs.vector_store.base_vector_store import VectorRecord


@dataclass(frozen=True)
class HybridSearchHit:
    chunk_id: str
    score: float
    record: VectorRecord
    dense_rank: Optional[int]
    sparse_rank: Optional[int]


class HybridSearch:
    def __init__(
        self,
        settings: Settings,
        *,
        query_processor: Optional[QueryProcessor] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        fusion: Optional[RRFFusion] = None,
    ) -> None:
        self._settings = settings
        self._query_processor = query_processor or QueryProcessor()
        self._dense = dense_retriever or DenseRetriever(settings)
        self._sparse = sparse_retriever or SparseRetriever(settings)
        self._fusion = fusion or RRFFusion()

    def search(
        self,
        query: str,
        *,
        top_k_dense: Optional[int] = None,
        top_k_sparse: Optional[int] = None,
        top_k_final: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[HybridSearchHit]:
        def record_stage(
            name: str,
            *,
            start_ms: float,
            end_ms: float,
            data: Optional[Dict[str, Any]] = None,
            metrics: Optional[Dict[str, float]] = None,
        ) -> None:
            if trace is None:
                return
            fn = getattr(trace, "record_stage", None)
            if not callable(fn):
                return
            fn(
                name,
                start_ms=float(start_ms),
                end_ms=float(end_ms),
                data=dict(data or {}),
                metrics=dict(metrics or {}),
            )

        normalized_query = (query or "").strip()
        if not normalized_query:
            return []

        processed = self._query_processor.process(normalized_query, trace=trace)
        filters = processed.filters
        sparse_query = " ".join(processed.keywords).strip() or normalized_query

        dense_start = time.time() * 1000.0
        dense_hits = self._dense.retrieve(
            normalized_query,
            filters=filters,
            top_k=top_k_dense,
            trace=trace,
        )
        dense_end = time.time() * 1000.0
        record_stage(
            "dense",
            start_ms=dense_start,
            end_ms=dense_end,
            data={
                "query": normalized_query,
                "filters": filters,
                "top_k": top_k_dense,
            },
            metrics={"n_hits": float(len(dense_hits))},
        )

        sparse_start = time.time() * 1000.0
        sparse_hits = self._sparse.retrieve(
            sparse_query,
            filters=filters,
            top_k=top_k_sparse,
            trace=trace,
        )
        sparse_end = time.time() * 1000.0
        record_stage(
            "sparse",
            start_ms=sparse_start,
            end_ms=sparse_end,
            data={
                "query": sparse_query,
                "filters": filters,
                "top_k": top_k_sparse,
            },
            metrics={"n_hits": float(len(sparse_hits))},
        )

        fusion_start = time.time() * 1000.0
        fused = self._fusion.fuse(dense_hits, sparse_hits, top_k=top_k_final)
        hydrated = _hydrate_fusion_hits(fused, dense_hits=dense_hits, dense=self._dense)
        fusion_end = time.time() * 1000.0
        record_stage(
            "fusion",
            start_ms=fusion_start,
            end_ms=fusion_end,
            data={"top_k": top_k_final},
            metrics={
                "n_hits": float(len(hydrated)),
                "n_input": float(len(dense_hits) + len(sparse_hits)),
                "n_output": float(len(hydrated)),
            },
        )
        return hydrated


def _hydrate_fusion_hits(
    fused: Sequence[FusionHit],
    *,
    dense_hits: Sequence[DenseHit],
    dense: DenseRetriever,
) -> List[HybridSearchHit]:
    dense_records: Dict[str, VectorRecord] = {
        str(h.record.id): h.record for h in dense_hits
    }
    out: List[HybridSearchHit] = []

    for h in fused:
        record = h.record or dense_records.get(str(h.chunk_id))
        if record is None:
            record = _resolve_record_from_dense_vector_store(dense, str(h.chunk_id))
        if record is None:
            continue
        out.append(
            HybridSearchHit(
                chunk_id=str(h.chunk_id),
                score=float(h.score),
                record=record,
                dense_rank=h.dense_rank,
                sparse_rank=h.sparse_rank,
            )
        )
    return out


def _resolve_record_from_dense_vector_store(
    dense: DenseRetriever, chunk_id: str
) -> Optional[VectorRecord]:
    vector_store = getattr(dense, "_vector_store", None)
    if vector_store is None:
        return None

    store = getattr(vector_store, "store", None)
    if isinstance(store, dict):
        v = store.get(chunk_id)
        if isinstance(v, VectorRecord):
            return v

    load_all = getattr(vector_store, "_load_all", None)
    if callable(load_all):
        try:
            all_records = load_all()
        except Exception:
            all_records = None
        if isinstance(all_records, dict):
            v = all_records.get(chunk_id)
            if isinstance(v, VectorRecord):
                return v

    collection = getattr(vector_store, "collection", None)
    if collection is not None and hasattr(collection, "get"):
        try:
            raw = collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas", "embeddings"],
            )
        except Exception:
            raw = None
        if isinstance(raw, dict) and raw.get("ids"):
            ids = raw.get("ids") or []
            if ids and ids[0] == chunk_id:
                embeddings = (raw.get("embeddings") or [[]])[0] or []
                documents = (raw.get("documents") or [""])[0] or ""
                metadatas = (raw.get("metadatas") or [{}])[0] or {}
                if (
                    isinstance(embeddings, list)
                    and isinstance(documents, str)
                    and isinstance(metadatas, dict)
                ):
                    return VectorRecord(
                        id=str(chunk_id),
                        embedding=[float(x) for x in embeddings],
                        content=documents,
                        metadata=metadatas,
                    )

    return None
