from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.core.settings import Settings
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


@dataclass(frozen=True)
class DenseHit:
    record: VectorRecord
    score: float


class DenseRetriever:
    def __init__(
        self,
        settings: Settings,
        *,
        embedding: Optional[BaseEmbedding] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ) -> None:
        self._settings = settings
        self._embedding = embedding or EmbeddingFactory.create(settings)
        self._vector_store = vector_store or VectorStoreFactory.create(settings)

    def retrieve(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[DenseHit]:
        normalized_query = (query or "").strip()
        if not normalized_query:
            return []

        effective_top_k = (
            int(top_k)
            if top_k is not None
            else int(self._settings.retrieval.top_k_dense)
        )
        if effective_top_k <= 0:
            return []

        if trace is None:
            vectors = self._embedding.embed([normalized_query])
        else:
            vectors = self._embedding.embed([normalized_query], trace=trace)
        if not vectors or not vectors[0]:
            return []

        query_vector = list(vectors[0])
        if trace is None:
            records = self._vector_store.query(
                vector=query_vector, top_k=effective_top_k, filters=filters
            )
        else:
            records = self._vector_store.query(
                vector=query_vector, top_k=effective_top_k, filters=filters, trace=trace
            )

        hits: List[DenseHit] = []
        for r in records:
            raw = _cosine_similarity(query_vector, r.embedding)
            hits.append(DenseHit(record=r, score=_normalize_cosine(raw)))
        return hits


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0

    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        fx = float(x)
        fy = float(y)
        dot += fx * fy
        na += fx * fx
        nb += fy * fy
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _normalize_cosine(score: float) -> float:
    s = max(-1.0, min(1.0, float(score)))
    return (s + 1.0) / 2.0
