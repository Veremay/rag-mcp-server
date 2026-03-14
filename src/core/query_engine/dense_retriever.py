"""
稠密检索器：基于向量相似度的语义检索。

与稀疏检索（关键词/BM25）互补，通过把 query 和文档块编码成向量并计算相似度，
能捕捉语义相近但用词不同的匹配，适合「意思相近」的查询。本模块负责 query 编码、
向量库查询以及相似度计算（余弦），并统一返回带归一化分数的 DenseHit 列表。
"""
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
    """单条稠密检索命中文档：包含向量库记录与相似度分数，便于后续融合与重排使用。"""
    record: VectorRecord
    score: float


class DenseRetriever:
    """
    稠密检索器：对 query 做向量化后在向量库中做 top_k 相似度检索。

    支持可选的 embedding/vector_store 注入，便于测试或切换实现；
    未注入时从 Settings 通过工厂创建，保证与配置一致。
    """

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
        """
        对 query 做向量检索，返回按相似度排序的 DenseHit 列表。

        先对 query 做 strip 与空校验，避免无意义调用；top_k 未传时用配置的
        top_k_dense，保证与全局检索策略一致。相似度用余弦并归一化到 [0,1]，
        便于与稀疏分数或后续 RRF 融合时的尺度统一。
        """
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
        if vectors is None:
            return []
        try:
            if len(vectors) == 0:
                return []
        except TypeError:
            return []

        first = vectors[0]
        try:
            if len(first) == 0:
                return []
        except TypeError:
            return []

        query_vector = list(first)
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
    """
    计算两向量的余弦相似度。向量库可能返回内积或未归一化向量，
    本地再算一遍余弦便于与其它检索分数在同一尺度上比较或融合。
    """
    try:
        if len(a) == 0 or len(b) == 0:
            return 0.0
    except TypeError:
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
    """
    将余弦相似度从 [-1, 1] 线性映射到 [0, 1]，便于与稀疏分数或 RRF 分数
    在同一非负范围内比较，并避免负分在排序或加权时产生反直觉结果。
    """
    s = max(-1.0, min(1.0, float(score)))
    return (s + 1.0) / 2.0
