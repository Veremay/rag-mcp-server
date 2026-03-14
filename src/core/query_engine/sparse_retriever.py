"""
稀疏检索器：基于 BM25 的关键词检索。

与稠密检索互补，对精确词匹配、专有名词、数字等更敏感，且不依赖向量模型。
本模块封装 BM25 索引的搜索，统一返回 SparseHit（chunk_id + 分数），
供上层与 DenseHit 做 RRF 等融合。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.settings import Settings
from src.ingestion.storage.bm25_indexer import BM25Hit, BM25Indexer


@dataclass(frozen=True)
class SparseHit:
    """单条稀疏检索命中：仅含 chunk_id 与分数，与 DenseHit 结构解耦便于融合层统一处理。"""
    chunk_id: str
    score: float


class SparseRetriever:
    """
    稀疏检索器：对 query 在 BM25 索引上做 top_k 检索。

    支持注入 indexer 与 base_dir，便于测试或使用不同索引路径；
    collection 优先从参数、再从 filters、最后从配置解析，保证与向量库集合一致。
    """

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
        """
        在 BM25 索引上检索，返回 SparseHit 列表。

        collection 的解析顺序（参数 > filters > 配置）是为了让调用方既能显式指定
        集合，也能通过 query_processor 解析出的 filters 隐式传递，与向量库集合对齐。
        """
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
