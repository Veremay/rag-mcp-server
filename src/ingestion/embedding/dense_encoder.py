"""
稠密编码器：将 chunks 的 text 列表交给 BaseEmbedding 做向量化。

支持同步 encode 与异步 aencode，并校验返回向量条数与维度一致，
避免与 VectorUpserter 的 chunks 错位或维度不统一导致落库失败。
"""
from __future__ import annotations

from typing import Any, List, Optional

from src.core.settings import Settings
from src.ingestion.models import Chunk
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory


class DenseEncoder:
    """
    封装 BaseEmbedding，从 chunks 抽 text 批量调用 embed/aembed。
    支持注入 embedding 便于测试；返回前 _validate 条数与维度。
    """

    def __init__(self, settings: Settings, embedding: Optional[BaseEmbedding] = None):
        self._settings = settings
        self._embedding = embedding or EmbeddingFactory.create(settings)

    def encode(
        self, chunks: List[Chunk], trace: Optional[Any] = None
    ) -> List[List[float]]:
        """批量稠密编码，支持 trace 传递；空列表直接返回。"""
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        if trace is None:
            vectors = self._embedding.embed(texts)
        else:
            vectors = self._embedding.embed(texts, trace=trace)

        self._validate(chunks, vectors)
        return vectors

    async def aencode(
        self, chunks: List[Chunk], trace: Optional[Any] = None
    ) -> List[List[float]]:
        """异步批量稠密编码，与 encode 行为一致。"""
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        if trace is None:
            vectors = await self._embedding.aembed(texts)
        else:
            vectors = await self._embedding.aembed(texts, trace=trace)

        self._validate(chunks, vectors)
        return vectors

    @staticmethod
    def _validate(chunks: List[Chunk], vectors: List[List[float]]) -> None:
        """校验向量条数等于 chunks、非空且维度一致，避免落库时错位或报错。"""
        if len(vectors) != len(chunks):
            raise ValueError(
                f"DenseEncoder vector count mismatch: chunks={len(chunks)} vectors={len(vectors)}"
            )

        if not vectors:
            return

        dim = len(vectors[0])
        if dim <= 0:
            raise ValueError("DenseEncoder produced empty vectors")

        for idx, vec in enumerate(vectors):
            if len(vec) != dim:
                raise ValueError(
                    f"DenseEncoder vector dimension mismatch at index {idx}: expected={dim} got={len(vec)}"
                )
