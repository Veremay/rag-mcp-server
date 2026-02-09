from __future__ import annotations

from typing import Any, List, Optional

from src.core.settings import Settings
from src.ingestion.models import Chunk
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory


class DenseEncoder:
    def __init__(self, settings: Settings, embedding: Optional[BaseEmbedding] = None):
        self._settings = settings
        self._embedding = embedding or EmbeddingFactory.create(settings)

    def encode(
        self, chunks: List[Chunk], trace: Optional[Any] = None
    ) -> List[List[float]]:
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
