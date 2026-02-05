from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.core.settings import Settings
from src.ingestion.models import Chunk
from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


def compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_chunk_id(source_path: str, section_path: str, content_hash: str) -> str:
    raw = "\x1f".join([source_path or "", section_path or "", content_hash or ""])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class UpsertResult:
    records: List[VectorRecord]


class VectorUpserter:
    def __init__(
        self, settings: Settings, vector_store: Optional[BaseVectorStore] = None
    ):
        self._settings = settings
        self._vector_store = vector_store or VectorStoreFactory.create(settings)

    def build_records(
        self, chunks: Sequence[Chunk], dense_vectors: Sequence[Sequence[float]]
    ) -> List[VectorRecord]:
        if len(chunks) != len(dense_vectors):
            raise ValueError(
                f"chunks and dense_vectors length mismatch: {len(chunks)} != {len(dense_vectors)}"
            )

        records: List[VectorRecord] = []
        for chunk, vector in zip(chunks, dense_vectors, strict=True):
            source_path = str(chunk.metadata.get("source_path", ""))
            section_path = str(chunk.metadata.get("section_path", ""))
            content_hash = compute_content_hash(chunk.text)
            chunk_id = compute_chunk_id(source_path, section_path, content_hash)

            metadata: Dict[str, Any] = dict(chunk.metadata)
            if chunk.doc_id is not None:
                metadata.setdefault("doc_id", chunk.doc_id)
            metadata.setdefault("source_path", source_path)
            metadata.setdefault("section_path", section_path)
            metadata.setdefault("content_hash", content_hash)

            records.append(
                VectorRecord(
                    id=chunk_id,
                    embedding=list(vector),
                    content=chunk.text,
                    metadata=metadata,
                )
            )

        return records

    def upsert(
        self,
        chunks: Sequence[Chunk],
        dense_vectors: Sequence[Sequence[float]],
        trace: Optional[Any] = None,
    ) -> UpsertResult:
        records = self.build_records(chunks, dense_vectors)
        if trace is None:
            self._vector_store.upsert(records)
        else:
            self._vector_store.upsert(records, trace=trace)
        return UpsertResult(records=records)
