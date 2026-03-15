"""
向量写入：将 chunks 与稠密向量转为 VectorRecord 并 upsert 到向量库。

chunk_id 由 source_path + section_path + content_hash 生成，保证同内容同 id 可覆盖；
Chroma 等后端要求 metadata 为标量或 JSON 字符串，故对复杂值做 _normalize_metadata_for_chroma。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.core.settings import Settings
from src.ingestion.models import Chunk
from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


def compute_content_hash(text: str) -> str:
    """正文 SHA256，用于与 source_path/section_path 一起生成稳定 chunk_id。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_chunk_id(source_path: str, section_path: str, content_hash: str) -> str:
    """用不可见分隔符拼接三部分再 SHA256，保证同文档同块同 id、重跑摄取可覆盖。"""
    raw = "\x1f".join([source_path or "", section_path or "", content_hash or ""])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class UpsertResult:
    """写入结果：返回本次 upsert 的 VectorRecord 列表，供 Pipeline 记录或 BM25 对齐 chunk_id。"""
    records: List[VectorRecord]


class VectorUpserter:
    """
    根据 chunks + dense_vectors 构建 VectorRecord 并调用 vector_store.upsert。
    支持注入 vector_store；metadata 若为 Chroma 则先做标量/JSON 规范化。
    """

    def __init__(
        self, settings: Settings, vector_store: Optional[BaseVectorStore] = None
    ):
        self._settings = settings
        self._vector_store = vector_store or VectorStoreFactory.create(settings)

    def _normalize_metadata_for_chroma(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Chroma 仅支持标量或需序列化的类型，将 list/dict 等转为 JSON 字符串。"""
        out: Dict[str, Any] = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[k] = v
                continue
            if isinstance(v, (list, dict, tuple, set)):
                out[k] = json.dumps(v, ensure_ascii=False)
                continue
            out[k] = str(v)
        return out

    def build_records(
        self, chunks: Sequence[Chunk], dense_vectors: Sequence[Sequence[float]]
    ) -> List[VectorRecord]:
        """为每个 chunk 生成稳定 chunk_id、补齐 metadata 并可选 Chroma 规范化后构造 VectorRecord。"""
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
            # 写入标量 image_count 便于 Dashboard 统计；Chroma 可能截断大 JSON，images 未必完整返回
            imgs = metadata.get("images")
            if isinstance(imgs, list):
                metadata["image_count"] = len(imgs)
            else:
                metadata.setdefault("image_count", 0)
            backend = getattr(
                getattr(self._settings, "vector_store", None), "backend", None
            )
            if backend is not None and str(backend).lower() == "chroma":
                metadata = self._normalize_metadata_for_chroma(metadata)

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
        """构建 records 后调用 vector_store.upsert，支持传入 trace 做可观测性。"""
        records = self.build_records(chunks, dense_vectors)
        if trace is None:
            self._vector_store.upsert(records)
        else:
            self._vector_store.upsert(records, trace=trace)
        return UpsertResult(records=records)
