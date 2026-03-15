"""
文档管理：按集合/来源列举文档、查看详情、删除文档及统计。

基于向量库 + BM25 + 图片存储 + 文件完整性注册表，保证删除时三处一致清理，
列表与详情兼容 source_path / source、images 的 list 或 JSON 字符串等形态。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.loader.file_integrity import FileIntegrityRegistry
from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord


def _image_count_from_metadata_value(images_meta: Any) -> int:
    """
    从 metadata 的 images 字段解析出图片数量。
    Chroma 存的是 JSON 字符串；兼容 list、可求长序列，以便正确统计。
    """
    return len(_images_list_from_metadata_value(images_meta))


def _images_list_from_metadata_value(images_meta: Any) -> List[Dict[str, Any]]:
    """
    从 metadata 的 images 字段解析出图片项列表（用于详情展示与删除）。
    兼容 Chroma 存的 JSON 字符串及 list、bytes 等。
    """
    if images_meta is None:
        return []
    if isinstance(images_meta, bytes):
        try:
            images_meta = images_meta.decode("utf-8")
        except Exception:
            return []
    if isinstance(images_meta, str):
        s = (images_meta or "").strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
            return []
        except json.JSONDecodeError:
            return []
    if isinstance(images_meta, list):
        return [x for x in images_meta if isinstance(x, dict)]
    if hasattr(images_meta, "__iter__") and not isinstance(images_meta, (str, dict)):
        return [x for x in images_meta if isinstance(x, dict)]
    return []


def _get_images_value_from_record_metadata(metadata: Dict[str, Any]) -> Any:
    """
    从单条 record 的 metadata 中取出 images 值。
    兼容键名 "images" / "Images"（Chroma 或序列化可能改变键名）。
    """
    v = metadata.get("images")
    if v is not None:
        return v
    for k, val in metadata.items():
        if k is not None and str(k).lower() == "images":
            return val
    return None


@dataclass
class DocumentInfo:
    """列表项：按 source_path 聚合的文档摘要，含块数与图片数便于 Dashboard 展示。"""
    source_path: str
    chunk_count: int
    image_count: int
    collection: str


@dataclass
class ChunkDetail:
    """单块详情：id、正文、metadata、解析后的 images 列表，供详情页展示。"""
    id: str
    content: str
    metadata: Dict[str, Any]
    images: List[Dict[str, Any]]


@dataclass
class DocumentDetail:
    """文档详情：来源路径与所属全部块，用于「查看文档」类接口。"""
    source_path: str
    chunks: List[ChunkDetail]


@dataclass
class DeleteResult:
    """删除结果：是否成功、删除的块数/图片数及提示信息，便于前端或日志反馈。"""
    source_path: str
    success: bool
    deleted_chunks: int
    deleted_images: int
    message: str = ""


@dataclass
class CollectionStats:
    """集合统计：文档数、块数、图片数及后端类型，供 Dashboard 或管理接口。"""
    collection_name: str
    total_documents: int
    total_chunks: int
    total_images: int
    backend: str


class DocumentManager:
    """
    统一封装向量库、BM25、图片存储与完整性注册表，提供列举/详情/删除/统计。
    删除时先查向量库再删图片、BM25、向量库、注册表，保证数据一致且可安全重跑摄取。
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        bm25_indexer: BM25Indexer,
        image_storage: ImageStorage,
        file_integrity: FileIntegrityRegistry,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_indexer = bm25_indexer
        self.image_storage = image_storage
        self.file_integrity = file_integrity

    def list_documents(self, collection: str = "") -> List[DocumentInfo]:
        """
        List all documents in the vector store.
        
        Args:
            collection: Optional collection name to filter/verify. 
                        Note: The vector_store instance usually binds to a specific collection.
                        This argument is mainly for verification or if we support multi-collection switching.
        
        Returns:
            List of DocumentInfo objects.

        按 metadata 聚合 source_path，兼容 collection 过滤与 images 的 list/JSON 字符串，
        便于 Dashboard 展示各文档块数与图片数。
        """
        # Get all records from vector store
        # Note: This loads all metadata into memory. Optimized for local usage.
        records = self.vector_store.get_records_by_metadata({})
        
        docs_map: Dict[str, DocumentInfo] = {}
        
        default_collection = getattr(self.vector_store, "collection_name", "default")

        for record in records:
            # Determine collection for this record from metadata, fallback to store's collection
            rec_collection = str(record.metadata.get("collection") or default_collection)
            
            # Filter if specific collection requested
            if collection and rec_collection != collection:
                continue

            source = str(record.metadata.get("source_path") or record.metadata.get("source") or "unknown")
            
            if source not in docs_map:
                docs_map[source] = DocumentInfo(
                    source_path=source,
                    chunk_count=0,
                    image_count=0,
                    collection=rec_collection
                )
            
            doc_info = docs_map[source]
            doc_info.chunk_count += 1
            
            # 统计 images：优先用标量 image_count（ingest 时写入，不受 Chroma 大值截断影响），否则解析 images
            n_imgs = record.metadata.get("image_count")
            if n_imgs is not None and isinstance(n_imgs, (int, float)):
                doc_info.image_count += int(n_imgs)
            else:
                images_meta = _get_images_value_from_record_metadata(record.metadata)
                if images_meta is not None:
                    doc_info.image_count += _image_count_from_metadata_value(images_meta)
        
        return list(docs_map.values())

    def get_document_detail(self, source_path: str) -> Optional[DocumentDetail]:
        """
        Get detailed information about a document.

        Args:
            source_path: The source path of the document.

        Returns:
            DocumentDetail object or None if not found.

        先按 source_path 查，无结果再按 source 查，兼容不同摄取阶段的 metadata 键名。
        """
        records = self.vector_store.get_records_by_metadata({"source_path": source_path})
        if not records:
            # Try "source" as fallback
            records = self.vector_store.get_records_by_metadata({"source": source_path})
            if not records:
                return None
        
        chunks: List[ChunkDetail] = []
        for record in records:
            images: List[Dict[str, Any]] = []
            images = _images_list_from_metadata_value(_get_images_value_from_record_metadata(record.metadata))

            chunks.append(ChunkDetail(
                id=record.id,
                content=record.content,
                metadata=record.metadata,
                images=images
            ))
            
        return DocumentDetail(source_path=source_path, chunks=chunks)

    def delete_document(self, source_path: str, collection: str) -> DeleteResult:
        """
        Delete a document and all its associated data (chunks, images, index entries).

        Args:
            source_path: The source path of the document to delete.
            collection: The collection name.

        Returns:
            DeleteResult object.

        顺序：查向量库 -> 删图片(按 metadata.images) -> 删 BM25 -> 删向量 -> 删注册表，
        避免残留引用或重复删除。
        """
        # 1. Find records in Vector Store
        records = self.vector_store.get_records_by_metadata({"source_path": source_path})
        if not records:
            records = self.vector_store.get_records_by_metadata({"source": source_path})
        
        if not records:
            # Even if not in VectorStore, try to remove from FileIntegrity to be safe/clean
            removed_registry = self.file_integrity.remove_record(source_path)
            return DeleteResult(
                source_path=source_path,
                success=removed_registry,
                deleted_chunks=0,
                deleted_images=0,
                message="Document not found in vector store." + (" Removed from registry." if removed_registry else "")
            )

        chunk_ids = [r.id for r in records]
        
        # 2. Find and delete images
        deleted_images_count = 0
        for record in records:
            images_list = _images_list_from_metadata_value(_get_images_value_from_record_metadata(record.metadata))
            for img in images_list:
                    img_id = img.get("image_id")
                    if img_id:
                        if self.image_storage.delete(collection=collection, image_id=str(img_id)):
                            deleted_images_count += 1

        # 3. Delete from BM25 Index
        self.bm25_indexer.remove_document(collection=collection, chunk_ids=chunk_ids)

        # 4. Delete from Vector Store
        # We can use chunk IDs or metadata. Using IDs is safer/faster if we have them.
        # But base_vector_store doesn't have delete_by_ids.
        # So we use delete_by_metadata.
        # We already know source_path works.
        self.vector_store.delete_by_metadata({"source_path": source_path})
        # Double check with "source" if we found records via that
        if any("source" in r.metadata and r.metadata["source"] == source_path for r in records):
             self.vector_store.delete_by_metadata({"source": source_path})

        # 5. Remove from FileIntegrityRegistry
        self.file_integrity.remove_record(source_path)

        return DeleteResult(
            source_path=source_path,
            success=True,
            deleted_chunks=len(chunk_ids),
            deleted_images=deleted_images_count,
            message="Successfully deleted document."
        )

    def get_collection_stats(self, collection: str = "") -> CollectionStats:
        """
        Get statistics for the collection.

        通过 list_documents 聚合文档/块/图片数，backend 取自向量库实例以便展示当前后端。
        """
        # Rely on VectorStore stats + aggregation
        # Or simpler: list_documents and sum up
        docs = self.list_documents(collection)
        total_chunks = sum(d.chunk_count for d in docs)
        total_images = sum(d.image_count for d in docs)
        
        backend = getattr(self.vector_store, "backend", "unknown")
        # Try to get backend from settings if possible, but here we only have the instance.
        # Maybe check class name
        if not backend or backend == "unknown":
            backend = self.vector_store.__class__.__name__

        return CollectionStats(
            collection_name=collection or getattr(self.vector_store, "collection_name", "default"),
            total_documents=len(docs),
            total_chunks=total_chunks,
            total_images=total_images,
            backend=backend
        )
