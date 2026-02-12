from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.loader.file_integrity import FileIntegrityRegistry
from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord


@dataclass
class DocumentInfo:
    source_path: str
    chunk_count: int
    image_count: int
    collection: str


@dataclass
class ChunkDetail:
    id: str
    content: str
    metadata: Dict[str, Any]
    images: List[Dict[str, Any]]


@dataclass
class DocumentDetail:
    source_path: str
    chunks: List[ChunkDetail]


@dataclass
class DeleteResult:
    source_path: str
    success: bool
    deleted_chunks: int
    deleted_images: int
    message: str = ""


@dataclass
class CollectionStats:
    collection_name: str
    total_documents: int
    total_chunks: int
    total_images: int
    backend: str


class DocumentManager:
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
            
            # Check for images in metadata
            images_meta = record.metadata.get("images")
            if images_meta:
                if isinstance(images_meta, str):
                    try:
                        images_list = json.loads(images_meta)
                        if isinstance(images_list, list):
                            doc_info.image_count += len(images_list)
                    except json.JSONDecodeError:
                        pass
                elif isinstance(images_meta, list):
                    doc_info.image_count += len(images_meta)
        
        return list(docs_map.values())

    def get_document_detail(self, source_path: str) -> Optional[DocumentDetail]:
        """
        Get detailed information about a document.

        Args:
            source_path: The source path of the document.

        Returns:
            DocumentDetail object or None if not found.
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
            images_meta = record.metadata.get("images")
            if images_meta:
                if isinstance(images_meta, str):
                    try:
                        parsed = json.loads(images_meta)
                        if isinstance(parsed, list):
                            images = parsed
                    except json.JSONDecodeError:
                        pass
                elif isinstance(images_meta, list):
                    images = images_meta # type: ignore

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
            images_meta = record.metadata.get("images")
            if images_meta:
                images_list: List[Dict[str, Any]] = []
                if isinstance(images_meta, str):
                    try:
                        parsed = json.loads(images_meta)
                        if isinstance(parsed, list):
                            images_list = parsed
                    except json.JSONDecodeError:
                        pass
                elif isinstance(images_meta, list):
                    images_list = images_meta # type: ignore
                
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
