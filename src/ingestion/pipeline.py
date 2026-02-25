import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.core.settings import Settings
from src.core.trace.trace_context import TraceContext
from src.ingestion.embedding.batch_processor import BatchProcessor, BatchProcessResult
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.models import Chunk, Document
from src.ingestion.storage.bm25_indexer import BM25Index, BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.ingestion.storage.vector_upserter import UpsertResult, VectorUpserter
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.file_integrity import FileIntegrityRegistry
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.splitter.splitter_factory import SplitterFactory
from src.libs.vector_store.base_vector_store import BaseVectorStore
from src.observability.logger import write_trace as _write_trace


@dataclass(frozen=True)
class SplitResult:
    document: Document
    chunks: List[Chunk]


@dataclass(frozen=True)
class IngestResult:
    skipped: bool
    file_hash: str
    document: Optional[Document]
    chunks: List[Chunk]
    batch: Optional[BatchProcessResult]
    upsert: Optional[UpsertResult]
    bm25: Optional[BM25Index]


class IngestionPipeline:
    """
    Minimal ingestion pipeline for MVP.

    This pipeline currently integrates the configured splitter and produces Chunk objects
    from a Document.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        integrity: Optional[FileIntegrityRegistry] = None,
        loader: Optional[BaseLoader] = None,
        transforms: Optional[Sequence[Any]] = None,
        dense_encoder: Optional[Any] = None,
        sparse_encoder: Optional[Any] = None,
        batch_processor: Optional[BatchProcessor] = None,
        vector_store: Optional[BaseVectorStore] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
        image_storage: Optional[ImageStorage] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            settings: Global application settings.
        """
        self._settings = settings
        self._integrity = integrity or FileIntegrityRegistry()
        self._loader = loader
        self._transforms = transforms
        self._dense_encoder = dense_encoder
        self._sparse_encoder = sparse_encoder
        self._batch_processor = batch_processor
        self._vector_store = vector_store
        self._bm25_indexer = bm25_indexer
        self._image_storage = image_storage

    def ingest(
        self,
        *,
        collection: str,
        file_path: str | Path,
        original_filename: Optional[str] = None,
        force: bool = False,
        trace: Optional[Any] = None,
        on_progress: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> IngestResult:
        effective_trace: Any = trace if trace is not None else TraceContext(trace_type="ingestion")
        # Ensure trace type is set to ingestion if passed in but empty type
        if isinstance(effective_trace, TraceContext) and effective_trace.trace_type == "query":
             effective_trace.trace_type = "ingestion"

        def record_stage(
            name: str,
            *,
            start_ms: float,
            end_ms: float,
            data: Optional[Dict[str, Any]] = None,
            metrics: Optional[Dict[str, float]] = None,
        ) -> None:
            fn = getattr(effective_trace, "record_stage", None)
            if not callable(fn):
                return
            fn(
                name,
                start_ms=float(start_ms),
                end_ms=float(end_ms),
                data=dict(data or {}),
                metrics=dict(metrics or {}),
            )

        def flush_trace() -> None:
            if not isinstance(effective_trace, TraceContext):
                return
            try:
                # Must finish trace before serializing
                effective_trace.finish()
                _write_trace(effective_trace.to_dict(), settings=self._settings)
            except Exception:
                return

        path = Path(file_path)

        if on_progress:
            on_progress("start", {"path": str(path)})

        integrity_start = time.time() * 1000.0
        try:
            file_hash = self._integrity.compute_sha256(path)
        except Exception as e:
            raise RuntimeError("IngestionPipeline integrity step failed") from e
        integrity_end = time.time() * 1000.0
        record_stage(
            "integrity",
            start_ms=integrity_start,
            end_ms=integrity_end,
            data={
                "path": str(path),
                "original_filename": original_filename or path.name,
                "hash": file_hash,
            },
        )

        if not force and self._integrity.should_skip(file_hash):
            if on_progress:
                on_progress("skipped", {"hash": file_hash})
            flush_trace()
            return IngestResult(
                skipped=True,
                file_hash=file_hash,
                document=None,
                chunks=[],
                batch=None,
                upsert=None,
                bm25=None,
            )

        if on_progress:
            on_progress("hash_calculated", {"hash": file_hash})

        load_start = time.time() * 1000.0
        try:
            loader = self._resolve_loader(path)
            document = loader.load(path)
        except Exception as e:
            raise RuntimeError("IngestionPipeline loader step failed") from e
        load_end = time.time() * 1000.0
        record_stage(
            "load",
            start_ms=load_start,
            end_ms=load_end,
            data={
                "doc_id": getattr(document, "id", None),
                "method": loader.__class__.__name__ if loader else "unknown",
            },
        )

        # Process images: move to storage and update references
        image_store_start = time.time() * 1000.0
        try:
            self._process_images(document, collection)
        except Exception as e:
            # If image processing fails, log it but don't fail the whole ingestion?
            # Or should we fail? Given images are critical for multimodal, failing is safer.
            raise RuntimeError("IngestionPipeline image storage step failed") from e
        image_store_end = time.time() * 1000.0
        record_stage(
            "image_store",
            start_ms=image_store_start,
            end_ms=image_store_end,
            data={"count": len(document.metadata.get("images", []))},
            metrics={"count": len(document.metadata.get("images", []))}
        )

        split_start = time.time() * 1000.0
        try:
            split = self.split(document, trace=effective_trace)
            chunks = split.chunks
            for c in chunks:
                meta = c.metadata
                if isinstance(meta, dict):
                    meta.setdefault("collection", str(collection))
            if on_progress:
                on_progress("split", {"chunks": chunks})
        except Exception as e:
            raise RuntimeError("IngestionPipeline splitter step failed") from e
        split_end = time.time() * 1000.0
        
        # Capture split preview
        split_data = {"method": self._settings.ingestion.splitter.provider}
        if chunks:
            preview = []
            for c in chunks[:3]:
                preview.append({
                    "id": c.id,
                    "text": c.text[:200] + "..." if len(c.text) > 200 else c.text,
                    "metadata": c.metadata
                })
            split_data["chunks_preview"] = preview

        record_stage(
            "split",
            start_ms=split_start,
            end_ms=split_end,
            data=split_data,
            metrics={"n_chunks": float(len(chunks))},
        )

        transform_start = time.time() * 1000.0
        try:
            chunks = self._apply_transforms(chunks, trace=effective_trace)
            if on_progress:
                on_progress("transformed", {"chunks": chunks})
        except Exception as e:
            raise RuntimeError("IngestionPipeline transform step failed") from e
        transform_end = time.time() * 1000.0
        
        # Capture transform preview
        transform_data = {"method": "chain", "transforms": [t.__class__.__name__ for t in (self._transforms or [])]}
        if chunks:
            preview = []
            for c in chunks[:3]:
                preview.append({
                    "id": c.id,
                    "text": c.text[:200] + "..." if len(c.text) > 200 else c.text,
                    "metadata": c.metadata
                })
            transform_data["chunks_preview"] = preview

        record_stage(
            "transform",
            start_ms=transform_start,
            end_ms=transform_end,
            data=transform_data,
            metrics={"n_chunks": float(len(chunks))},
        )

        # Ensure collection is in metadata for filtering
        for chunk in chunks:
            chunk.metadata["collection"] = collection

        encode_start = time.time() * 1000.0
        try:
            batch = self._encode(chunks, trace=effective_trace)
            if on_progress:
                on_progress("encoded", {"batch": batch})
        except Exception as e:
            raise RuntimeError("IngestionPipeline embedding step failed") from e
        encode_end = time.time() * 1000.0
        
        # Capture encode preview
        encode_data = {
            "dense_model": self._settings.embedding.model,
            "sparse_model": "bm25",
        }
        if batch and batch.dense_vectors and len(batch.dense_vectors) > 0:
            encode_data["embedding_dim"] = len(batch.dense_vectors[0])

        record_stage(
            "encode",
            start_ms=encode_start,
            end_ms=encode_end,
            data=encode_data,
            metrics={
                "n_dense": float(len(batch.dense_vectors)),
                "n_sparse": float(len(batch.sparse_vectors)),
            },
        )

        upsert_start = time.time() * 1000.0
        try:
            upsert = self._upsert(chunks, batch.dense_vectors, trace=effective_trace)
            if on_progress:
                on_progress("upserted", {"upsert": upsert})
        except Exception as e:
            raise RuntimeError("IngestionPipeline vector upsert step failed") from e
        upsert_end = time.time() * 1000.0
        
        # Capture upsert preview
        upsert_data = {"method": self._settings.vector_store.backend}
        if upsert and upsert.records:
             upsert_data["upserted_ids"] = [r.id for r in upsert.records[:10]]

        record_stage(
            "upsert",
            start_ms=upsert_start,
            end_ms=upsert_end,
            data=upsert_data,
            metrics={"n_records": float(len(upsert.records))},
        )

        bm25_start = time.time() * 1000.0
        try:
            chunk_ids = [r.id for r in upsert.records]
            bm25 = self._build_bm25(
                collection=collection,
                chunk_ids=chunk_ids,
                sparse_vectors=batch.sparse_vectors,
            )
            if on_progress:
                on_progress("bm25_built", {})
        except Exception as e:
            raise RuntimeError("IngestionPipeline bm25 step failed") from e
        bm25_end = time.time() * 1000.0
        record_stage(
            "bm25",
            start_ms=bm25_start,
            end_ms=bm25_end,
            metrics={"n_terms": float(len(getattr(bm25, "postings", []) or []))},
        )

        finalize_start = time.time() * 1000.0
        try:
            self._integrity.mark_success(file_hash)
            if on_progress:
                on_progress("complete", {})
        except Exception as e:
            raise RuntimeError("IngestionPipeline finalize step failed") from e
        finalize_end = time.time() * 1000.0
        record_stage(
            "finalize",
            start_ms=finalize_start,
            end_ms=finalize_end,
        )
        flush_trace()

        return IngestResult(
            skipped=False,
            file_hash=file_hash,
            document=document,
            chunks=chunks,
            batch=batch,
            upsert=upsert,
            bm25=bm25,
        )

    def split(self, document: Document, trace: Optional[Any] = None) -> SplitResult:
        """
        Split a Document into Chunks using the configured splitter implementation.

        Args:
            document: Input document to split.
            trace: Optional trace context for observability.

        Returns:
            SplitResult containing the original document and generated chunks.
        """
        splitter = SplitterFactory.create(self._settings)
        chunk_texts = splitter.split_text(document.text, trace=trace)

        chunks: List[Chunk] = []
        for idx, text in enumerate(chunk_texts):
            chunks.append(
                Chunk(
                    text=text,
                    metadata={**document.metadata, "chunk_index": idx},
                    doc_id=document.id,
                )
            )

        return SplitResult(document=document, chunks=chunks)

    def _resolve_loader(self, path: Path) -> BaseLoader:
        if self._loader is not None:
            return self._loader

        ext = path.suffix.lower()
        if ext == ".pdf":
            return PdfLoader(settings=self._settings.ingestion.loader)

        raise ValueError(f"Unsupported file extension: {ext}")

    def _apply_transforms(
        self, chunks: List[Chunk], *, trace: Optional[Any]
    ) -> List[Chunk]:
        transforms = list(self._transforms) if self._transforms is not None else None
        if transforms is None:
            transforms = [
                ChunkRefiner(self._settings),
                MetadataEnricher(self._settings),
                ImageCaptioner(self._settings),
            ]

        current = chunks
        for t in transforms:
            current = t.transform(current, trace=trace)
        return current

    def _encode(
        self, chunks: List[Chunk], *, trace: Optional[Any]
    ) -> BatchProcessResult:
        dense_encoder = self._dense_encoder or DenseEncoder(self._settings)
        sparse_encoder = self._sparse_encoder or SparseEncoder()
        # Default batch size reduced to 10 to comply with strict API limits (e.g. SiliconFlow/OpenAI)
        batcher = self._batch_processor or BatchProcessor(batch_size=10)
        return batcher.process(
            chunks,
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            trace=trace,
        )

    def _upsert(
        self,
        chunks: Sequence[Chunk],
        dense_vectors: Sequence[Sequence[float]],
        *,
        trace: Optional[Any],
    ) -> UpsertResult:
        upserter = VectorUpserter(self._settings, vector_store=self._vector_store)
        return upserter.upsert(chunks, dense_vectors, trace=trace)

    def _build_bm25(
        self,
        *,
        collection: str,
        chunk_ids: Sequence[str],
        sparse_vectors: Sequence[dict[str, float]],
    ) -> BM25Index:
        indexer = self._bm25_indexer or BM25Indexer()
        return indexer.upsert(
            collection=collection, chunk_ids=chunk_ids, sparse_vectors=sparse_vectors
        )

    def _process_images(self, document: Document, collection: str) -> None:
        """
        Process images extracted from the document:
        1. Move images to ImageStorage (organized by collection).
        2. Update image references in document text and metadata.
        3. Ensure metadata contains detailed image info for Dashboard.
        """
        image_refs = document.metadata.get("image_refs", [])
        if not image_refs:
            return

        storage = self._image_storage or ImageStorage()
        
        # New list for updated references
        new_image_refs = []
        # Detailed image info for Dashboard (List[Dict])
        detailed_images = []
        
        updated_text = document.text
        
        for ref_path in image_refs:
            path_obj = Path(ref_path)
            # Handle potential relative paths (e.g. data/images/...)
            if not path_obj.is_absolute():
                path_obj = path_obj.resolve()
                
            if not path_obj.exists():
                # If absolute path doesn't exist, try relative to CWD
                if not path_obj.is_absolute():
                     path_obj = Path.cwd() / ref_path
                
                if not path_obj.exists():
                    continue
                
            image_id = path_obj.stem
            
            try:
                # Add to storage (move file to collection folder)
                stored_path = storage.add_file(
                    file_path=path_obj,
                    collection=collection,
                    image_id=image_id,
                    move=True
                )
                
                # Update reference in text
                # PdfLoader uses str(path) which might differ in slashes on Windows
                # We try to replace both original ref string and resolved path string
                original_ref_str = str(ref_path)
                stored_path_str = str(stored_path.resolve())
                
                if original_ref_str in updated_text:
                    updated_text = updated_text.replace(original_ref_str, stored_path_str)
                elif str(path_obj) in updated_text:
                    updated_text = updated_text.replace(str(path_obj), stored_path_str)
                
                new_image_refs.append(stored_path_str)
                detailed_images.append({
                    "image_id": image_id,
                    "path": stored_path_str,
                    "collection": collection,
                    "original_path": str(path_obj)
                })
            except Exception as e:
                # Log error but continue processing other images
                print(f"Error processing image {ref_path}: {e}")
                continue
            
        document.text = updated_text
        document.metadata["image_refs"] = new_image_refs
        # Key fix for Dashboard: Set "images" metadata
        document.metadata["images"] = detailed_images

    def _store_images(self, *, collection: str, document: Document) -> None:
        images = document.metadata.get("images")
        if not isinstance(images, list) or not images:
            return

        storage = self._image_storage or ImageStorage()
        for item in images:
            if not isinstance(item, dict):
                continue
            image_id = item.get("image_id")
            data = item.get("data")
            ext = item.get("ext") or item.get("extension") or ".png"
            if not isinstance(image_id, str) or not isinstance(
                data, (bytes, bytearray)
            ):
                continue
            storage.save(
                collection=collection, image_id=image_id, data=bytes(data), ext=str(ext)
            )


def split_document(
    settings: Settings, document: Document, trace: Optional[Any] = None
) -> List[Chunk]:
    """
    Convenience function to split a document using the ingestion pipeline.

    Args:
        settings: Global application settings.
        document: Input document to split.
        trace: Optional trace context for observability.

    Returns:
        List of generated chunks.
    """
    return IngestionPipeline(settings).split(document, trace=trace).chunks
