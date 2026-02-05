from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

from src.core.settings import Settings
from src.ingestion.models import Document, Chunk
from src.ingestion.embedding.batch_processor import BatchProcessResult, BatchProcessor
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.storage.bm25_indexer import BM25Indexer, BM25Index
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
        force: bool = False,
        trace: Optional[Any] = None,
    ) -> IngestResult:
        path = Path(file_path)

        try:
            file_hash = self._integrity.compute_sha256(path)
        except Exception as e:
            raise RuntimeError("IngestionPipeline integrity step failed") from e

        if not force and self._integrity.should_skip(file_hash):
            return IngestResult(
                skipped=True,
                file_hash=file_hash,
                document=None,
                chunks=[],
                batch=None,
                upsert=None,
                bm25=None,
            )

        try:
            loader = self._resolve_loader(path)
            document = loader.load(path)
        except Exception as e:
            raise RuntimeError("IngestionPipeline loader step failed") from e

        try:
            split = self.split(document, trace=trace)
            chunks = split.chunks
        except Exception as e:
            raise RuntimeError("IngestionPipeline splitter step failed") from e

        try:
            chunks = self._apply_transforms(chunks, trace=trace)
        except Exception as e:
            raise RuntimeError("IngestionPipeline transform step failed") from e

        try:
            batch = self._encode(chunks, trace=trace)
        except Exception as e:
            raise RuntimeError("IngestionPipeline embedding step failed") from e

        try:
            upsert = self._upsert(chunks, batch.dense_vectors, trace=trace)
        except Exception as e:
            raise RuntimeError("IngestionPipeline vector upsert step failed") from e

        try:
            chunk_ids = [r.id for r in upsert.records]
            bm25 = self._build_bm25(
                collection=collection, chunk_ids=chunk_ids, sparse_vectors=batch.sparse_vectors
            )
        except Exception as e:
            raise RuntimeError("IngestionPipeline bm25 step failed") from e

        try:
            self._store_images(collection=collection, document=document)
        except Exception as e:
            raise RuntimeError("IngestionPipeline image storage step failed") from e

        try:
            self._integrity.mark_success(file_hash)
        except Exception as e:
            raise RuntimeError("IngestionPipeline finalize step failed") from e

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
            return PdfLoader()

        raise ValueError(f"Unsupported file extension: {ext}")

    def _apply_transforms(self, chunks: List[Chunk], *, trace: Optional[Any]) -> List[Chunk]:
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

    def _encode(self, chunks: List[Chunk], *, trace: Optional[Any]) -> BatchProcessResult:
        dense_encoder = self._dense_encoder or DenseEncoder(self._settings)
        sparse_encoder = self._sparse_encoder or SparseEncoder()
        batcher = self._batch_processor or BatchProcessor(batch_size=16)
        return batcher.process(chunks, dense_encoder=dense_encoder, sparse_encoder=sparse_encoder, trace=trace)

    def _upsert(
        self, chunks: Sequence[Chunk], dense_vectors: Sequence[Sequence[float]], *, trace: Optional[Any]
    ) -> UpsertResult:
        upserter = VectorUpserter(self._settings, vector_store=self._vector_store)
        return upserter.upsert(chunks, dense_vectors, trace=trace)

    def _build_bm25(
        self, *, collection: str, chunk_ids: Sequence[str], sparse_vectors: Sequence[dict[str, float]]
    ) -> BM25Index:
        indexer = self._bm25_indexer or BM25Indexer()
        return indexer.build(collection=collection, chunk_ids=chunk_ids, sparse_vectors=sparse_vectors)

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
            if not isinstance(image_id, str) or not isinstance(data, (bytes, bytearray)):
                continue
            storage.save(collection=collection, image_id=image_id, data=bytes(data), ext=str(ext))


def split_document(settings: Settings, document: Document, trace: Optional[Any] = None) -> List[Chunk]:
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
