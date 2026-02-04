from dataclasses import dataclass
from typing import List, Optional, Any

from src.core.settings import Settings
from src.ingestion.models import Document, Chunk
from src.libs.splitter.splitter_factory import SplitterFactory


@dataclass(frozen=True)
class SplitResult:
    document: Document
    chunks: List[Chunk]


class IngestionPipeline:
    """
    Minimal ingestion pipeline for MVP.

    This pipeline currently integrates the configured splitter and produces Chunk objects
    from a Document.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the pipeline.

        Args:
            settings: Global application settings.
        """
        self._settings = settings

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

