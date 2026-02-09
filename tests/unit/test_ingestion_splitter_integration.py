from typing import Any, List, Optional
from unittest.mock import MagicMock

from src.core.settings import IngestionSettings, Settings, SplitterSettings
from src.ingestion.models import Chunk, Document
from src.ingestion.pipeline import split_document
from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import SplitterFactory


def _make_settings(provider: str, chunk_size: int, chunk_overlap: int) -> Settings:
    settings = MagicMock(spec=Settings)
    settings.ingestion = MagicMock(spec=IngestionSettings)
    settings.ingestion.splitter = MagicMock(spec=SplitterSettings)
    settings.ingestion.splitter.provider = provider
    settings.ingestion.splitter.chunk_size = chunk_size
    settings.ingestion.splitter.chunk_overlap = chunk_overlap
    return settings


class FakeSplitter(BaseSplitter):
    def __init__(self, settings: Settings):
        self._chunk_size = int(settings.ingestion.splitter.chunk_size)

    def split_text(
        self, text: str, trace: Optional[Any] = None, **kwargs: Any
    ) -> List[str]:
        if not text:
            return []
        return [
            text[i : i + self._chunk_size]
            for i in range(0, len(text), self._chunk_size)
        ]


def test_splitter_config_affects_chunk_output_lengths():
    text = ("Hello world. " * 100).strip()
    document = Document(text=text, metadata={"source_path": "fixtures/sample.pdf"})

    SplitterFactory.register("fake_c4", FakeSplitter)
    small = _make_settings(provider="fake_c4", chunk_size=80, chunk_overlap=0)
    large = _make_settings(provider="fake_c4", chunk_size=200, chunk_overlap=0)

    small_chunks = split_document(small, document)
    large_chunks = split_document(large, document)

    assert len(small_chunks) > len(large_chunks)
    assert all(isinstance(c, Chunk) for c in small_chunks)
    assert all(len(c.text) <= 80 for c in small_chunks)
    assert all(len(c.text) <= 200 for c in large_chunks)
    assert all(
        c.metadata.get("source_path") == "fixtures/sample.pdf" for c in small_chunks
    )
    assert all(c.doc_id == document.id for c in small_chunks)
