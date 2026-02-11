import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

from src.core.settings import EmbeddingSettings, IngestionSettings, Settings, SplitterSettings, VectorStoreSettings
from src.ingestion.models import Document
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.file_integrity import FileIntegrityRegistry
from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import SplitterFactory
from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord


class FakeLoader(BaseLoader):
    def __init__(self, text: str):
        self._text = text

    def load(self, file_path: str | Path) -> Document:
        path = Path(file_path)
        return Document(
            text=self._text,
            metadata={"source_path": str(path.absolute()), "section_path": "s0"},
        )


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


class FakeDenseEncoder:
    def encode(self, chunks, trace: Optional[Any] = None):
        return [[0.0, 0.0, 0.0] for _ in chunks]


class MockVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self.store: dict[str, VectorRecord] = {}

    def upsert(self, records, trace: Optional[Any] = None) -> None:
        for r in records:
            self.store[r.id] = r

    def query(self, vector, top_k, filters=None, trace: Optional[Any] = None):
        return list(self.store.values())[:top_k]


def _make_settings(provider: str, chunk_size: int) -> Settings:
    settings = MagicMock(spec=Settings)
    settings.ingestion = MagicMock(spec=IngestionSettings)
    settings.ingestion.splitter = MagicMock(spec=SplitterSettings)
    settings.ingestion.splitter.provider = provider
    settings.ingestion.splitter.chunk_size = chunk_size
    settings.ingestion.splitter.chunk_overlap = 0

    settings.ingestion.transform = MagicMock()
    settings.ingestion.transform.chunk_refiner = MagicMock()
    settings.ingestion.transform.metadata_enricher = MagicMock()
    settings.ingestion.transform.image_captioner = MagicMock()

    settings.embedding = MagicMock(spec=EmbeddingSettings)
    settings.vector_store = MagicMock(spec=VectorStoreSettings)
    return settings


from src.core.trace.trace_context import TraceContext

@pytest.mark.integration
def test_ingestion_pipeline_generates_trace_with_all_stages(tmp_path: Path) -> None:
    """Validate F4: Ingestion pipeline records all required stages with details."""
    SplitterFactory.register("fake_c14", FakeSplitter)
    settings = _make_settings(provider="fake_c14", chunk_size=20)
    settings.embedding.model = "fake-embed"
    settings.vector_store.backend = "mock-store"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(b"%PDF-1.4 dummy")
        file_path = Path(f.name)
    
    try:
        integrity = FileIntegrityRegistry(storage_path=tmp_path / "history.json")
        bm25 = BM25Indexer(base_dir=tmp_path / "bm25")
        images = ImageStorage(base_dir=tmp_path / "images")
        store = MockVectorStore()

        pipeline = IngestionPipeline(
            settings,
            integrity=integrity,
            loader=FakeLoader(text=("Hello world. " * 10).strip()),
            transforms=[],
            dense_encoder=FakeDenseEncoder(),
            vector_store=store,
            bm25_indexer=bm25,
            image_storage=images,
        )

        trace = TraceContext()
        pipeline.ingest(collection="c14", file_path=file_path, trace=trace)

        # 1. Verify trace type
        assert trace.trace_type == "ingestion"

        # 2. Verify all stages are present
        stage_names = [s.name for s in trace.stages]
        required_stages = ["load", "split", "transform", "encode", "upsert"]
        for req in required_stages:
            assert req in stage_names, f"Missing stage: {req}"

        # 3. Verify stage details (method, elapsed_ms)
        for s in trace.stages:
            assert s.duration_ms is not None
            assert s.duration_ms >= 0.0
            
            if s.name == "load":
                assert s.data["method"] == "FakeLoader"
            elif s.name == "split":
                assert s.data["method"] == "fake_c14"
            elif s.name == "transform":
                assert s.data["method"] == "chain"
            elif s.name == "encode":
                assert s.data["dense_model"] == "fake-embed"
            elif s.name == "upsert":
                assert s.data["method"] == "mock-store"

    finally:
        if file_path.exists():
            os.unlink(file_path)
