from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from src.core.query_engine.dense_retriever import DenseHit
from src.core.query_engine.fusion import RRFFusion
from src.core.query_engine.hybrid_search import HybridSearch
from src.core.query_engine.sparse_retriever import SparseHit
from src.core.settings import (
    EmbeddingSettings,
    EvaluationSettings,
    IngestionSettings,
    LLMSettings,
    ObservabilitySettings,
    RerankSettings,
    RetrievalSettings,
    Settings,
    SplitterSettings,
    TransformSettings,
    VectorStoreSettings,
    VisionLLMSettings,
)
from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord


class MockVectorStore(BaseVectorStore):
    def __init__(self, records: List[VectorRecord]) -> None:
        self.store: Dict[str, VectorRecord] = {r.id: r for r in records}

    def upsert(self, records: List[VectorRecord], trace: Optional[Any] = None) -> None:
        for r in records:
            self.store[r.id] = r

    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[VectorRecord]:
        return list(self.store.values())[:top_k]


@dataclass
class FakeDenseRetriever:
    _vector_store: MockVectorStore

    def retrieve(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[DenseHit]:
        ids = ["a", "b"]
        return [
            DenseHit(record=self._vector_store.store[i], score=1.0)
            for i in ids
            if i in self._vector_store.store
        ]


@dataclass
class FakeSparseRetriever:
    def retrieve(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None,
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[SparseHit]:
        return [SparseHit(chunk_id="d", score=10.0), SparseHit(chunk_id="a", score=9.0)]


def _settings() -> Settings:
    return Settings(
        llm=LLMSettings(provider="ollama", model="x", api_key=None, base_url=None),
        embedding=EmbeddingSettings(provider="local", model="fake"),
        vision_llm=VisionLLMSettings(provider="ollama", model="x"),
        vector_store=VectorStoreSettings(
            backend="jsonl", persist_path="data/db/vector"
        ),
        ingestion=IngestionSettings(
            splitter=SplitterSettings(provider="recursive"),
            transform=TransformSettings(),
        ),
        retrieval=RetrievalSettings(
            sparse_backend="bm25",
            fusion_algorithm="rrf",
            top_k_dense=2,
            top_k_sparse=2,
            top_k_final=3,
        ),
        rerank=RerankSettings(backend="none", model="x", top_m=5),
        evaluation=EvaluationSettings(backends=["custom"], golden_test_set=""),
        observability=ObservabilitySettings(
            enabled=False, log_file="", dashboard_port=0
        ),
    )


@pytest.mark.integration
def test_hybrid_search_returns_topk_with_text_and_metadata() -> None:
    store = MockVectorStore(
        records=[
            VectorRecord(id="a", embedding=[0.0], content="A", metadata={"lang": "zh"}),
            VectorRecord(id="b", embedding=[0.0], content="B", metadata={"lang": "en"}),
            VectorRecord(id="d", embedding=[0.0], content="D", metadata={"lang": "zh"}),
        ]
    )
    hs = HybridSearch(
        _settings(),
        dense_retriever=FakeDenseRetriever(store),
        sparse_retriever=FakeSparseRetriever(),
        fusion=RRFFusion(k=60),
    )

    out = hs.search("collection:demo 介绍 BM25 和 RRF", top_k_final=3)
    assert len(out) > 0
    assert all(h.record.content for h in out)
    assert all(isinstance(h.record.metadata, dict) for h in out)
