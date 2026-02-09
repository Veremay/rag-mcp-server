from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.core.query_engine.dense_retriever import DenseRetriever
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
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord


class FakeEmbedding(BaseEmbedding):
    def embed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        return [[1.0, 0.0] for _ in texts]

    async def aembed(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        return self.embed(texts, **kwargs)


class FakeVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self.last_filters: Optional[Dict[str, Any]] = None

    def upsert(self, records: List[VectorRecord], trace: Optional[Any] = None) -> None:
        return None

    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[VectorRecord]:
        self.last_filters = filters
        records = [
            VectorRecord(id="a", embedding=[1.0, 0.0], content="A", metadata={}),
            VectorRecord(id="b", embedding=[0.0, 1.0], content="B", metadata={}),
            VectorRecord(id="c", embedding=[-1.0, 0.0], content="C", metadata={}),
        ]
        return records[:top_k]


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


def test_dense_retriever_returns_hits_with_normalized_score():
    store = FakeVectorStore()
    retriever = DenseRetriever(
        _settings(), embedding=FakeEmbedding(), vector_store=store
    )

    hits = retriever.retrieve("hello", top_k=3)
    assert len(hits) == 3

    assert hits[0].record.id == "a"
    assert hits[0].score == 1.0
    assert hits[1].record.id == "b"
    assert hits[1].score == 0.5
    assert hits[2].record.id == "c"
    assert hits[2].score == 0.0


def test_dense_retriever_passes_filters_to_vector_store():
    store = FakeVectorStore()
    retriever = DenseRetriever(
        _settings(), embedding=FakeEmbedding(), vector_store=store
    )

    hits = retriever.retrieve("hello", filters={"collection": "demo"}, top_k=1)
    assert len(hits) == 1
    assert store.last_filters == {"collection": "demo"}
