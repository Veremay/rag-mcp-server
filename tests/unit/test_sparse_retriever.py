from pathlib import Path

from src.core.query_engine.sparse_retriever import SparseRetriever
from src.core.settings import (
    EmbeddingSettings,
    EvaluationSettings,
    IngestionSettings,
    LLMSettings,
    ObservabilitySettings,
    RetrievalSettings,
    RerankSettings,
    Settings,
    SplitterSettings,
    TransformSettings,
    VectorStoreSettings,
    VisionLLMSettings,
)
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.models import Chunk
from src.ingestion.storage.bm25_indexer import BM25Indexer


def _settings() -> Settings:
    return Settings(
        llm=LLMSettings(provider="ollama", model="x", api_key=None, base_url=None),
        embedding=EmbeddingSettings(provider="local", model="fake"),
        vision_llm=VisionLLMSettings(provider="ollama", model="x"),
        vector_store=VectorStoreSettings(
            backend="jsonl", persist_path="data/db/vector", collection_name="test"
        ),
        ingestion=IngestionSettings(
            splitter=SplitterSettings(provider="recursive"),
            transform=TransformSettings(),
        ),
        retrieval=RetrievalSettings(
            sparse_backend="bm25",
            fusion_algorithm="rrf",
            top_k_dense=2,
            top_k_sparse=3,
            top_k_final=3,
        ),
        rerank=RerankSettings(backend="none", model="x", top_m=5),
        evaluation=EvaluationSettings(backends=["custom"], golden_test_set=""),
        observability=ObservabilitySettings(
            enabled=False, log_file="", dashboard_port=0
        ),
    )


def test_sparse_retriever_hits_expected_chunk_ids(tmp_path: Path) -> None:
    chunks = [
        Chunk(text="apple banana", metadata={"source_path": "a", "section_path": "s1"}),
        Chunk(
            text="apple apple carrot",
            metadata={"source_path": "a", "section_path": "s2"},
        ),
        Chunk(
            text="banana carrot carrot",
            metadata={"source_path": "a", "section_path": "s3"},
        ),
    ]
    chunk_ids = ["c1", "c2", "c3"]
    sparse_vectors = SparseEncoder().encode(chunks)

    indexer = BM25Indexer(base_dir=tmp_path)
    indexer.build(collection="test", chunk_ids=chunk_ids, sparse_vectors=sparse_vectors)

    retriever = SparseRetriever(_settings(), indexer=indexer)
    hits = retriever.retrieve("apple")
    assert [h.chunk_id for h in hits] == ["c2", "c1"]


def test_sparse_retriever_uses_collection_from_filters(tmp_path: Path) -> None:
    chunks = [
        Chunk(text="alpha beta", metadata={"source_path": "a", "section_path": "s1"}),
        Chunk(
            text="beta beta gamma", metadata={"source_path": "a", "section_path": "s2"}
        ),
    ]
    chunk_ids = ["x1", "x2"]
    sparse_vectors = SparseEncoder().encode(chunks)

    indexer = BM25Indexer(base_dir=tmp_path)
    indexer.build(collection="demo", chunk_ids=chunk_ids, sparse_vectors=sparse_vectors)

    retriever = SparseRetriever(_settings(), indexer=indexer)
    hits = retriever.retrieve("gamma", filters={"collection": "demo"}, top_k=2)
    assert [h.chunk_id for h in hits] == ["x2"]
