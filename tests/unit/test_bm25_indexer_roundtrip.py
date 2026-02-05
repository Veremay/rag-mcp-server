from pathlib import Path

from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.models import Chunk
from src.ingestion.storage.bm25_indexer import BM25Indexer


def test_bm25_index_roundtrip_and_stable_ranking(tmp_path: Path) -> None:
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

    hits_apple = indexer.search(collection="test", query="apple", top_k=3)
    assert [h.chunk_id for h in hits_apple] == ["c2", "c1"]

    hits_banana = indexer.search(collection="test", query="banana", top_k=3)
    assert [h.chunk_id for h in hits_banana] == ["c1", "c3"]

    indexer2 = BM25Indexer(base_dir=tmp_path)
    hits_apple_2 = indexer2.search(collection="test", query="apple", top_k=3)
    assert [h.chunk_id for h in hits_apple_2] == [h.chunk_id for h in hits_apple]


def test_build_requires_matching_lengths(tmp_path: Path) -> None:
    indexer = BM25Indexer(base_dir=tmp_path)
    indexer.build(collection="c", chunk_ids=[], sparse_vectors=[])

    try:
        indexer.build(collection="c", chunk_ids=["a"], sparse_vectors=[])
    except ValueError as e:
        assert "length mismatch" in str(e)
    else:
        raise AssertionError("Expected ValueError")
