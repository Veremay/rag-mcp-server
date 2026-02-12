from typing import List
from unittest.mock import MagicMock

import pytest

from src.ingestion.models import Chunk
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord


class InMemoryVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self.store: dict[str, VectorRecord] = {}

    def upsert(self, records: List[VectorRecord], trace=None) -> None:
        for record in records:
            self.store[record.id] = record

    def query(self, vector, top_k, filters=None, trace=None):
        raise NotImplementedError

    def delete_by_metadata(self, filters):
        pass

    def get_records_by_metadata(self, filters):
        return []


def test_same_chunk_twice_produces_same_id() -> None:
    vector_store = InMemoryVectorStore()
    upserter = VectorUpserter(settings=MagicMock(), vector_store=vector_store)

    chunk_a = Chunk(
        text="hello world",
        metadata={"source_path": "a.pdf", "section_path": "p1/s1"},
        doc_id="d1",
    )
    chunk_b = Chunk(
        text="hello world",
        metadata={"source_path": "a.pdf", "section_path": "p1/s1"},
        doc_id="d1",
    )

    r1 = upserter.upsert([chunk_a], [[0.1, 0.2]]).records[0]
    r2 = upserter.upsert([chunk_b], [[0.1, 0.2]]).records[0]

    assert r1.id == r2.id
    assert len(vector_store.store) == 1


def test_content_change_produces_new_id() -> None:
    vector_store = InMemoryVectorStore()
    upserter = VectorUpserter(settings=MagicMock(), vector_store=vector_store)

    chunk_v1 = Chunk(
        text="v1",
        metadata={"source_path": "a.pdf", "section_path": "p1/s1"},
        doc_id="d1",
    )
    chunk_v2 = Chunk(
        text="v2",
        metadata={"source_path": "a.pdf", "section_path": "p1/s1"},
        doc_id="d1",
    )

    r1 = upserter.upsert([chunk_v1], [[0.1, 0.2]]).records[0]
    r2 = upserter.upsert([chunk_v2], [[0.1, 0.2]]).records[0]

    assert r1.id != r2.id
    assert len(vector_store.store) == 2


def test_mismatched_lengths_raise() -> None:
    upserter = VectorUpserter(settings=MagicMock(), vector_store=InMemoryVectorStore())
    chunk = Chunk(text="x", metadata={"source_path": "a", "section_path": "b"})

    with pytest.raises(ValueError, match="length mismatch"):
        upserter.upsert([chunk], [])
