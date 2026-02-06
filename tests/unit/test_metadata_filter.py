from src.core.query_engine.fusion import FusionHit
from src.core.query_engine.metadata_filter import MetadataFilter
from src.libs.vector_store.base_vector_store import VectorRecord


def test_metadata_filter_allows_missing_fields() -> None:
    f = MetadataFilter()
    candidates = [
        VectorRecord(id="a", embedding=[0.0], content="a", metadata={"language": "zh"}),
        VectorRecord(id="b", embedding=[0.0], content="b", metadata={}),
    ]
    out = f.apply(candidates, {"language": "zh"})
    assert [c.id for c in out] == ["a", "b"]


def test_metadata_filter_filters_mismatched_values() -> None:
    f = MetadataFilter()
    candidates = [
        VectorRecord(
            id="a", embedding=[0.0], content="a", metadata={"doc_type": "pdf"}
        ),
        VectorRecord(id="b", embedding=[0.0], content="b", metadata={"doc_type": "md"}),
    ]
    out = f.apply(candidates, {"doc_type": "pdf"})
    assert [c.id for c in out] == ["a"]


def test_metadata_filter_supports_expected_list_membership() -> None:
    f = MetadataFilter()
    candidates = [
        VectorRecord(id="a", embedding=[0.0], content="a", metadata={"language": "zh"}),
        VectorRecord(id="b", embedding=[0.0], content="b", metadata={"language": "en"}),
    ]
    out = f.apply(candidates, {"language": ["zh", "ja"]})
    assert [c.id for c in out] == ["a"]


def test_metadata_filter_supports_actual_list_membership() -> None:
    f = MetadataFilter()
    candidates = [
        VectorRecord(
            id="a", embedding=[0.0], content="a", metadata={"tags": ["a", "b"]}
        ),
        VectorRecord(id="b", embedding=[0.0], content="b", metadata={"tags": ["c"]}),
    ]
    out = f.apply(candidates, {"tags": "b"})
    assert [c.id for c in out] == ["a"]


def test_metadata_filter_uses_fusion_hit_record_metadata_when_available() -> None:
    f = MetadataFilter()
    hit_a = FusionHit(
        chunk_id="a",
        score=1.0,
        record=VectorRecord(
            id="a", embedding=[0.0], content="a", metadata={"access_level": "public"}
        ),
        dense_rank=1,
        sparse_rank=None,
    )
    hit_b = FusionHit(
        chunk_id="b",
        score=1.0,
        record=None,
        dense_rank=None,
        sparse_rank=1,
    )

    out = f.apply([hit_a, hit_b], {"access_level": "public"})
    assert [h.chunk_id for h in out] == ["a", "b"]
