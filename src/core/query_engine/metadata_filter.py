from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TypeVar

from src.core.query_engine.fusion import FusionHit
from src.libs.vector_store.base_vector_store import VectorRecord

T = TypeVar("T")


class MetadataFilter:
    def apply(
        self, candidates: Sequence[T], filters: Optional[Dict[str, Any]]
    ) -> List[T]:
        if not candidates:
            return []
        if not filters:
            return list(candidates)

        out: List[T] = []
        for c in candidates:
            metadata = _extract_metadata(c)
            if _match_filters(metadata, filters):
                out.append(c)
        return out


def _extract_metadata(candidate: Any) -> Dict[str, Any]:
    if isinstance(candidate, VectorRecord):
        return dict(candidate.metadata or {})

    if isinstance(candidate, FusionHit):
        if candidate.record is None:
            return {}
        return dict(candidate.record.metadata or {})

    meta = getattr(candidate, "metadata", None)
    if isinstance(meta, dict):
        return dict(meta)

    record = getattr(candidate, "record", None)
    if isinstance(record, VectorRecord):
        return dict(record.metadata or {})

    record_meta = getattr(record, "metadata", None)
    if isinstance(record_meta, dict):
        return dict(record_meta)

    return {}


def _match_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if expected is None:
            continue

        if key not in metadata:
            continue

        actual = metadata.get(key)
        if not _match_value(actual, expected):
            return False

    return True


def _match_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (list, tuple, set, frozenset)):
        expected_set = _normalize_expected_set(expected)
        return _match_membership(actual, expected_set)

    if isinstance(actual, list):
        return expected in actual

    return actual == expected


def _normalize_expected_set(values: Iterable[Any]) -> Set[Any]:
    out: Set[Any] = set()
    for v in values:
        if isinstance(v, (list, tuple, set, frozenset)):
            out.update(v)
        else:
            out.add(v)
    return out


def _match_membership(actual: Any, expected: Set[Any]) -> bool:
    if isinstance(actual, list):
        return any(v in expected for v in actual)
    return actual in expected
