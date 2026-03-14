"""
元数据过滤：根据 query 解析出的 filters 对候选列表做内存侧过滤。

向量库/稀疏检索可能已带过滤，此处对融合后的候选再做一次过滤，可统一处理
来自不同来源的候选（如 FusionHit、VectorRecord），并支持「期望值为列表」的
多值匹配（如 language in [en, zh]），保证与 QueryProcessor 解析出的 filters 语义一致。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TypeVar

from src.core.query_engine.fusion import FusionHit
from src.libs.vector_store.base_vector_store import VectorRecord

T = TypeVar("T")


class MetadataFilter:
    """
    对候选列表按 metadata 应用过滤条件，只保留满足 filters 的项。

    无 filters 时原样返回，避免多余拷贝；有 filters 时逐条 _extract_metadata 并
    _match_filters，支持 VectorRecord、FusionHit 及带 record/metadata 的 duck typing。
    """

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
    """
    从候选对象中安全取出 metadata 字典。兼容 VectorRecord、FusionHit 以及
    仅带 record/metadata 属性的对象，避免因结构不同导致过滤漏掉或报错。
    """
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
    """
    判断一条 metadata 是否满足 filters。仅对 filters 中出现的 key 做校验，
    expected 为 None 的 key 跳过，这样调用方可以只传需要过滤的键。
    """
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
    """
    单值匹配：expected 为列表/集合时表示「actual 属于 expected」；
    否则要求 actual == expected，以支持 doc_type=report 这类单值条件。
    """
    if isinstance(expected, (list, tuple, set, frozenset)):
        expected_set = _normalize_expected_set(expected)
        return _match_membership(actual, expected_set)

    if isinstance(actual, list):
        return expected in actual

    return actual == expected


def _normalize_expected_set(values: Iterable[Any]) -> Set[Any]:
    """
    将「期望值」规范为集合：若元素本身是列表/元组等则展平，便于统一做成员判断。
    """
    out: Set[Any] = set()
    for v in values:
        if isinstance(v, (list, tuple, set, frozenset)):
            out.update(v)
        else:
            out.add(v)
    return out


def _match_membership(actual: Any, expected: Set[Any]) -> bool:
    """
    判断 actual 是否在 expected 中。actual 为列表时表示「多值字段至少有一个在 expected 内」。
    """
    if isinstance(actual, list):
        return any(v in expected for v in actual)
    return actual in expected
