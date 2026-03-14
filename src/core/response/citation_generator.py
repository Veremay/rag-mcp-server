"""
引用生成器：从检索命中的 metadata 抽取出前端所需的引用信息。

命中顺序与返回的 citations 一一对应，便于 ResponseBuilder 拼 Markdown 时用 [idx]
与 citation.id 对齐。支持多种 metadata 键名（source/source_file/source_path、page/page_number）
以兼容不同摄取管道的写入习惯。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.core.query_engine.hybrid_search import HybridSearchHit

JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class Citation:
    """
    单条引用：id 与命中序号一致，source/page 用于展示与跳转，chunk_id/score 便于调试或再检索。
    """
    id: int
    source: str
    chunk_id: str
    score: float
    page: Optional[int] = None

    def to_dict(self) -> JsonDict:
        out: JsonDict = {
            "id": int(self.id),
            "source": str(self.source),
            "chunk_id": str(self.chunk_id),
            "score": float(self.score),
        }
        if self.page is not None:
            out["page"] = int(self.page)
        return out


class CitationGenerator:
    """
    按命中顺序生成引用列表，每条命中对应一条 Citation。
    用 _pick_first_str 与 _parse_int 兼容多种 metadata 形态，避免因字段名或类型不一致导致缺显。
    """

    def generate(self, hits: Sequence[HybridSearchHit]) -> List[JsonDict]:
        """
        遍历 hits 从 record.metadata 抽取 source、page、chunk_id、score 并转为 dict 列表。
        与 hits 同序保证 ResponseBuilder 里 citations[idx-1] 与 hits[idx-1] 对应。
        """
        citations: List[JsonDict] = []
        for idx, h in enumerate(hits, start=1):
            meta = dict(getattr(getattr(h, "record", None), "metadata", None) or {})
            source = (
                _pick_first_str(meta, ["source", "source_file", "source_path"]) or ""
            )
            page = _parse_int(meta.get("page") or meta.get("page_number"))
            citations.append(
                Citation(
                    id=idx,
                    source=source,
                    page=page,
                    chunk_id=str(h.chunk_id),
                    score=float(h.score),
                ).to_dict()
            )
        return citations


def _pick_first_str(meta: Dict[str, Any], keys: List[str]) -> Optional[str]:
    """
    在 meta 中按 keys 顺序取第一个非空字符串。兼容 source / source_file / source_path
    等不同摄取阶段的命名，避免只认一个键导致引用为空。
    """
    for k in keys:
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _parse_int(value: Any) -> Optional[int]:
    """
    将 metadata 中的 page 安全转为 int。拒绝 bool、非整数 float 和非法字符串，
    避免前端拿到 NaN 或异常类型导致展示或跳转失败。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            return None
    return None
