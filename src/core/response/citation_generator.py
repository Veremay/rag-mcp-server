from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.core.query_engine.hybrid_search import HybridSearchHit

JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class Citation:
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
    def generate(self, hits: Sequence[HybridSearchHit]) -> List[JsonDict]:
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
    for k in keys:
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _parse_int(value: Any) -> Optional[int]:
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
