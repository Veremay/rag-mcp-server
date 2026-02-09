from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class CollectionInfo:
    name: str
    file_count: int

    def to_dict(self) -> JsonDict:
        return {"name": self.name, "file_count": int(self.file_count)}


def list_collections(*, trace: Optional[Any] = None) -> JsonDict:
    _ = trace
    collections = _scan_documents_dir(Path("data/documents"))
    markdown = _to_markdown(collections)
    return {
        "content": [{"type": "text", "text": markdown}],
        "structuredContent": {"collections": [c.to_dict() for c in collections]},
    }


def _scan_documents_dir(base_dir: Path) -> List[CollectionInfo]:
    if not base_dir.exists() or not base_dir.is_dir():
        return []

    out: List[CollectionInfo] = []
    for child in sorted(base_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        file_count = sum(1 for p in child.rglob("*") if p.is_file())
        out.append(CollectionInfo(name=child.name, file_count=file_count))
    return out


def _to_markdown(collections: Sequence[CollectionInfo]) -> str:
    if not collections:
        msg = "未发现任何集合（data/documents/ 为空或不存在）"
        return msg + "\n"

    lines: List[str] = ["可用集合：", ""]
    for c in collections:
        lines.append(f"- {c.name}（files={c.file_count}）")
    return "\n".join(lines).strip() + "\n"
