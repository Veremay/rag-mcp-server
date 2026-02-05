from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.settings import Settings
from src.libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord


class JsonlStore(BaseVectorStore):
    def __init__(self, settings: Settings) -> None:
        self._base_dir = Path(settings.vector_store.persist_path)
        self._collection_name = str(settings.vector_store.collection_name)

    def upsert(self, records: List[VectorRecord], trace: Optional[Any] = None) -> None:
        path = self._path()
        if not records:
            if not path.exists():
                path.write_text("", encoding="utf-8")
            return

        existing = self._load_all()
        for r in records:
            existing[r.id] = r
        self._persist_all(existing)

    def query(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        trace: Optional[Any] = None,
    ) -> List[VectorRecord]:
        records = list(self._load_all().values())
        if filters:
            records = [r for r in records if _metadata_match(r.metadata, filters)]
        if not records or top_k <= 0:
            return []

        scored: List[Tuple[float, VectorRecord]] = []
        for r in records:
            score = _cosine_similarity(vector, r.embedding)
            scored.append((score, r))
        scored.sort(key=lambda x: (-x[0], x[1].id))
        return [r for _, r in scored[:top_k]]

    def _path(self) -> Path:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        safe = self._collection_name.replace(os.sep, "_")
        return self._base_dir / f"{safe}.jsonl"

    def _load_all(self) -> Dict[str, VectorRecord]:
        path = self._path()
        if not path.exists():
            return {}

        out: Dict[str, VectorRecord] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue

            rid = payload.get("id")
            embedding = payload.get("embedding")
            content = payload.get("content")
            metadata = payload.get("metadata")
            if (
                not isinstance(rid, str)
                or not isinstance(embedding, list)
                or not isinstance(content, str)
            ):
                continue
            if not isinstance(metadata, dict):
                metadata = {}
            out[rid] = VectorRecord(
                id=rid,
                embedding=[float(x) for x in embedding],
                content=content,
                metadata=metadata,
            )
        return out

    def _persist_all(self, records: Dict[str, VectorRecord]) -> None:
        path = self._path()
        lines: List[str] = []
        for rid in sorted(records.keys()):
            r = records[rid]
            lines.append(
                json.dumps(
                    {
                        "id": r.id,
                        "embedding": list(r.embedding),
                        "content": r.content,
                        "metadata": dict(r.metadata),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        payload = "\n".join(lines) + ("\n" if lines else "")

        fd = None
        tmp_path: Optional[str] = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=path.name + ".",
                suffix=".tmp",
                dir=str(path.parent),
                text=True,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp_path, path)
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def _metadata_match(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    for k, v in filters.items():
        if metadata.get(k) != v:
            return False
    return True


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        x = float(a[i])
        y = float(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)
