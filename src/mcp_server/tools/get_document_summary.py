from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.core.settings import Settings, load_settings
from src.libs.vector_store.chroma_store import ChromaStore

JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class GetDocumentSummaryParams:
    doc_id: str


def get_document_summary(
    params: GetDocumentSummaryParams, *, trace: Optional[Any] = None
) -> JsonDict:
    _ = trace
    doc_id = (params.doc_id or "").strip()
    if not doc_id:
        raise ValueError("doc_id must be a non-empty string")

    settings = _get_settings()
    metadata = _load_one_metadata_by_doc_id(settings, doc_id=doc_id)
    title, summary, tags = _extract_summary_fields(metadata)

    markdown = _to_markdown(doc_id=doc_id, title=title, summary=summary, tags=tags)
    return {
        "content": [{"type": "text", "text": markdown}],
        "structuredContent": {"title": title, "summary": summary, "tags": tags},
    }


@lru_cache(maxsize=1)
def _get_settings() -> Settings:
    return load_settings()


def _load_one_metadata_by_doc_id(settings: Settings, *, doc_id: str) -> Dict[str, Any]:
    backend = str(
        getattr(getattr(settings, "vector_store", None), "backend", "")
    ).lower()
    persist_path = str(
        getattr(getattr(settings, "vector_store", None), "persist_path", "")
    )
    collection_name = str(
        getattr(getattr(settings, "vector_store", None), "collection_name", "")
    )

    if backend == "jsonl":
        base_dir = Path(persist_path or ".")
        jsonl_path = base_dir / f"{collection_name}.jsonl"
        metadata = _find_metadata_in_jsonl(jsonl_path, doc_id=doc_id)
        if metadata is None:
            raise ValueError("doc_id not found")
        return metadata

    if backend == "chroma":
        store = ChromaStore(settings)
        try:
            raw = store.collection.get(
                where={"doc_id": doc_id}, include=["metadatas"], limit=1
            )
        except TypeError:
            raw = store.collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        metadatas = raw.get("metadatas") if isinstance(raw, dict) else None
        if isinstance(metadatas, list) and metadatas and isinstance(metadatas[0], dict):
            return dict(metadatas[0])
        raise ValueError("doc_id not found")

    raise ValueError(f"Unsupported vector store backend: {backend}")


def _find_metadata_in_jsonl(path: Path, *, doc_id: str) -> Optional[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return None

    for payload in _iter_jsonl(path):
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if str(metadata.get("doc_id") or "") == doc_id:
            return dict(metadata)
    return None


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


def _extract_summary_fields(metadata: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    title = metadata.get("title")
    summary = metadata.get("summary")
    tags = metadata.get("tags")

    if not isinstance(title, str) or not title.strip():
        title = str(metadata.get("filename") or "Untitled")
    if not isinstance(summary, str) or not summary.strip():
        summary = title

    if isinstance(tags, list):
        normalized: List[str] = []
        for t in tags:
            if isinstance(t, str) and t.strip():
                normalized.append(t.strip())
        tags = normalized
    else:
        tags = []

    return title.strip(), summary.strip(), tags


def _to_markdown(*, doc_id: str, title: str, summary: str, tags: Sequence[str]) -> str:
    lines: List[str] = [f"doc_id: {doc_id}", ""]
    lines.append(f"标题：{title}")
    lines.append("")
    lines.append("摘要：")
    lines.append(summary)
    lines.append("")
    if tags:
        lines.append("标签：")
        lines.append(", ".join(tags))
        lines.append("")
    return "\n".join(lines).strip() + "\n"
