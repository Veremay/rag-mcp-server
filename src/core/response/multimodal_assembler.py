from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.core.query_engine.hybrid_search import HybridSearchHit
from src.ingestion.storage.image_storage import ImageStorage

JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class ImageContent:
    data: str
    mimeType: str

    def to_dict(self) -> JsonDict:
        return {"type": "image", "data": str(self.data), "mimeType": str(self.mimeType)}


class MultimodalAssembler:
    def __init__(
        self,
        *,
        image_storage: Optional[ImageStorage] = None,
        max_images: int = 3,
    ) -> None:
        self._image_storage = image_storage or ImageStorage()
        self._max_images = int(max_images)

    def assemble(
        self,
        hits: Sequence[HybridSearchHit],
        *,
        collection: Optional[str],
    ) -> List[JsonDict]:
        default_collection = (collection or "").strip() or None
        image_refs = _collect_image_refs(
            hits, default_collection=default_collection, max_items=self._max_images
        )
        if not image_refs:
            return []

        out: List[JsonDict] = []
        for ref_collection, image_id in image_refs:
            content = self._try_load_image(collection=ref_collection, image_id=image_id)
            if content is None:
                continue
            out.append(content.to_dict())
        return out

    def _try_load_image(
        self, *, collection: str, image_id: str
    ) -> Optional[ImageContent]:
        try:
            path = self._image_storage.get_path(
                collection=collection, image_id=image_id
            )
        except Exception:
            return None
        if path is None:
            return None

        try:
            data = path.read_bytes()
        except OSError:
            return None
        if not data:
            return None

        mime_type = _guess_mime_type(path) or "image/png"
        encoded = base64.b64encode(data).decode("ascii")
        return ImageContent(data=encoded, mimeType=mime_type)


def _collect_image_refs(
    hits: Sequence[HybridSearchHit],
    *,
    default_collection: Optional[str],
    max_items: int,
) -> List[Tuple[str, str]]:
    limit = int(max_items)
    if limit <= 0:
        return []

    out: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for hit in hits:
        meta = getattr(getattr(hit, "record", None), "metadata", None) or {}
        raw_collection = meta.get("collection")
        hit_collection = (
            raw_collection.strip()
            if isinstance(raw_collection, str) and raw_collection.strip()
            else default_collection
        )
        if not hit_collection:
            continue

        image_ids = _normalize_image_refs(meta.get("image_refs"))
        if not image_ids:
            continue
        for image_id in image_ids:
            key = (hit_collection, image_id)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
            if len(out) >= limit:
                return out
    return out


def _guess_mime_type(path: Path) -> Optional[str]:
    mime, _ = mimetypes.guess_type(str(path))
    if isinstance(mime, str) and mime.strip():
        return mime.strip()
    suffix = str(path.suffix or "").lower()
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".gif":
        return "image/gif"
    if suffix == ".webp":
        return "image/webp"
    return None


def _normalize_image_refs(raw: Any) -> List[str]:
    if isinstance(raw, list):
        out: List[str] = []
        for item in raw:
            if not isinstance(item, str):
                continue
            image_id = item.strip()
            if image_id:
                out.append(image_id)
        return out

    if isinstance(raw, str) and raw.strip():
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(decoded, list):
            out2: List[str] = []
            for item in decoded:
                if not isinstance(item, str):
                    continue
                image_id = item.strip()
                if image_id:
                    out2.append(image_id)
            return out2

    return []
