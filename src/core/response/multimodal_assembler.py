"""
多模态组装器：从检索命中的 metadata 收集图片引用并转为内联图片块。

命中 metadata 中的 image_refs 与 collection 决定图片路径，通过 ImageStorage 解析
为本地文件后读入并 base64 编码，便于 API 直接返回 data URL 形态，客户端无需再请求图片接口。
"""
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
    """
    单张图片的 payload：data 为 base64 字符串、mimeType 供前端正确渲染。
    to_dict 产出与 content 数组中 image 块一致的结构。
    """
    data: str
    mimeType: str

    def to_dict(self) -> JsonDict:
        return {"type": "image", "data": str(self.data), "mimeType": str(self.mimeType)}


class MultimodalAssembler:
    """
    从 hits 的 metadata 收集 image_refs，按 collection 从 ImageStorage 加载图片并编码。

    max_images 限制返回图片数量，避免单次响应过大；支持注入 image_storage 便于测试或替换存储。
    """

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
        """
        按命中顺序收集 image_refs（去重、限制条数），逐条加载并转为 ImageContent.to_dict()。
        collection 作为命中未带 collection 时的默认值，与检索阶段使用的集合保持一致才能找到图片路径。
        """
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
        """
        根据 collection + image_id 解析路径、读文件并 base64 编码。
        任何异常或空文件都返回 None，避免单张图片失败导致整段 assemble 报错。
        """
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
    """
    按命中顺序从 metadata 抽取 (collection, image_id)，去重并最多返回 max_items 条。
    命中无 collection 时用 default_collection，保证与当前检索集合一致才能在后端找到图片。
    """
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
    """
    根据路径或后缀推断 MIME 类型，供前端正确渲染。先试 mimetypes，再按常见图片后缀兜底。
    """
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
    """
    将 metadata 中的 image_refs 规范为字符串列表。支持 list 或 JSON 字符串，
    过滤非字符串与空串，避免下游解析或路径拼接出错。
    """
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
