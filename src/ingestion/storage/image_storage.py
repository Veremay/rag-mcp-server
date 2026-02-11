from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional


class ImageStorage:
    def __init__(self, base_dir: str | Path = "data/images") -> None:
        self._base_dir = Path(base_dir)

    def save(
        self,
        *,
        collection: str,
        image_id: str,
        data: bytes,
        ext: str = ".png",
    ) -> Path:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data must be bytes")
        if not data:
            raise ValueError("data is empty")

        collection = self._validate_name(collection, name="collection")
        image_id = self._validate_name(image_id, name="image_id")
        ext = self._normalize_ext(ext)

        target_dir = self._base_dir / collection
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / f"{image_id}{ext}"
        self._atomic_write_bytes(target_path, bytes(data))

        index = self._load_index(collection=collection)
        index[image_id] = str(target_path.relative_to(self._base_dir))
        self._save_index(collection=collection, index=index)

        return target_path

    def get_path(self, *, collection: str, image_id: str) -> Optional[Path]:
        collection = self._validate_name(collection, name="collection")
        image_id = self._validate_name(image_id, name="image_id")

        index = self._load_index(collection=collection)
        rel = index.get(image_id)
        if not rel:
            return None

        path = (self._base_dir / rel).resolve()
        base = self._base_dir.resolve()
        try:
            path.relative_to(base)
        except ValueError:
            return None

        return path

    def delete(self, *, collection: str, image_id: str) -> bool:
        """
        Delete an image.

        Args:
            collection: Collection name.
            image_id: Image ID.

        Returns:
            True if deleted, False if not found.
        """
        collection = self._validate_name(collection, name="collection")
        image_id = self._validate_name(image_id, name="image_id")

        index = self._load_index(collection=collection)
        rel = index.get(image_id)
        if not rel:
            return False

        path = (self._base_dir / rel).resolve()
        
        # Remove from index first
        del index[image_id]
        self._save_index(collection=collection, index=index)

        # Then delete file
        try:
            if path.exists():
                os.remove(path)
                return True
        except OSError:
            pass
            
        return False

    def _index_path(self, *, collection: str) -> Path:
        return self._base_dir / collection / "index.json"

    def _load_index(self, *, collection: str) -> Dict[str, str]:
        index_path = self._index_path(collection=collection)
        if not index_path.exists():
            return {}

        try:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        if not isinstance(raw, dict):
            return {}

        index: Dict[str, str] = {}
        for k, v in raw.items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            index[k] = v
        return index

    def _save_index(self, *, collection: str, index: Dict[str, str]) -> None:
        index_path = self._index_path(collection=collection)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        payload = json.dumps(index, ensure_ascii=False, sort_keys=True)
        self._atomic_write_text(index_path, payload)

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
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
                f.write(content)
            os.replace(tmp_path, path)
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _atomic_write_bytes(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = None
        tmp_path: Optional[str] = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=path.name + ".",
                suffix=".tmp",
                dir=str(path.parent),
            )
            with os.fdopen(fd, "wb") as f:
                f.write(content)
            os.replace(tmp_path, path)
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _normalize_ext(ext: str) -> str:
        ext = (ext or "").strip()
        if not ext:
            return ".png"
        if not ext.startswith("."):
            ext = "." + ext
        if len(ext) > 16:
            raise ValueError("ext too long")
        if any(ch in ext for ch in ("/", "\\", "\x00")):
            raise ValueError("ext contains invalid character")
        return ext

    @staticmethod
    def _validate_name(value: str, *, name: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError(f"{name} is empty")
        if any(ch in value for ch in ("/", "\\", "\x00")):
            raise ValueError(f"{name} contains invalid character")
        if value in (".", ".."):
            raise ValueError(f"{name} is invalid")
        return value
