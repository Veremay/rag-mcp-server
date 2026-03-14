"""
图片存储：按 collection 组织目录，维护 image_id -> 相对路径的 index.json。

支持 add_file(移动/复制)、save(字节写入)、get_path、delete、list_images；
写入用原子写(tmp+replace)避免并发或中断导致索引与文件不一致。
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional


class ImageStorage:
    """
    本地文件系统图片存储，每 collection 一个目录 + index.json。
    _validate_name / _normalize_ext 防止路径注入与异常扩展名。
    """

    def __init__(self, base_dir: str | Path = "data/images") -> None:
        self._base_dir = Path(base_dir)

    def add_file(
        self,
        *,
        file_path: str | Path,
        collection: str,
        image_id: str,
        move: bool = True,
    ) -> Path:
        """
        Add an existing file to the storage.
        
        Args:
            file_path: Path to the source file.
            collection: Collection name.
            image_id: Image ID.
            move: If True, move the file; otherwise, copy it.
            
        Returns:
            Path to the stored file.

        若源与目标为同一文件则只更新索引不移动；否则 move/copy 后更新 index，
        保证 get_path 能根据 collection+image_id 解析到正确路径。
        """
        src_path = Path(file_path).resolve()
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_path}")
            
        ext = src_path.suffix
        collection = self._validate_name(collection, name="collection")
        image_id = self._validate_name(image_id, name="image_id")
        
        target_dir = self._base_dir / collection
        target_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = target_dir / f"{image_id}{ext}"
        
        # Check if source and target are the same file
        if src_path == target_path.resolve():
             index = self._load_index(collection=collection)
             index[image_id] = str(target_path.relative_to(self._base_dir))
             self._save_index(collection=collection, index=index)
             return target_path

        # Atomic update not strictly possible with move/copy without temp file,
        # but we can try to be safe.
        import shutil
        
        if move:
            shutil.move(str(src_path), str(target_path))
        else:
            shutil.copy2(str(src_path), str(target_path))
            
        index = self._load_index(collection=collection)
        index[image_id] = str(target_path.relative_to(self._base_dir))
        self._save_index(collection=collection, index=index)
        
        return target_path

    def list_images(self, *, collection: str) -> Dict[str, Path]:
        """
        List all images in a collection.
        
        Args:
            collection: Collection name.
            
        Returns:
            Dictionary mapping image_id to absolute file path.
        """
        collection = self._validate_name(collection, name="collection")
        index = self._load_index(collection=collection)
        
        results = {}
        for image_id, rel_path in index.items():
            full_path = (self._base_dir / rel_path).resolve()
            if full_path.exists():
                results[image_id] = full_path
                
        return results

    def save(
        self,
        *,
        collection: str,
        image_id: str,
        data: bytes,
        ext: str = ".png",
    ) -> Path:
        """将内存中的图片字节写入 collection 目录并更新索引，使用原子写避免半写。"""
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
        """读取 index.json 并过滤为非空 str->str，损坏或不存在时返回空 dict。"""
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
        """原子写 index.json，避免写入中断导致索引损坏。"""
        index_path = self._index_path(collection=collection)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        payload = json.dumps(index, ensure_ascii=False, sort_keys=True)
        self._atomic_write_text(index_path, payload)

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        """先写临时文件再 replace，保证读者要么看到旧版要么看到完整新版。"""
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
        """与 _atomic_write_text 同理，用于 save() 写入图片文件。"""
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
        """统一为 .xxx 且禁止路径/空字符，防止扩展名注入。"""
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
        """禁止空、路径符、. / ..，防止目录穿越。"""
        value = (value or "").strip()
        if not value:
            raise ValueError(f"{name} is empty")
        if any(ch in value for ch in ("/", "\\", "\x00")):
            raise ValueError(f"{name} contains invalid character")
        if value in (".", ".."):
            raise ValueError(f"{name} is invalid")
        return value
