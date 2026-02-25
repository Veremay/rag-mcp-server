from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional, Union

from PIL import Image

from src.core.settings import LoaderSettings
from src.ingestion.models import Document
from src.libs.loader.base_loader import BaseLoader


class PdfLoader(BaseLoader):
    def __init__(self, settings: Optional[LoaderSettings] = None):
        self._settings = settings or LoaderSettings()

    def load(self, file_path: Union[str, Path]) -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        text = ""
        page_count: int | None = None

        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "缺少依赖 pypdf，无法解析 PDF。请先执行：pip install pypdf"
            ) from e

        try:
            reader = PdfReader(str(path))
            if getattr(reader, "is_encrypted", False):
                try:
                    reader.decrypt("")
                except Exception:
                    pass

            page_count = len(reader.pages)
            parts: list[str] = []
            image_refs: list[str] = []

            # Ensure image directory exists
            image_dir = Path("data/images")
            image_dir.mkdir(parents=True, exist_ok=True)

            for page_num, page in enumerate(reader.pages):
                extracted = page.extract_text()
                page_part = extracted if extracted else ""

                # Extract images from page
                try:
                    for image_file in page.images:
                        # Filter by file size
                        if len(image_file.data) < self._settings.min_image_size_kb * 1024:
                            continue

                        # Filter by dimensions
                        try:
                            with Image.open(io.BytesIO(image_file.data)) as img:
                                if img.width < self._settings.min_image_width or img.height < self._settings.min_image_height:
                                    continue
                        except Exception:
                            # Skip if image cannot be opened/verified
                            continue

                        image_name = f"{path.stem}_p{page_num}_{image_file.name}"
                        image_save_path = image_dir / image_name
                        
                        with open(image_save_path, "wb") as f:
                            f.write(image_file.data)
                        
                        abs_path = str(image_save_path.absolute())
                        image_refs.append(abs_path)
                        # Append image reference to text so splitter keeps it in context
                        page_part += f"\n\n![Image]({abs_path})\n"
                except Exception:
                    # Ignore image extraction errors to keep text processing safe
                    pass

                if page_part:
                    parts.append(page_part)

            text = "\n\n".join(parts).strip()
        except Exception:
            text = ""
            image_refs = []

        metadata: Dict[str, Any] = {
            "source_path": str(path.absolute()),
            "filename": path.name,
            "extension": path.suffix.lower(),
        }
        if image_refs:
            metadata["image_refs"] = image_refs
        if page_count is not None:
            metadata["page_count"] = page_count

        return Document(text=text, metadata=metadata)
