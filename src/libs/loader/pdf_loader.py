from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

from src.ingestion.models import Document
from src.libs.loader.base_loader import BaseLoader


class PdfLoader(BaseLoader):
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
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    parts.append(extracted)
            text = "\n\n".join(parts).strip()
        except Exception:
            text = ""

        metadata: Dict[str, Any] = {
            "source_path": str(path.absolute()),
            "filename": path.name,
            "extension": path.suffix.lower(),
        }
        if page_count is not None:
            metadata["page_count"] = page_count

        return Document(text=text, metadata=metadata)
