"""
DeepDoc PDF Loader：当 pdf_parser=deepdoc 时使用，内部调用本仓库内实现的 DeepDoc 解析器。

将解析得到的 sections/tables 转为 Document，与现有 pipeline 兼容。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.core.settings import LoaderSettings
from src.ingestion.models import Document
from src.libs.loader.base_loader import BaseLoader


class DeepDocPdfLoader(BaseLoader):
    """
    当 config 中 pdf_parser=deepdoc 时使用的 PDF Loader，调用本仓库内 DeepDoc 解析器，
    将 sections/tables 转为 Document。original 时应由 Pipeline 选用 PdfLoader。
    """

    def __init__(self, settings: Optional[LoaderSettings] = None):
        self._settings = settings or LoaderSettings()

    def load(self, file_path: Union[str, Path]) -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError("File not found: %s" % path)

        parser = getattr(self._settings, "pdf_parser", "original")
        if parser not in ("original", "deepdoc"):
            raise ValueError(
                "pdf_parser 仅支持 original 或 deepdoc，当前为: %s" % parser
            )
        if parser == "original":
            raise RuntimeError(
                "DeepDocPdfLoader 仅在 pdf_parser=deepdoc 时使用；original 请使用 PdfLoader。"
            )

        # pdf_parser == "deepdoc"：调用本仓库内 DeepDoc 解析器，转为 Document
        from src.libs.deepdoc import PlainParser

        pdf_parser = PlainParser()
        sections, tables = pdf_parser(str(path))

        text_parts = []
        for line, _ in sections:
            text_parts.append(line)
        if tables:
            for tbl in tables:
                if isinstance(tbl, (list, tuple)) and len(tbl) >= 2:
                    # (image_or_markdown, content)
                    text_parts.append(tbl[1] if isinstance(tbl[1], str) else "\n".join(tbl[1]))
                elif isinstance(tbl, str):
                    text_parts.append(tbl)
        text = "\n".join(text_parts).strip()

        metadata: Dict[str, Any] = {
            "source_path": str(path.absolute()),
            "filename": path.name,
            "extension": path.suffix.lower(),
            "parser": "deepdoc",
        }
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            metadata["page_count"] = len(reader.pages)
        except Exception:
            pass

        return Document(text=text, metadata=metadata)
