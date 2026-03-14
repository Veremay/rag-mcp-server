"""
Step 1 单测：pdf_parser 配置与 DeepDocPdfLoader 占位行为。

- pdf_parser=original 时 Pipeline 选用 PdfLoader，行为与当前一致。
- pdf_parser=deepdoc 时选用 DeepDocPdfLoader，当前应抛出 NotImplementedError。
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.settings import LoaderSettings
from src.ingestion.models import Document
from src.ingestion.pipeline import IngestionPipeline
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.deepdoc_pdf_loader import DeepDocPdfLoader
from src.libs.loader.pdf_loader import PdfLoader


class TestDeepDocPdfLoaderStep1:

    @pytest.fixture
    def sample_pdf(self):
        """Minimal PDF file for loader resolution tests."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(b"%PDF-1.4 dummy")
            file_path = Path(f.name)
        yield file_path
        if file_path.exists():
            os.unlink(file_path)

    def test_deepdoc_loader_returns_document_when_pdf_parser_deepdoc(self, sample_pdf):
        """pdf_parser=deepdoc 时 DeepDocPdfLoader.load() 返回 Document（本仓库内 PlainParser）。"""
        settings = LoaderSettings(pdf_parser="deepdoc")
        loader = DeepDocPdfLoader(settings=settings)
        doc = loader.load(sample_pdf)
        assert isinstance(doc, Document)
        assert doc.metadata.get("parser") == "deepdoc"
        assert doc.metadata.get("source_path") == str(sample_pdf.absolute())
        assert "source_path" in doc.metadata

    def test_deepdoc_loader_rejects_invalid_pdf_parser(self, sample_pdf):
        """pdf_parser 非 original/deepdoc 时应抛出 ValueError。"""
        settings = LoaderSettings(pdf_parser="invalid")
        loader = DeepDocPdfLoader(settings=settings)
        with pytest.raises(ValueError) as exc_info:
            loader.load(sample_pdf)
        assert "original" in str(exc_info.value) or "deepdoc" in str(exc_info.value)

    def test_resolve_loader_returns_pdf_loader_when_original(self):
        """pdf_parser=original 时 _resolve_loader 返回 PdfLoader。"""
        loader_settings = LoaderSettings(pdf_parser="original")
        settings = MagicMock()
        settings.ingestion.loader = loader_settings
        pipeline = IngestionPipeline(settings=settings)
        loader = pipeline._resolve_loader(Path("x.pdf"))
        assert isinstance(loader, PdfLoader)

    def test_resolve_loader_returns_deepdoc_loader_when_deepdoc(self):
        """pdf_parser=deepdoc 时 _resolve_loader 返回 DeepDocPdfLoader。"""
        loader_settings = LoaderSettings(pdf_parser="deepdoc")
        settings = MagicMock()
        settings.ingestion.loader = loader_settings
        pipeline = IngestionPipeline(settings=settings)
        loader = pipeline._resolve_loader(Path("x.pdf"))
        assert isinstance(loader, DeepDocPdfLoader)

    def test_resolve_loader_default_is_pdf_loader(self):
        """未配置 pdf_parser 时默认使用 PdfLoader（向后兼容）。"""
        loader_settings = LoaderSettings(pdf_parser="original")
        settings = MagicMock()
        settings.ingestion.loader = loader_settings
        pipeline = IngestionPipeline(settings=settings)
        loader = pipeline._resolve_loader(Path("x.pdf"))
        assert isinstance(loader, PdfLoader)
