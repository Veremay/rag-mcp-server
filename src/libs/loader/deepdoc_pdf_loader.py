"""
DeepDoc PDF Loader：当 pdf_parser=deepdoc 时使用，内部调用本仓库内实现的 DeepDoc 解析器。

优先使用完整 RAGFlowPdfParser（pdfplumber 转图 + OCR + 版面 + 表格 + crop），
若依赖或模型不可用则回退到 PlainParser。输出 Document 与 PdfLoader 接口兼容（text + metadata.image_refs）。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.core.settings import LoaderSettings
from src.ingestion.models import Document
from src.libs.loader.base_loader import BaseLoader


def _build_document(
    path: Path,
    text: str,
    page_count: Optional[int] = None,
    image_refs: Optional[List[str]] = None,
) -> Document:
    """统一构建与 PdfLoader 兼容的 Document。"""
    metadata: Dict[str, Any] = {
        "source_path": str(path.absolute()),
        "filename": path.name,
        "extension": path.suffix.lower(),
        "parser": "deepdoc",
    }
    if page_count is not None:
        metadata["page_count"] = page_count
    if image_refs:
        metadata["image_refs"] = image_refs
    return Document(text=text.strip(), metadata=metadata)


class DeepDocPdfLoader(BaseLoader):
    """
    当 config 中 pdf_parser=deepdoc 时使用的 PDF Loader。优先使用完整 DeepDoc（OCR+版面+表格），
    否则使用 PlainParser。输出与 PdfLoader 兼容的 Document。
    """

    def __init__(self, settings: Optional[LoaderSettings] = None):
        self._settings = settings or LoaderSettings()

    def load(self, file_path: Union[str, Path]) -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError("File not found: %s" % path)

        parser_name = getattr(self._settings, "pdf_parser", "original")
        if parser_name not in ("original", "deepdoc"):
            raise ValueError(
                "pdf_parser 仅支持 original 或 deepdoc，当前为: %s" % parser_name
            )
        if parser_name == "original":
            raise RuntimeError(
                "DeepDocPdfLoader 仅在 pdf_parser=deepdoc 时使用；original 请使用 PdfLoader。"
            )

        # 优先完整 DeepDoc（RAGFlowPdfParser），失败时回退到 PlainParser
        try:
            return self._load_full(path)
        except Exception as e:
            logging.warning(
                "DeepDoc 完整解析不可用（%s），回退到 PlainParser。", e
            )
            return self._load_plain(path)

    def _load_plain(self, path: Path) -> Document:
        """仅文本的 PlainParser，与现有行为一致。"""
        from src.libs.deepdoc import PlainParser

        pdf_parser = PlainParser()
        sections, tables = pdf_parser(str(path))

        text_parts = []
        for line, _ in sections:
            text_parts.append(line)
        if tables:
            for tbl in tables:
                if isinstance(tbl, (list, tuple)) and len(tbl) >= 2:
                    text_parts.append(
                        tbl[1]
                        if isinstance(tbl[1], str)
                        else "\n".join(tbl[1])
                    )
                elif isinstance(tbl, str):
                    text_parts.append(tbl)
        text = "\n".join(text_parts).strip()

        page_count = None
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            page_count = len(reader.pages)
        except Exception:
            pass

        return _build_document(path, text, page_count=page_count)

    def _load_full(self, path: Path) -> Document:
        """
        完整 DeepDoc：pdfplumber 转图 + OCR + 版面 + 表格 + crop。
        正文放入 document.text，表格放入 metadata["tables"]，便于 pipeline 将每个表格作为独立 chunk
        （与 RAGFlow 的 tokenize_table 使用方式一致）；image_refs 仍写入 metadata 供图片入库。
        """
        from src.libs.deepdoc.parser.pdf_parser import RAGFlowPdfParser

        parser = RAGFlowPdfParser()
        # return_html=True 以得到表格 HTML；need_image=True 以 crop 表格/图
        sections_str, tbls = parser(str(path), need_image=True, zoomin=3, return_html=True)

        # 正文单独作为 document.text，不把表格内容拼进去，以便 split 阶段将表格作为独立 chunk
        body_text = (sections_str or "").strip()
        image_refs: List[str] = []
        tables_meta: List[Dict[str, Any]] = []
        image_dir = Path("data/images")
        image_dir.mkdir(parents=True, exist_ok=True)
        stem = path.stem

        for i, item in enumerate(tbls):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            crop_img, content = item[0], item[1]
            image_ref = None
            if crop_img is not None:
                try:
                    from PIL import Image
                    if isinstance(crop_img, Image.Image):
                        image_name = "%s_deepdoc_%s.png" % (stem, i)
                        save_path = image_dir / image_name
                        crop_img.save(save_path)
                        abs_path = str(save_path.absolute())
                        image_refs.append(abs_path)
                        image_ref = abs_path
                except Exception as ex:
                    logging.debug("Save crop image %s: %s", i, ex)
            if content is not None:
                content_str = (
                    content if isinstance(content, str)
                    else "\n".join(str(c) for c in content)
                )
                tables_meta.append({"text": content_str, "image_ref": image_ref})

        doc = _build_document(path, body_text, page_count=None, image_refs=image_refs)
        if tables_meta:
            doc.metadata["tables"] = tables_meta
        page_count = None
        try:
            page_count = parser.total_page_number(str(path))
        except Exception:
            try:
                from pypdf import PdfReader
                page_count = len(PdfReader(str(path)).pages)
            except Exception:
                pass
        if page_count is not None:
            doc.metadata["page_count"] = page_count
        return doc
