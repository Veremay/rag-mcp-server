"""
PlainParser：仅用 pypdf 逐页 extract_text 提取文本，产出 (sections, tables)。

移植自 RAGFlow deepdoc.parser.pdf_parser.PlainParser，无 RAGFlow 依赖，供 DeepDoc 路径
在未迁入完整 OCR/版面/表格 时使用，满足「能对给定 PDF 跑通解析得到 sections/tables」。
"""
from __future__ import annotations

import logging
from io import BytesIO
from typing import Any, List, Tuple, Union

logger = logging.getLogger(__name__)


class PlainParser:
    """
    纯文本 PDF 解析器：pypdf 按页 extract_text，返回 (sections, tables)。
    sections 为 [(line_text, position_tag)]，tables 为空列表。
    """

    def __init__(self) -> None:
        self.outlines: List[tuple] = []

    def __call__(
        self,
        filename: Union[str, bytes],
        from_page: int = 0,
        to_page: int = 100000,
        **kwargs: Any,
    ) -> Tuple[List[Tuple[str, str]], List[Any]]:
        """
        解析 PDF，返回 (sections, tables)。
        sections: [(line, "")] 每行文本与占位 position_tag；
        tables: [] 本解析器不提取表格。
        """
        self.outlines = []
        lines: List[str] = []
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError(
                "PlainParser 需要 pypdf，请执行: pip install pypdf"
            ) from e

        try:
            pdf = PdfReader(
                filename if isinstance(filename, str) else BytesIO(filename)
            )
            for page in pdf.pages[from_page:to_page]:
                text = page.extract_text()
                if text:
                    lines.extend(text.split("\n"))

            outlines = getattr(pdf, "outline", None) or []

            def _dfs(arr: list, depth: int) -> None:
                for a in arr:
                    if isinstance(a, dict) and "/Title" in a:
                        self.outlines.append((a["/Title"], depth))
                    elif isinstance(a, list):
                        _dfs(a, depth + 1)

            if outlines:
                _dfs(outlines, 0)
        except Exception:
            logger.exception("Outlines exception")
        if not self.outlines:
            logger.debug("No outlines in PDF")

        return [(line, "") for line in lines], []

    def crop(self, ck: Any, need_position: bool) -> None:
        raise NotImplementedError("PlainParser does not support crop")

    @staticmethod
    def remove_tag(txt: str) -> str:
        """PlainParser 无 position tag，直接返回原文。"""
        return txt
