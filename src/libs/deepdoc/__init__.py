"""
DeepDoc PDF 解析：本仓库内实现的 PDF 解析能力，不依赖 RAGFlow 运行时。

当前提供 PlainParser（pypdf 纯文本）；完整 OCR+版面+表格 的 RAGFlowPdfParser 可后续迁入。
"""
from src.libs.deepdoc.parser.plain_parser import PlainParser

__all__ = ["PlainParser"]
