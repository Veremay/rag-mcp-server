"""Document loaders：按扩展名与配置加载文件为 Document。"""
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.deepdoc_pdf_loader import DeepDocPdfLoader
from src.libs.loader.pdf_loader import PdfLoader

__all__ = ["BaseLoader", "PdfLoader", "DeepDocPdfLoader"]
