"""PlainParser 单测：本仓库内 DeepDoc 最小解析实现可独立运行，无 RAGFlow 依赖。"""
import os
import tempfile
from pathlib import Path

import pytest

from src.libs.deepdoc import PlainParser


class TestPlainParser:

    @pytest.fixture
    def sample_pdf(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(b"%PDF-1.4 dummy content for minimal PDF")
            file_path = Path(f.name)
        yield file_path
        if file_path.exists():
            os.unlink(file_path)

    def test_plain_parser_returns_sections_tables(self, sample_pdf):
        """能对给定 PDF 跑通解析得到 sections/tables。"""
        parser = PlainParser()
        sections, tables = parser(sample_pdf)
        assert isinstance(sections, list)
        assert isinstance(tables, list)
        assert tables == []
        # 最小 PDF 可能无文本，sections 可为空或若干 (line, "")
        for item in sections:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert item[1] == ""

    def test_plain_parser_accepts_path_or_binary(self, sample_pdf):
        """支持文件路径或 bytes。"""
        parser = PlainParser()
        with open(sample_pdf, "rb") as f:
            binary = f.read()
        s1, t1 = parser(str(sample_pdf))
        s2, t2 = parser(binary)
        assert t1 == t2 == []
        assert isinstance(s1, list) and isinstance(s2, list)
