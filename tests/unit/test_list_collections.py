from pathlib import Path

import pytest

from src.mcp_server.tools.list_collections import _scan_documents_dir, list_collections


@pytest.mark.unit
def test_scan_documents_dir_returns_empty_when_missing(tmp_path: Path) -> None:
    base_dir = tmp_path / "data" / "documents"
    assert _scan_documents_dir(base_dir) == []


@pytest.mark.unit
def test_scan_documents_dir_lists_collections_and_counts_files(tmp_path: Path) -> None:
    base_dir = tmp_path / "data" / "documents"
    (base_dir / "c1").mkdir(parents=True)
    (base_dir / "c2").mkdir(parents=True)
    (base_dir / "c1" / "a.txt").write_text("a", encoding="utf-8")
    (base_dir / "c1" / "nested").mkdir()
    (base_dir / "c1" / "nested" / "b.txt").write_text("b", encoding="utf-8")
    (base_dir / "c2" / "x.pdf").write_bytes(b"%PDF-1.4")

    out = _scan_documents_dir(base_dir)
    assert [c.name for c in out] == ["c1", "c2"]
    assert [c.file_count for c in out] == [2, 1]


@pytest.mark.unit
def test_list_collections_returns_mcp_shape_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(Path.cwd())
    out = list_collections()
    assert out["content"][0]["type"] == "text"
    assert isinstance(out["content"][0]["text"], str)
    assert "structuredContent" in out
    assert isinstance(out["structuredContent"]["collections"], list)
