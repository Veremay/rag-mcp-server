from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.mcp_server.tools.get_document_summary import (
    _extract_summary_fields,
    _find_metadata_in_jsonl,
    _load_one_metadata_by_doc_id,
)


@pytest.mark.unit
def test_find_metadata_in_jsonl_returns_none_when_missing(tmp_path: Path) -> None:
    assert _find_metadata_in_jsonl(tmp_path / "missing.jsonl", doc_id="x") is None


@pytest.mark.unit
def test_find_metadata_in_jsonl_finds_first_match(tmp_path: Path) -> None:
    doc_id = "doc-123"
    path = tmp_path / "store.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {"id": "c1", "metadata": {"doc_id": "other", "title": "t0"}}
                ),
                json.dumps(
                    {
                        "id": "c2",
                        "metadata": {
                            "doc_id": doc_id,
                            "title": "t1",
                            "summary": "s1",
                            "tags": ["a", "b"],
                        },
                    }
                ),
                json.dumps({"id": "c3", "metadata": {"doc_id": doc_id, "title": "t2"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = _find_metadata_in_jsonl(path, doc_id=doc_id)
    assert out is not None
    assert out["doc_id"] == doc_id
    assert out["title"] == "t1"


@pytest.mark.unit
def test_extract_summary_fields_applies_fallbacks() -> None:
    title, summary, tags = _extract_summary_fields({"filename": "a.pdf"})
    assert title == "a.pdf"
    assert summary == "a.pdf"
    assert tags == []


@dataclass
class _VectorStoreCfg:
    backend: str
    persist_path: str
    collection_name: str


@dataclass
class _Settings:
    vector_store: Any


@pytest.mark.unit
def test_load_one_metadata_by_doc_id_jsonl(tmp_path: Path) -> None:
    collection_name = "knowledge_hub"
    doc_id = "doc-1"
    jsonl_path = tmp_path / f"{collection_name}.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "id": "c1",
                "metadata": {
                    "doc_id": doc_id,
                    "title": "T",
                    "summary": "S",
                    "tags": ["x"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    settings = _Settings(
        vector_store=_VectorStoreCfg(
            backend="jsonl", persist_path=str(tmp_path), collection_name=collection_name
        )
    )
    meta = _load_one_metadata_by_doc_id(settings, doc_id=doc_id)
    assert meta["doc_id"] == doc_id
    assert meta["title"] == "T"


@pytest.mark.unit
def test_load_one_metadata_by_doc_id_raises_for_missing_doc_id(tmp_path: Path) -> None:
    collection_name = "knowledge_hub"
    jsonl_path = tmp_path / f"{collection_name}.jsonl"
    jsonl_path.write_text(
        json.dumps({"id": "c1", "metadata": {"doc_id": "x"}}) + "\n", encoding="utf-8"
    )

    settings = _Settings(
        vector_store=_VectorStoreCfg(
            backend="jsonl", persist_path=str(tmp_path), collection_name=collection_name
        )
    )
    with pytest.raises(ValueError):
        _load_one_metadata_by_doc_id(settings, doc_id="nope")
