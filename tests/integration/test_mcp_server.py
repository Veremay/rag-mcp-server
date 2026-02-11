import json
import os
import subprocess
import sys
from base64 import b64decode
from pathlib import Path
from typing import Any, Dict

import pytest

from src.core.query_engine.hybrid_search import HybridSearchHit
from src.core.response.multimodal_assembler import MultimodalAssembler
from src.core.response.response_builder import ResponseBuilder
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.vector_store.base_vector_store import VectorRecord


@pytest.mark.integration
def test_mcp_stdio_server_initialize_stdout_clean_stderr_has_logs() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, "-m", "src.mcp_server.server"]

    req: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18"},
    }
    input_payload = json.dumps(req, ensure_ascii=False) + "\n"

    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    try:
        stdout, stderr = proc.communicate(input=input_payload, timeout=8)
    finally:
        if proc.poll() is None:
            proc.kill()

    assert proc.returncode == 0

    out_lines = [l for l in (stdout or "").splitlines() if l.strip()]
    assert len(out_lines) == 1

    resp = json.loads(out_lines[0])
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert "result" in resp
    assert resp["result"]["serverInfo"]["name"] == "modular-rag-mcp-server"
    assert "capabilities" in resp["result"]
    assert "tools" in resp["result"]["capabilities"]

    assert (stderr or "").strip() != ""


@pytest.mark.integration
def test_mcp_response_builder_includes_image_content_when_image_refs_present(
    tmp_path: Path,
) -> None:
    collection = "c_img"
    image_id = "img_1"
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x0bIDAT\x08\xd7c``\x00\x00\x00\x05\x00\x01"
        b"\x0d\n\x2d\xb4"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    images_dir = tmp_path / "images"
    storage = ImageStorage(base_dir=images_dir)
    storage.save(collection=collection, image_id=image_id, data=png_bytes, ext=".png")

    record = VectorRecord(
        id="cid",
        embedding=[0.0, 0.0, 0.0],
        content="这里是一段包含图片描述的文本。",
        metadata={
            "source_path": "architecture.pdf",
            "page": 5,
            "section_path": "s1",
            "image_refs": [image_id],
        },
    )
    hits = [
        HybridSearchHit(
            chunk_id="cid",
            score=0.9,
            record=record,
            dense_rank=1,
            sparse_rank=1,
        )
    ]

    builder = ResponseBuilder(
        multimodal_assembler=MultimodalAssembler(image_storage=storage, max_images=3)
    )
    resp = builder.build(hits, query="test", collection=collection)

    assert isinstance(resp.get("content"), list)
    assert resp["content"][0]["type"] == "text"
    assert resp["content"][1]["type"] == "image"
    assert resp["content"][1]["mimeType"] == "image/png"
    assert b64decode(resp["content"][1]["data"]) == png_bytes


@pytest.mark.integration
def test_mcp_response_builder_supports_json_encoded_image_refs_and_collection_metadata(
    tmp_path: Path,
) -> None:
    image_id = "img_shared"
    png_bytes_1 = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x0bIDAT\x08\xd7c``\x00\x00\x00\x05\x00\x01"
        b"\x0d\n\x2d\xb4"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    png_bytes_2 = png_bytes_1[:-1] + b"\x81"

    images_dir = tmp_path / "images"
    storage = ImageStorage(base_dir=images_dir)
    storage.save(collection="c1", image_id=image_id, data=png_bytes_1, ext=".png")
    storage.save(collection="c2", image_id=image_id, data=png_bytes_2, ext=".png")

    hits = [
        HybridSearchHit(
            chunk_id="cid1",
            score=0.9,
            record=VectorRecord(
                id="cid1",
                embedding=[0.0, 0.0, 0.0],
                content="chunk 1",
                metadata={
                    "collection": "c1",
                    "image_refs": json.dumps([image_id], ensure_ascii=False),
                },
            ),
            dense_rank=1,
            sparse_rank=1,
        ),
        HybridSearchHit(
            chunk_id="cid2",
            score=0.8,
            record=VectorRecord(
                id="cid2",
                embedding=[0.0, 0.0, 0.0],
                content="chunk 2",
                metadata={
                    "collection": "c2",
                    "image_refs": json.dumps([image_id], ensure_ascii=False),
                },
            ),
            dense_rank=2,
            sparse_rank=2,
        ),
    ]

    builder = ResponseBuilder(
        multimodal_assembler=MultimodalAssembler(image_storage=storage, max_images=3)
    )
    resp = builder.build(hits, query="test", collection=None)

    assert isinstance(resp.get("content"), list)
    assert resp["content"][0]["type"] == "text"
    assert resp["content"][1]["type"] == "image"
    assert resp["content"][2]["type"] == "image"
    assert b64decode(resp["content"][1]["data"]) == png_bytes_1
    assert b64decode(resp["content"][2]["data"]) == png_bytes_2


@pytest.mark.integration
def test_mcp_stdio_server_query_knowledge_hub_tool_roundtrip(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, "-m", "src.mcp_server.server"]

    cfg_path = tmp_path / "settings.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "llm:",
                "  provider: ollama",
                "  model: x",
                "  azure_endpoint: null",
                "  api_key: null",
                "  base_url: null",
                "embedding:",
                "  provider: local",
                "  model: fake",
                "  base_url: null",
                "  api_key: null",
                "vision_llm:",
                "  provider: ollama",
                "  model: x",
                "vector_store:",
                "  backend: jsonl",
                f"  persist_path: {str(tmp_path / 'vector')}",
                "  collection_name: __empty__",
                "ingestion:",
                "  splitter:",
                "    provider: recursive",
                "    chunk_size: 1000",
                "    chunk_overlap: 200",
                "  transform: {}",
                "retrieval:",
                "  sparse_backend: bm25",
                "  fusion_algorithm: rrf",
                "  top_k_dense: 5",
                "  top_k_sparse: 5",
                "  top_k_final: 5",
                "rerank:",
                "  backend: none",
                "  model: x",
                "  top_m: 5",
                "evaluation:",
                "  backends: [custom]",
                "  golden_test_set: ''",
                "observability:",
                "  enabled: false",
                "  log_file: ''",
                "  dashboard_port: 0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    req_init: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18"},
    }
    req_list: Dict[str, Any] = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    req_call: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "query_knowledge_hub", "arguments": {"query": "test"}},
    }
    input_payload = (
        json.dumps(req_init, ensure_ascii=False)
        + "\n"
        + json.dumps(req_list, ensure_ascii=False)
        + "\n"
        + json.dumps(req_call, ensure_ascii=False)
        + "\n"
    )

    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env={**os.environ, "MODULAR_RAG_CONFIG_PATH": str(cfg_path)},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    try:
        stdout, stderr = proc.communicate(input=input_payload, timeout=8)
    finally:
        if proc.poll() is None:
            proc.kill()

    assert proc.returncode == 0

    out_lines = [l for l in (stdout or "").splitlines() if l.strip()]
    assert len(out_lines) == 3
    init_resp = json.loads(out_lines[0])
    list_resp = json.loads(out_lines[1])
    call_resp = json.loads(out_lines[2])

    assert init_resp["id"] == 1
    assert list_resp["id"] == 2
    tools = list_resp["result"]["tools"]
    assert any(t.get("name") == "query_knowledge_hub" for t in tools)

    assert call_resp["id"] == 3
    result = call_resp["result"]
    assert isinstance(result.get("content"), list)
    assert result["content"][0]["type"] == "text"
    assert (
        isinstance(result["content"][0]["text"], str)
        and result["content"][0]["text"].strip()
    )
    assert "structuredContent" in result
    assert isinstance(result["structuredContent"].get("citations"), list)

    assert (stderr or "").strip() != ""
