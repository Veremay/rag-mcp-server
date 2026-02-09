import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest


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
def test_mcp_stdio_server_query_knowledge_hub_tool_roundtrip() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, "-m", "src.mcp_server.server"]

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
