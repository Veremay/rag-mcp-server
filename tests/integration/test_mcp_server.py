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
