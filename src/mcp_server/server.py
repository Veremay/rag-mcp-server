from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _write_stdout_message(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _jsonrpc_error(
    request_id: Any, code: int, message: str, data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": err}


def _handle_initialize(request_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    protocol_version = params.get("protocolVersion") or "2025-06-18"
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "protocolVersion": protocol_version,
            "serverInfo": {"name": "modular-rag-mcp-server", "version": "0.1.0"},
            "capabilities": {"tools": {}},
        },
    }


def _handle_request(req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if req.get("jsonrpc") != "2.0":
        return _jsonrpc_error(req.get("id"), -32600, "Invalid Request")

    method = req.get("method")
    request_id = req.get("id")
    params = req.get("params")
    params_dict = params if isinstance(params, dict) else {}

    if method == "initialize":
        return _handle_initialize(request_id, params_dict)

    if method is None:
        return _jsonrpc_error(request_id, -32600, "Invalid Request")

    return _jsonrpc_error(
        request_id, -32601, "Method not found", data={"method": method}
    )


def run_stdio_server() -> int:
    _setup_logging()
    logger.info("MCP stdio server started")

    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except Exception as e:
            logger.exception("Failed to parse JSON-RPC message: %s", e)
            continue

        if not isinstance(req, dict):
            logger.warning("Ignoring non-object JSON-RPC message")
            continue

        resp = _handle_request(req)
        if resp is not None:
            _write_stdout_message(resp)
            if req.get("method") == "initialize":
                logger.info("initialize handled (id=%s)", req.get("id"))

    logger.info("MCP stdio server stopped")
    return 0


def main() -> int:
    return run_stdio_server()


if __name__ == "__main__":
    raise SystemExit(main())
