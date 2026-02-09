from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict

from src.mcp_server.protocol_handler import ProtocolHandler

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


def run_stdio_server() -> int:
    _setup_logging()
    logger.info("MCP stdio server started")
    handler = ProtocolHandler()

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

        resp = handler.handle(req)
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
