from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict

from src.mcp_server.protocol_handler import ProtocolHandler, ToolSchema
from src.mcp_server.tools.query_knowledge_hub import (
    QueryKnowledgeHubParams,
    query_knowledge_hub,
)

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
    handler.register_tool(
        ToolSchema(
            name="query_knowledge_hub",
            description="主检索入口：混合检索 + Rerank，返回带引用的结果",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1},
                    "collection": {"type": "string"},
                },
                "required": ["query"],
            },
        ),
        handler=_handle_query_knowledge_hub,
    )

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


def _handle_query_knowledge_hub(args: Dict[str, Any]) -> Dict[str, Any]:
    query = args.get("query")
    if not isinstance(query, str):
        raise ValueError("query must be a string")

    top_k = args.get("top_k")
    if top_k is not None and not isinstance(top_k, int):
        raise ValueError("top_k must be an integer")

    collection = args.get("collection")
    if collection is not None and not isinstance(collection, str):
        raise ValueError("collection must be a string")

    return query_knowledge_hub(
        QueryKnowledgeHubParams(query=query, top_k=top_k, collection=collection)
    )


def main() -> int:
    return run_stdio_server()


if __name__ == "__main__":
    raise SystemExit(main())
