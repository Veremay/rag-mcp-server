from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

from src.mcp_server.protocol_handler import ProtocolHandler, ToolSchema
from src.mcp_server.tools.get_document_summary import (
    GetDocumentSummaryParams,
    get_document_summary,
)
from src.mcp_server.tools.list_collections import list_collections
from src.mcp_server.tools.query_knowledge_hub import (
    QueryKnowledgeHubParams,
    query_knowledge_hub,
)
from src.core.trace.trace_context import TraceContext
from src.observability.logger import write_trace
from src.core.settings import load_settings

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


def _ensure_project_root() -> None:
    """Ensure the current working directory is the project root."""
    if os.environ.get("MODULAR_RAG_SKIP_ROOT_CHECK"):
        return

    # src/mcp_server/server.py -> parents[2] is project root
    root = Path(__file__).resolve().parents[2]
    if os.getcwd() != str(root):
        os.chdir(root)
        logger.info("Changed working directory to project root: %s", root)


def run_stdio_server() -> int:
    _setup_logging()
    _ensure_project_root()
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
    handler.register_tool(
        ToolSchema(
            name="list_collections",
            description="列出 data/documents/ 下的集合目录并返回基础统计",
            input_schema={"type": "object", "properties": {}},
        ),
        handler=_handle_list_collections,
    )
    handler.register_tool(
        ToolSchema(
            name="get_document_summary",
            description="按 doc_id 返回文档的 title/summary/tags",
            input_schema={
                "type": "object",
                "properties": {"doc_id": {"type": "string"}},
                "required": ["doc_id"],
            },
        ),
        handler=_handle_get_document_summary,
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

    trace = TraceContext(trace_type="query")
    try:
        return query_knowledge_hub(
            QueryKnowledgeHubParams(query=query, top_k=top_k, collection=collection),
            trace=trace,
        )
    finally:
        try:
            trace.finish()
            settings = load_settings()
            write_trace(trace.to_dict(), settings=settings)
        except Exception as e:
            logger.error("Failed to write trace: %s", e)


def _handle_list_collections(args: Dict[str, Any]) -> Dict[str, Any]:
    _ = args
    return list_collections()


def _handle_get_document_summary(args: Dict[str, Any]) -> Dict[str, Any]:
    doc_id = args.get("doc_id")
    if not isinstance(doc_id, str):
        raise ValueError("doc_id must be a string")
    return get_document_summary(GetDocumentSummaryParams(doc_id=doc_id))


def main() -> int:
    return run_stdio_server()


if __name__ == "__main__":
    raise SystemExit(main())
