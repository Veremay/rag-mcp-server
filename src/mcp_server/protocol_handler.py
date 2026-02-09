from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)


JsonDict = Dict[str, Any]
ToolHandler = Callable[[JsonDict], JsonDict]


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    input_schema: JsonDict

    def to_dict(self) -> JsonDict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": dict(self.input_schema),
        }


class ProtocolHandler:
    def __init__(
        self,
        *,
        server_name: str = "modular-rag-mcp-server",
        server_version: str = "0.1.0",
        tools: Optional[Mapping[str, ToolSchema]] = None,
        tool_handlers: Optional[Mapping[str, ToolHandler]] = None,
    ) -> None:
        self._server_name = server_name
        self._server_version = server_version
        self._tools: Dict[str, ToolSchema] = dict(tools or {})
        self._tool_handlers: Dict[str, ToolHandler] = dict(tool_handlers or {})

    def handle(self, req: JsonDict) -> Optional[JsonDict]:
        request_id = req.get("id")

        if req.get("jsonrpc") != "2.0":
            return self._error(request_id, -32600, "Invalid Request")

        method = req.get("method")
        if not isinstance(method, str) or not method:
            return self._error(request_id, -32600, "Invalid Request")

        if request_id is None:
            return None

        params = req.get("params")
        params_dict = params if isinstance(params, dict) else None

        try:
            if method == "initialize":
                return self._result(
                    request_id, self._handle_initialize(params_dict or {})
                )

            if method == "tools/list":
                if params_dict not in (None, {}):
                    return self._error(request_id, -32602, "Invalid params")
                return self._result(request_id, self._handle_tools_list())

            if method == "tools/call":
                if params_dict is None:
                    return self._error(request_id, -32602, "Invalid params")
                return self._result(request_id, self._handle_tools_call(params_dict))

            return self._error(
                request_id, -32601, "Method not found", data={"method": method}
            )
        except (ValueError, TypeError) as e:
            logger.info("Protocol handler invalid params: %s", e)
            return self._error(request_id, -32602, "Invalid params")
        except LookupError as e:
            logger.info("Protocol handler not found: %s", e)
            return self._error(request_id, -32601, "Method not found")
        except Exception:
            logger.exception("Protocol handler internal error")
            return self._error(request_id, -32603, "Internal error")

    def register_tool(
        self, schema: ToolSchema, handler: Optional[ToolHandler] = None
    ) -> None:
        self._tools[schema.name] = schema
        if handler is not None:
            self._tool_handlers[schema.name] = handler

    def _handle_initialize(self, params: JsonDict) -> JsonDict:
        protocol_version = params.get("protocolVersion") or "2025-06-18"
        return {
            "protocolVersion": protocol_version,
            "serverInfo": {"name": self._server_name, "version": self._server_version},
            "capabilities": {"tools": {}},
        }

    def _handle_tools_list(self) -> JsonDict:
        tools: List[JsonDict] = [t.to_dict() for t in self._tools.values()]
        tools.sort(key=lambda x: str(x.get("name") or ""))
        return {"tools": tools}

    def _handle_tools_call(self, params: JsonDict) -> JsonDict:
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Invalid params: missing tool name")

        arguments = params.get("arguments")
        if arguments is None:
            arguments_dict: JsonDict = {}
        elif isinstance(arguments, dict):
            arguments_dict = arguments
        else:
            raise ValueError("Invalid params: arguments must be an object")

        handler = self._tool_handlers.get(name)
        if handler is None:
            raise LookupError(f"Tool not found: {name}")

        out = handler(arguments_dict)
        if not isinstance(out, dict):
            raise TypeError("Tool result must be an object")
        return out

    def _result(self, request_id: Any, result: JsonDict) -> JsonDict:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _error(
        self,
        request_id: Any,
        code: int,
        message: str,
        data: Optional[JsonDict] = None,
    ) -> JsonDict:
        err: JsonDict = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": request_id, "error": err}
