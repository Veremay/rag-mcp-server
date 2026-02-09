import pytest

from src.mcp_server.protocol_handler import ProtocolHandler, ToolSchema


@pytest.mark.unit
def test_protocol_initialize_ok() -> None:
    handler = ProtocolHandler()
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18"},
    }
    resp = handler.handle(req)
    assert resp is not None
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert resp["result"]["serverInfo"]["name"] == "modular-rag-mcp-server"
    assert "capabilities" in resp["result"]
    assert "tools" in resp["result"]["capabilities"]


@pytest.mark.unit
def test_protocol_tools_list_empty_ok() -> None:
    handler = ProtocolHandler()
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    resp = handler.handle(req)
    assert resp is not None
    assert resp["result"]["tools"] == []


@pytest.mark.unit
def test_protocol_tools_call_missing_name_is_invalid_params() -> None:
    handler = ProtocolHandler()
    req = {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {}}
    resp = handler.handle(req)
    assert resp is not None
    assert resp["error"]["code"] == -32602


@pytest.mark.unit
def test_protocol_tools_call_unknown_tool_is_not_found() -> None:
    handler = ProtocolHandler()
    req = {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "x"}}
    resp = handler.handle(req)
    assert resp is not None
    assert resp["error"]["code"] == -32601


@pytest.mark.unit
def test_protocol_method_not_found() -> None:
    handler = ProtocolHandler()
    req = {"jsonrpc": "2.0", "id": 5, "method": "nope"}
    resp = handler.handle(req)
    assert resp is not None
    assert resp["error"]["code"] == -32601


@pytest.mark.unit
def test_protocol_notification_no_response() -> None:
    handler = ProtocolHandler()
    req = {"jsonrpc": "2.0", "method": "tools/list"}
    assert handler.handle(req) is None


@pytest.mark.unit
def test_protocol_tools_call_routes_to_registered_handler() -> None:
    handler = ProtocolHandler()
    handler.register_tool(
        ToolSchema(
            name="echo",
            description="Echo",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        ),
        handler=lambda args: {"ok": True, "args": dict(args)},
    )

    req = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {"name": "echo", "arguments": {"q": "hi"}},
    }
    resp = handler.handle(req)
    assert resp is not None
    assert resp["result"]["ok"] is True
    assert resp["result"]["args"]["q"] == "hi"
