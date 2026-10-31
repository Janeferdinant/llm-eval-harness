"""A minimal MCP server over stdio with one tool, `echo`, used to record and test the agent route.

Newline-delimited JSON-RPC 2.0. It implements only initialize, tools/list and tools/call.
"""
import json
import sys

TOOL = {
    "name": "echo",
    "description": "Return the given text prefixed with 'echo: '.",
    "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
}


def reply(msg_id, result=None, error=None):
    body = {"jsonrpc": "2.0", "id": msg_id}
    body.update({"error": error} if error else {"result": result})
    sys.stdout.write(json.dumps(body) + "\n")
    sys.stdout.flush()


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line)
        method, msg_id = msg.get("method"), msg.get("id")
        if method == "initialize":
            version = msg.get("params", {}).get("protocolVersion", "2024-11-05")
            reply(msg_id, {"protocolVersion": version, "capabilities": {"tools": {}},
                           "serverInfo": {"name": "echo", "version": "1"}})
        elif method == "tools/list":
            reply(msg_id, {"tools": [TOOL]})
        elif method == "tools/call":
            text = msg["params"].get("arguments", {}).get("text", "")
            reply(msg_id, {"content": [{"type": "text", "text": f"echo: {text}"}]})
        elif msg_id is not None:
            reply(msg_id, error={"code": -32601, "message": f"unknown method {method}"})


if __name__ == "__main__":
    main()
