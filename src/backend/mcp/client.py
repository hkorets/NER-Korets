import json, subprocess, threading, queue, sys
from typing import Any, Dict, Optional, List

class MCPClient:
    def __init__(self, cmd: Optional[List[str]] = None):
        self.proc = subprocess.Popen(
            cmd or [sys.executable, "-m", "src.backend.mcp.server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._rid = 0
        self._out_q: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._out_q.put(line)

    def _rpc(self, method: str, params: Dict[str, Any], timeout: float = 30.0):
        self._rid += 1
        req = {"jsonrpc": "2.0", "id": self._rid, "method": method, "params": params}
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self._out_q.get(timeout=timeout)
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError(resp["error"])
        return resp["result"]

    def list_tools(self):
        return self._rpc("tools/list", {})

    def call_tool(self, name: str, arguments: Dict[str, Any], timeout: float = 30.0):
        return self._rpc("tools/call", {"name": name, "arguments": arguments}, timeout=timeout)
