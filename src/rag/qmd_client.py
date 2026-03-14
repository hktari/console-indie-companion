"""QMD client for querying local knowledge base via MCP stdio."""

import json
import logging
import subprocess
from typing import Any, Protocol, cast
from io import TextIOWrapper

logger = logging.getLogger(__name__)


class QmdQueryResult:
    """Result from a QMD query."""

    def __init__(
        self,
        content: str,
        score: float,
        file: str,
        docid: str,
        metadata: dict[str, Any] | None = None,
    ):
        self.content = content
        self.score = score
        self.file = file
        self.docid = docid
        self.metadata = metadata or {}


class QmdClient(Protocol):
    """Protocol for QMD clients."""

    def query(self, text: str, game_id: str) -> list[QmdQueryResult]:
        """Query the QMD index.

        Args:
            text: Query text.
            game_id: Game/collection identifier.

        Returns:
            List of query results.
        """
        ...


class QmdMcpStdioClient:
    """QMD MCP client using stdio transport.

    Queries the MCP server directly via its stdio transport by spawning
    the 'qmd mcp' command and communicating via JSON-RPC.
    """

    def __init__(self, index_name: str = "game-companion"):
        self.index_name = index_name
        self._process = None
        self._id_counter = 0

    def _start(self):
        if self._process is not None:
            return

        command = ["qmd", "--index", self.index_name, "mcp"]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Send initialize request
        self._call(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "qmd-python-client", "version": "0.1.0"},
            },
        )

        # Send initialized notification
        self._notify("notifications/initialized")

    def _call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if self._process is None:
            raise RuntimeError("MCP process not started")

        self._id_counter += 1
        request_id = self._id_counter
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }

        logger.debug("MCP Request: %s", json.dumps(request, indent=2))
        stdin = cast(TextIOWrapper, self._process.stdin)
        stdin.write(json.dumps(request) + "\n")
        stdin.flush()

        # Read response (blocking for simplicity in POC)
        stdout = cast(TextIOWrapper, self._process.stdout)
        while True:
            line = stdout.readline()
            if not line:
                raise RuntimeError("MCP process exited unexpectedly")

            # Skip non-JSON lines (like build logs or warnings)
            line = line.strip()
            if not line or not line.startswith("{"):
                continue

            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue

            if response.get("id") == request_id:
                logger.debug("MCP Response: %s", json.dumps(response, indent=2))
                if "error" in response:
                    raise RuntimeError(f"MCP error: {response['error']}")
                return response.get("result")
            # Ignore notifications/other messages for now

    def _notify(self, method: str, params: dict[str, Any] | None = None):
        if self._process is None:
            raise RuntimeError("MCP process not started")

        notification = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        stdin = cast(TextIOWrapper, self._process.stdin)
        stdin.write(json.dumps(notification) + "\n")
        stdin.flush()

    def query(self, text: str, game_id: str, limit: int = 5) -> list[QmdQueryResult]:
        """Query via MCP stdio using the 'query' tool with typed sub-queries."""
        self._start()

        # Use the 'query' tool with both lex (keyword) and vec (semantic) searches
        result = self._call(
            "tools/call",
            {
                "name": "query",
                "arguments": {
                    "searches": [
                        {"type": "lex", "query": text},
                        {"type": "vec", "query": text},
                    ],
                    "collections": [game_id],
                    "limit": limit,
                },
            },
        )

        # MCP tools return { content: [{ type: 'text', text: '...' }], structuredContent: { ... } }
        structured = result.get("structuredContent", {})
        results = structured.get("results", [])

        return [
            QmdQueryResult(
                content=r.get("snippet", ""),
                score=r.get("score", 0.0),
                file=r.get("file", ""),
                docid=r.get("docid", ""),
                metadata={"source": r.get("file"), "mcp": True},
            )
            for r in results
        ]

    def __del__(self):
        if self._process:
            self._process.terminate()
