"""QMD client for querying local knowledge base via MCP stdio."""

import json
import logging
import subprocess
import select
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
        self._started = False

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
        self._started = True

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

    def _call(
        self, method: str, params: dict[str, Any] | None = None, timeout: float = 15.0
    ) -> Any:
        if self._process is None:
            raise RuntimeError("MCP process not started")

        # Check if process is still alive
        if self._process.poll() is not None:
            stderr_output = self._process.stderr.read() if self._process.stderr else ""
            raise RuntimeError(
                f"MCP process died with exit code {self._process.returncode}. "
                f"Stderr: {stderr_output[:500]}"
            )

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

        # Read response with timeout
        stdout = cast(TextIOWrapper, self._process.stdout)
        stderr = cast(TextIOWrapper, self._process.stderr)
        import time

        start_time = time.time()
        skipped_lines = []

        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                # Capture stderr for diagnostics
                stderr_ready, _, _ = select.select([stderr], [], [], 0)
                stderr_output = ""
                if stderr_ready:
                    stderr_output = stderr.read()

                error_msg = (
                    f"MCP call to {method} timed out after {timeout}s. "
                    f"Request ID: {request_id}, "
                    f"Params: {json.dumps(params, indent=2) if params else 'None'}, "
                    f"Skipped {len(skipped_lines)} non-JSON lines, "
                    f"Process alive: {self._process.poll() is None}"
                )
                if stderr_output:
                    error_msg += f", Stderr: {stderr_output[:500]}"
                if skipped_lines:
                    error_msg += f", Last skipped: {skipped_lines[-3:]}"

                logger.error(error_msg)
                raise TimeoutError(error_msg)

            # Check if process died
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"MCP process exited unexpectedly with code {self._process.returncode}"
                )

            # Use select to check if data is available (with short timeout)
            ready, _, _ = select.select([stdout], [], [], 0.1)
            if not ready:
                continue

            line = stdout.readline()
            if not line:
                raise RuntimeError("MCP process exited unexpectedly")

            # Skip non-JSON lines (like build logs or warnings)
            line = line.strip()
            if not line or not line.startswith("{"):
                if line:  # Log non-empty skipped lines
                    skipped_lines.append(line[:100])
                    logger.debug("Skipping non-JSON line: %s", line[:100])
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
        import time

        self._start()

        query_start = time.time()
        logger.debug(
            "QMD query starting: text='%s', game_id='%s', limit=%d, index='%s'",
            text[:50],
            game_id,
            limit,
            self.index_name,
        )

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

        query_elapsed = time.time() - query_start
        logger.debug("QMD query completed in %.2fs", query_elapsed)

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

    def shutdown(self):
        """Properly shutdown the QMD MCP process."""
        if self._process is not None:
            logger.debug("Shutting down QMD MCP process (index=%s)", self.index_name)
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                logger.warning("QMD process did not terminate, killing it")
                self._process.kill()
                self._process.wait()
            except Exception as e:
                logger.warning("Error shutting down QMD process: %s", e)
            finally:
                self._process = None
                self._started = False

    def __del__(self):
        """Cleanup on garbage collection (fallback only)."""
        if self._started:
            self.shutdown()
