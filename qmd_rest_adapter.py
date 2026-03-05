"""QMD REST API Adapter - Wraps MCP protocol in a simple REST endpoint.

This adapter provides a REST API interface to QMD's MCP server, enabling
GPU-accelerated queries without the complexity of the MCP protocol.

Usage:
    python qmd_rest_adapter.py --port 18788 --qmd-url http://localhost:8181/mcp
"""

import argparse
import logging
import uuid
from typing import Any

import requests
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global MCP session state
MCP_SESSION_ID: str | None = None
QMD_MCP_URL: str = "http://localhost:8181/mcp"


def initialize_mcp_session() -> str:
    """Initialize MCP session and return session ID."""
    global MCP_SESSION_ID

    if MCP_SESSION_ID:
        return MCP_SESSION_ID

    session_id = str(uuid.uuid4())

    # Initialize MCP session
    response = requests.post(
        QMD_MCP_URL,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Mcp-Session-Id": session_id,
        },
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "qmd-rest-adapter",
                    "version": "1.0.0",
                },
            },
        },
        timeout=10.0,
    )

    if response.status_code != 200:
        raise RuntimeError(f"MCP initialization failed: {response.text}")

    data = response.json()
    if "error" in data:
        error_msg = data["error"].get("message", "")
        # If already initialized, just use this session
        if "already initialized" in error_msg.lower():
            logger.info(
                "MCP session already initialized, using session: %s", session_id
            )
            MCP_SESSION_ID = session_id
            return session_id
        raise RuntimeError(f"MCP error: {data['error']}")

    MCP_SESSION_ID = session_id
    logger.info("MCP session initialized: %s", session_id)
    logger.info("Server info: %s", data.get("result", {}).get("serverInfo"))

    return session_id


def mcp_search(query: str, collection: str | None, limit: int) -> list[dict[str, Any]]:
    """Execute search via MCP protocol."""
    session_id = initialize_mcp_session()

    # Build search arguments
    args: dict[str, Any] = {
        "query": query,
        "limit": limit,
    }
    if collection:
        args["collection"] = collection

    # Call MCP search tool
    response = requests.post(
        QMD_MCP_URL,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Mcp-Session-Id": session_id,
        },
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": args,
            },
        },
        timeout=30.0,
    )

    if response.status_code != 200:
        raise RuntimeError(f"MCP search failed: {response.text}")

    data = response.json()
    if "error" in data:
        raise RuntimeError(f"MCP error: {data['error']}")

    # Parse MCP response - results are in text content
    result = data.get("result", {})
    content = result.get("content", [])

    # Extract text from content array
    results_text = ""
    for item in content:
        if item.get("type") == "text":
            results_text = item.get("text", "")
            break

    # Parse results (QMD returns formatted text, convert to structured format)
    results = []
    if results_text:
        # Simple parsing - adapt based on actual QMD output format
        for line in results_text.split("\n"):
            if line.strip():
                results.append(
                    {
                        "content": line,
                        "score": 0.8,  # MCP doesn't return scores in text format
                        "file": "unknown",
                        "docid": "unknown",
                    }
                )

    return results


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/query", methods=["POST"])
def query():
    """REST query endpoint compatible with QmdHttpClient.

    Request format:
    {
        "searches": [
            {"type": "lex", "query": "text"},
            {"type": "vec", "query": "text"}
        ],
        "collections": ["collection_name"],
        "limit": 5
    }

    Response format:
    {
        "results": [
            {
                "content": "...",
                "score": 0.95,
                "file": "path/to/file.md",
                "docid": "#abc123",
                "metadata": {}
            }
        ]
    }
    """
    try:
        data = request.get_json()

        # Extract parameters
        searches = data.get("searches", [])
        collections = data.get("collections", [])
        limit = data.get("limit", 5)

        # Combine search queries (QMD MCP search takes single query)
        query_text = " ".join(s.get("query", "") for s in searches)
        collection = collections[0] if collections else None

        # Execute search via MCP
        results = mcp_search(query_text, collection, limit)

        return jsonify({"results": results})

    except Exception as e:
        logger.exception("Query failed")
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="QMD REST API Adapter")
    parser.add_argument("--port", type=int, default=18788, help="Port to listen on")
    parser.add_argument(
        "--qmd-url",
        default="http://localhost:8181/mcp",
        help="QMD MCP server URL",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    global QMD_MCP_URL
    QMD_MCP_URL = args.qmd_url

    logger.info("Starting QMD REST adapter on %s:%d", args.host, args.port)
    logger.info("Connecting to QMD MCP server at %s", QMD_MCP_URL)

    # Initialize session on startup
    try:
        initialize_mcp_session()
    except Exception as e:
        logger.error("Failed to initialize MCP session: %s", e)
        logger.error("Make sure QMD MCP server is running: qmd mcp --http --daemon")
        return

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
