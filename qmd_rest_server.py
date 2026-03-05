"""QMD REST API Server - Direct CLI wrapper with persistent Flask server.

This provides a REST API that calls QMD CLI directly, avoiding MCP complexity
while still providing a persistent HTTP server for GPU-accelerated queries.

Usage:
    python qmd_rest_server.py --port 18788
"""

import argparse
import json
import logging
import subprocess
from typing import Any

from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global configuration
QMD_INDEX_NAME = "index"


def qmd_search(query: str, collection: str | None, limit: int) -> list[dict[str, Any]]:
    """Execute search via QMD CLI."""
    try:
        # Build command
        cmd = [
            "qmd",
            "--index", QMD_INDEX_NAME,
            "search",
            query,
            "--json",
            "-n", str(limit),
        ]
        
        if collection:
            cmd.extend(["-c", collection])
        
        # Execute
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30.0,
        )
        
        # Parse JSON output
        if result.stdout.strip():
            results = json.loads(result.stdout)
            if isinstance(results, list):
                return results
        
        return []
    
    except subprocess.CalledProcessError as e:
        logger.error("QMD search failed: %s", e.stderr)
        raise RuntimeError(f"QMD search failed: {e.stderr}")
    except json.JSONDecodeError as e:
        logger.error("Failed to parse QMD output: %s", e)
        raise RuntimeError(f"Failed to parse QMD output: {e}")
    except Exception as e:
        logger.error("QMD search error: %s", e)
        raise


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
                "snippet": "...",
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
        
        # Combine search queries (QMD CLI search takes single query)
        query_text = " ".join(s.get("query", "") for s in searches if s.get("query"))
        collection = collections[0] if collections else None
        
        if not query_text:
            return jsonify({"error": "No query provided"}), 400
        
        # Execute search via CLI
        qmd_results = qmd_search(query_text, collection, limit)
        
        # Transform to expected format
        results = []
        for r in qmd_results:
            results.append({
                "content": r.get("snippet", ""),
                "score": r.get("score", 0.0),
                "file": r.get("file", ""),
                "docid": r.get("docid", ""),
                "snippet": r.get("snippet", ""),
                "metadata": {
                    "title": r.get("title", ""),
                    "context": r.get("context", ""),
                },
            })
        
        return jsonify({"results": results})
    
    except Exception as e:
        logger.exception("Query failed")
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="QMD REST API Server")
    parser.add_argument("--port", type=int, default=18788, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--index", default="index", help="QMD index name")
    args = parser.parse_args()
    
    global QMD_INDEX_NAME
    QMD_INDEX_NAME = args.index
    
    logger.info("Starting QMD REST server on %s:%d", args.host, args.port)
    logger.info("Using QMD index: %s", QMD_INDEX_NAME)
    
    # Test QMD availability
    try:
        result = subprocess.run(
            ["qmd", "status"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        logger.info("QMD status: %s", result.stdout.strip())
    except Exception as e:
        logger.error("Failed to check QMD status: %s", e)
        logger.error("Make sure QMD is installed: cargo install qmd")
        return
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
