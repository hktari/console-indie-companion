"""QMD client for querying local knowledge base."""

import json
import logging
import subprocess
from typing import Any, Protocol

import requests

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


class QmdHttpClient:
    """QMD HTTP client for querying via REST API."""

    def __init__(self, base_url: str):
        """Initialize HTTP client.

        Args:
            base_url: Base URL of QMD server (e.g., http://localhost:18788).
        """
        self.base_url = base_url.rstrip("/")

    def query(self, text: str, game_id: str, limit: int = 5) -> list[QmdQueryResult]:
        """Query via HTTP API.

        Args:
            text: Query text.
            game_id: Collection name.
            limit: Maximum number of results.

        Returns:
            List of query results.
        """
        try:
            response = requests.post(
                f"{self.base_url}/query",
                headers={"Content-Type": "application/json"},
                json={
                    "searches": [
                        {"type": "lex", "query": text},
                        {"type": "vec", "query": text},
                    ],
                    "collections": [game_id],
                    "limit": limit,
                },
                timeout=10.0,
            )

            if not response.ok:
                raise RuntimeError(
                    f"QMD HTTP error: {response.status_code} {response.text}"
                )

            data = response.json()
            results = data.get("results", [])

            return [
                QmdQueryResult(
                    content=r.get("snippet") or r.get("content", ""),
                    score=r.get("score", 0.0),
                    file=r.get("file", ""),
                    docid=r.get("docid", ""),
                    metadata=r.get("metadata", {"source": r.get("file")}),
                )
                for r in results
            ]

        except Exception as e:
            logger.error("QMD HTTP query failed: %s", e, exc_info=True)
            raise


class QmdCliClient:
    """QMD CLI client for querying via command-line interface."""

    def __init__(self, index_name: str = "game-companion"):
        """Initialize CLI client.

        Args:
            index_name: Name of the QMD index.
        """
        self.index_name = index_name

    def query(self, text: str, game_id: str, limit: int = 5) -> list[QmdQueryResult]:
        """Query via CLI.

        Args:
            text: Query text.
            game_id: Collection name.
            limit: Maximum number of results.

        Returns:
            List of query results.
        """
        try:
            # Escape quotes in query text
            escaped_text = text.replace('"', '\\"')
            command = [
                "qmd",
                "--index",
                self.index_name,
                "query",
                "--json",
                "-n",
                str(limit),
                "-c",
                game_id,
                escaped_text,
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=10.0,
            )

            data = json.loads(result.stdout)
            results = data.get("results", [])

            return [
                QmdQueryResult(
                    content=r.get("snippet") or r.get("content", ""),
                    score=r.get("score", 0.0),
                    file=r.get("file", ""),
                    docid=r.get("docid", ""),
                    metadata=r.get("metadata", {"source": r.get("file")}),
                )
                for r in results
            ]

        except subprocess.CalledProcessError as e:
            logger.error("QMD CLI query failed: %s", e.stderr, exc_info=True)
            raise
        except Exception as e:
            logger.error("QMD CLI query failed: %s", e, exc_info=True)
            raise
