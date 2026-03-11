"""Memory retriever for querying conversation history."""

import logging
from typing import Optional

from src.rag.orchestrator import RetrievalResult
from src.rag.qmd_client import QmdCliClient, QmdHttpClient

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """Retrieves memory documents from QMD storage.

    Memory documents are stored in a separate collection or with distinct
    metadata to avoid mixing with static game knowledge.
    """

    def __init__(
        self,
        qmd_url: Optional[str] = None,
        index_name: str = "game-companion",
    ) -> None:
        """Initialize the memory retriever.

        Args:
            qmd_url: QMD server URL (e.g., http://localhost:18788). If None, uses CLI.
            index_name: Name of the QMD index (used for CLI mode).
        """
        self.qmd_url = qmd_url
        self.index_name = index_name

        if qmd_url:
            self._client = QmdHttpClient(qmd_url)
            logger.info("MemoryRetriever using QMD HTTP client: %s", qmd_url)
        else:
            self._client = QmdCliClient(index_name)
            logger.info(
                "MemoryRetriever using QMD CLI client with index: %s", index_name
            )

    def query(
        self,
        text: str,
        game_id: str,
        n_results: int = 3,
    ) -> list[RetrievalResult]:
        """Query memory documents.

        Args:
            text: Query text.
            game_id: Game identifier.
            n_results: Number of results to return.

        Returns:
            List of retrieval results from memory.
        """
        collection_name = f"{game_id}-memory"

        try:
            qmd_results = self._client.query(text, collection_name, limit=n_results)

            retrieval_results: list[RetrievalResult] = []
            for r in qmd_results:
                confidence = min(1.0, max(0.0, r.score)) * 0.8

                session_id = r.metadata.get("session_id", "unknown")
                timestamp = r.metadata.get("timestamp", "")

                retrieval_results.append(
                    RetrievalResult(
                        content=r.content,
                        source=f"memory:{session_id}:{timestamp}",
                        confidence=confidence,
                        metadata={
                            **r.metadata,
                            "docid": r.docid,
                            "file": r.file,
                            "type": "memory",
                        },
                    )
                )

            logger.debug(
                "Memory query returned %d results for '%s'",
                len(retrieval_results),
                text,
            )
            return retrieval_results

        except Exception as e:
            logger.debug("Memory retrieval failed for '%s': %s", text, e)
            return []
