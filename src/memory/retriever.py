"""Memory retriever for querying conversation history."""

import logging

from src.rag.orchestrator import RetrievalResult
from src.rag.qmd_client import QmdMcpStdioClient

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """Retrieves memory documents from QMD storage.

    Memory documents are stored in a separate collection or with distinct
    metadata to avoid mixing with static game knowledge.
    """

    def __init__(
        self,
        index_name: str = "game-companion",
    ) -> None:
        """Initialize the memory retriever.

        Args:
            index_name: Name of the QMD index.
        """
        self.index_name = index_name
        self._client = QmdMcpStdioClient(index_name)
        logger.info(
            "MemoryRetriever using QMD MCP stdio client with index: %s", index_name
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
