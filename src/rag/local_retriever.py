"""Local QMD knowledge retriever."""

import logging

from .orchestrator import RetrievalResult
from .qmd_client import QmdMcpStdioClient

logger = logging.getLogger(__name__)


class LocalGameRetriever:
    """Retrieves knowledge from local QMD index."""

    def __init__(
        self,
        index_name: str = "game-companion",
    ) -> None:
        """Initialize the local retriever.

        Args:
            index_name: Name of the QMD index.
        """
        self.index_name = index_name
        self._client = QmdMcpStdioClient(index_name)
        logger.info(
            "LocalGameRetriever using QMD MCP stdio client with index: %s", index_name
        )

    def query(
        self, text: str, game_id: str, n_results: int = 5
    ) -> list[RetrievalResult]:
        """Query the local knowledge base.

        Args:
            text: Query text.
            game_id: Game identifier (collection name in QMD).
            n_results: Number of results to return.

        Returns:
            List of retrieval results.
        """
        try:
            qmd_results = self._client.query(text, game_id, limit=n_results)

            retrieval_results: list[RetrievalResult] = []
            for r in qmd_results:
                # QMD score is typically in [0, 1] range, use directly as confidence
                confidence = min(1.0, max(0.0, r.score))

                source_page = (
                    r.metadata.get("source_page") or r.metadata.get("source") or r.file
                )

                retrieval_results.append(
                    RetrievalResult(
                        content=r.content,
                        source=f"local-kb:{source_page}",
                        confidence=confidence,
                        metadata={
                            **r.metadata,
                            "docid": r.docid,
                            "file": r.file,
                        },
                    )
                )

            logger.debug(
                "QMD query returned %d results for '%s'", len(retrieval_results), text
            )
            return retrieval_results

        except Exception as e:
            logger.error("QMD retrieval failed for '%s': %s", text, e, exc_info=True)
            return []

    def shutdown(self) -> None:
        """Shutdown the QMD client."""
        if hasattr(self._client, "shutdown"):
            self._client.shutdown()
