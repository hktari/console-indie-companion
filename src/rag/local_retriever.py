"""Local QMD knowledge retriever."""

import logging
from typing import Optional

from .orchestrator import RetrievalResult
from .qmd_client import QmdCliClient, QmdHttpClient

logger = logging.getLogger(__name__)


class LocalGameRetriever:
    """Retrieves knowledge from local QMD index."""

    def __init__(
        self,
        qmd_url: Optional[str] = None,
        index_name: str = "game-companion",
    ) -> None:
        """Initialize the local retriever.

        Args:
            qmd_url: QMD server URL (e.g., http://localhost:18788). If None, uses CLI.
            index_name: Name of the QMD index (used for CLI mode).
        """
        self.qmd_url = qmd_url
        self.index_name = index_name

        # Initialize client based on configuration
        if qmd_url:
            self._client = QmdHttpClient(qmd_url)
            logger.info("LocalGameRetriever using QMD HTTP client: %s", qmd_url)
        else:
            self._client = QmdCliClient(index_name)
            logger.info(
                "LocalGameRetriever using QMD CLI client with index: %s", index_name
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
