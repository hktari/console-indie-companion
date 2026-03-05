"""RAG retrieval orchestration for multi-source knowledge retrieval."""

import logging
from dataclasses import dataclass
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from a knowledge retriever."""

    content: str
    source: str
    confidence: float
    metadata: Optional[dict[str, Any]] = None


class KnowledgeRetriever(Protocol):
    """Protocol for knowledge retrievers."""

    def query(self, text: str, game_id: str) -> list[RetrievalResult]:
        """Query the knowledge source.

        Args:
            text: Query text.
            game_id: Game identifier for filtering/context.

        Returns:
            List of retrieval results.
        """
        ...


class KnowledgeOrchestrator:
    """Orchestrates multiple knowledge retrievers and merges results."""

    def __init__(self) -> None:
        """Initialize the orchestrator."""
        self._retrievers: list[KnowledgeRetriever] = []

    def register_retriever(self, retriever: KnowledgeRetriever) -> None:
        """Register a retriever.

        Args:
            retriever: Retriever instance implementing KnowledgeRetriever protocol.
        """
        self._retrievers.append(retriever)
        logger.info("Registered retriever: %s", type(retriever).__name__)

    def resolve(self, question: str, game_id: str) -> list[RetrievalResult]:
        """Query all retrievers and return merged, sorted results.

        Args:
            question: Query text.
            game_id: Game identifier.

        Returns:
            List of results sorted by confidence (descending).
        """
        all_results: list[RetrievalResult] = []

        for retriever in self._retrievers:
            try:
                results = retriever.query(question, game_id)
                all_results.extend(results)
            except Exception as e:
                logger.error(
                    "Retriever %s failed: %s", type(retriever).__name__, e, exc_info=True
                )

        # Sort by confidence descending
        all_results.sort(key=lambda r: r.confidence, reverse=True)
        return all_results
