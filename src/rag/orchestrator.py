"""RAG retrieval orchestration for multi-source knowledge retrieval."""

import logging
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from src.rag.qmd_pool import get_qmd_pool

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
        self._memory_retriever: Optional[KnowledgeRetriever] = None

    def register_retriever(self, retriever: KnowledgeRetriever) -> None:
        """Register a retriever.

        Args:
            retriever: Retriever instance implementing KnowledgeRetriever protocol.
        """
        self._retrievers.append(retriever)
        logger.info("Registered retriever: %s", type(retriever).__name__)

    def register_memory_retriever(self, retriever: KnowledgeRetriever) -> None:
        """Register a memory retriever (queried separately with lower priority).

        Args:
            retriever: Memory retriever instance.
        """
        self._memory_retriever = retriever
        logger.info("Registered memory retriever: %s", type(retriever).__name__)

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
                    "Retriever %s failed: %s",
                    type(retriever).__name__,
                    e,
                    exc_info=True,
                )

        if self._memory_retriever:
            try:
                memory_results = self._memory_retriever.query(question, game_id)
                all_results.extend(memory_results[:2])
            except Exception as e:
                logger.debug("Memory retriever failed: %s", e)

        # Sort by confidence descending
        all_results.sort(key=lambda r: r.confidence, reverse=True)
        return all_results

    def shutdown(self) -> None:
        """Shutdown all registered retrievers and the QMD connection pool."""
        for retriever in self._retrievers:
            if hasattr(retriever, "shutdown"):
                try:
                    retriever.shutdown()  # type: ignore[attr-defined]
                except Exception as e:
                    logger.warning(
                        "Error shutting down retriever %s: %s",
                        type(retriever).__name__,
                        e,
                    )

        if self._memory_retriever and hasattr(self._memory_retriever, "shutdown"):
            try:
                self._memory_retriever.shutdown()  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning("Error shutting down memory retriever: %s", e)

        # Shutdown the global QMD connection pool
        try:
            pool = get_qmd_pool()
            pool.shutdown()
        except Exception as e:
            logger.warning("Error shutting down QMD connection pool: %s", e)
