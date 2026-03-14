"""Tool wrappers for planner-controlled retrieval."""

import logging
from typing import Optional

from langchain_core.tools import tool

from src.memory.retriever import MemoryRetriever
from src.rag import ExaRetriever, LocalGameRetriever
from src.rag.orchestrator import RetrievalResult

logger = logging.getLogger(__name__)


@tool
def knowledge_base_search(
    query: str, game_id: str, qmd_url: Optional[str] = None
) -> list[RetrievalResult]:
    """Search the local game knowledge base (QMD collection).

    Args:
        query: Search query text
        game_id: Game identifier (e.g., 'tunic')
        qmd_url: Optional QMD server URL

    Returns:
        List of retrieval results from the knowledge base
    """
    try:
        retriever = LocalGameRetriever()
        results = retriever.query(query, game_id)
        logger.info(
            "KB search returned %d results for query: %s", len(results), query[:50]
        )
        return results
    except Exception as e:
        logger.error("Knowledge base search failed: %s", e, exc_info=True)
        return []


@tool
def memory_search(query: str, game_id: str) -> list[RetrievalResult]:
    """Search conversation memory for past interactions and discoveries.

    Args:
        query: Search query text
        game_id: Game identifier (e.g., 'tunic')

    Returns:
        List of retrieval results from memory
    """
    try:
        retriever = MemoryRetriever()
        results = retriever.query(query, game_id)
        logger.info(
            "Memory search returned %d results for query: %s", len(results), query[:50]
        )
        return results[:2]
    except Exception as e:
        logger.debug("Memory search failed: %s", e)
        return []


@tool
def web_search(query: str, game_id: str) -> list[RetrievalResult]:
    """Search the web using Exa for external information.

    Args:
        query: Search query text
        game_id: Game identifier (e.g., 'tunic')

    Returns:
        List of retrieval results from web search
    """
    try:
        retriever = ExaRetriever()
        results = retriever.query(query, game_id)
        logger.info(
            "Web search returned %d results for query: %s", len(results), query[:50]
        )
        return results
    except Exception as e:
        logger.error("Web search failed: %s", e, exc_info=True)
        return []
