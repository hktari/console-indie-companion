"""Tool wrappers for planner-controlled retrieval."""

import logging
from typing import Any, Optional

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


def visual_analysis(
    frame_provider: Any, scene_analyzer: Any
) -> Optional[dict[str, Any]]:
    """Analyze current game screenshot using VLM.

    NOTE: This tool is not yet wired to the planner. It exists as a placeholder
    for future tool-calling functionality where the agent can request visual
    analysis on-demand instead of running it upfront.

    Args:
        frame_provider: Provider for game screenshots
        scene_analyzer: VLM analyzer instance

    Returns:
        Scene analysis dict with location, activity, enemies, etc., or None on error
    """
    try:
        frame = frame_provider.capture_once()
        if frame is None:
            frame = frame_provider.get_latest_frame()

        if frame is None:
            logger.warning("No frame available for visual analysis")
            return None

        scene = scene_analyzer.analyze_screenshot(frame, "image/jpeg")
        if scene and isinstance(scene, dict) and "error" not in scene:
            logger.info(
                "Visual analysis completed: %s", scene.get("description", "")[:50]
            )
            return scene
        else:
            logger.warning("Visual analysis returned error or invalid result")
            return None
    except Exception as e:
        logger.error("Visual analysis failed: %s", e, exc_info=True)
        return None
