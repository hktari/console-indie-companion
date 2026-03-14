#!/usr/bin/env python3
"""
RAG Query Interface for Tunic Wiki

Query the QMD index to retrieve relevant wiki content.

Usage:
    python -m src.rag.query "how to beat the garden knight"
"""

import logging
import os
import sys
from typing import Optional

from src.rag.qmd_client import QmdMcpStdioClient
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def query_tunic_knowledge(
    question: str, category_filter: Optional[str] = None, n_results: int = 5
) -> list[str]:
    """
    Query the Tunic wiki knowledge base for relevant information.

    Args:
        question: User's question about Tunic
        category_filter: Optional metadata category to filter by (currently unused with QMD)
        n_results: Number of relevant chunks to return (default: 5)

    Returns:
        List of relevant text chunks
    """
    # Use QMD MCP stdio client
    index_name = os.environ.get("QMD_INDEX", "game-companion")
    client = QmdMcpStdioClient(index_name=index_name)
    logger.info("Using QMD MCP stdio client with index: %s", index_name)

    try:
        logger.info("Querying QMD: '%s'", question)
        results = client.query(question, game_id="tunic", limit=n_results)
    except Exception as e:
        raise RuntimeError(
            f"QMD query failed. Make sure QMD is installed and indexed.\nError: {e}"
        )

    if not results:
        return []

    formatted_results = []
    for r in results:
        source_page = (
            r.metadata.get("source_page") or r.metadata.get("source") or r.file
        )
        category = str(r.metadata.get("category", "general")).upper()
        formatted_results.append(f"[{category} | {source_page}]\n{r.content}")

    logger.debug("Found %d relevant passages", len(formatted_results))
    return formatted_results


def main():
    """CLI for querying Tunic knowledge."""
    # Setup basic logging for standalone execution
    setup_logging("INFO")

    if len(sys.argv) < 2:
        logger.error("Usage: python -m src.rag.query <question>")
        logger.error('Example: python -m src.rag.query "how to beat the garden knight"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    logger.info("=" * 60)
    logger.info("Question: %s", question)
    logger.info("=" * 60)

    try:
        results = query_tunic_knowledge(question)

        if not results:
            logger.info("No relevant results found.")
        else:
            logger.info("Found %d relevant passages:", len(results))

            for i, result in enumerate(results, 1):
                logger.info("--- Result %d ---", i)
                logger.info("\n%s\n", result)

    except Exception as e:
        logger.error("Error: %s", e)
        sys.exit(1)

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
