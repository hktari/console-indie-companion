#!/usr/bin/env python3
"""
RAG Query Interface for Tunic Wiki

Query the ChromaDB collection to retrieve relevant wiki content.

Usage:
    python -m src.rag.query "how to beat the garden knight"
"""

import logging
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


CHROMA_DIR = Path(__file__).parent.parent.parent / "data" / "chroma"
COLLECTION_NAME = "tunic_wiki"


def query_tunic_knowledge(question: str, n_results: int = 5) -> list[str]:
    """
    Query the Tunic wiki knowledge base for relevant information.
    
    Args:
        question: User's question about Tunic
        n_results: Number of relevant chunks to return (default: 5)
    
    Returns:
        List of relevant text chunks
    """
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found: {CHROMA_DIR}\n"
            f"Please run 'python -m src.rag.index' first."
        )
    
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' not found.\n"
            f"Please run 'python -m src.rag.index' first.\n"
            f"Error: {e}"
        )
    
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    
    formatted_results = []
    for doc, meta in zip(documents, metadatas):
        source_page = meta.get("source_page", "Unknown")
        section = meta.get("section_header", "Unknown")
        formatted_results.append(f"[{source_page} > {section}]\n{doc}")
    
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
