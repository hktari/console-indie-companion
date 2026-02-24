#!/usr/bin/env python3
"""
RAG Query Interface for Tunic Wiki

Query the ChromaDB collection to retrieve relevant wiki content.

Usage:
    python -m src.rag.query "how to beat the garden knight"
"""

import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings


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
    if len(sys.argv) < 2:
        print("Usage: python -m src.rag.query <question>")
        print('Example: python -m src.rag.query "how to beat the garden knight"')
        sys.exit(1)
    
    question = " ".join(sys.argv[1:])
    
    print("=" * 60)
    print(f"Question: {question}")
    print("=" * 60)
    
    try:
        results = query_tunic_knowledge(question)
        
        if not results:
            print("\nNo relevant results found.")
        else:
            print(f"\nFound {len(results)} relevant passages:\n")
            
            for i, result in enumerate(results, 1):
                print(f"\n--- Result {i} ---")
                print(result)
                print()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    
    print("=" * 60)


if __name__ == "__main__":
    main()
