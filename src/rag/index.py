#!/usr/bin/env python3
"""
RAG Indexer for Tunic Wiki

Reads scraped wiki JSON files, chunks content, generates embeddings,
and stores in ChromaDB for semantic search.

Usage:
    python -m src.rag.index
"""

import logging
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


DATA_DIR = Path(__file__).parent.parent.parent / "data" / "wiki"
CHROMA_DIR = Path(__file__).parent.parent.parent / "data" / "chroma"
COLLECTION_NAME = "tunic_wiki"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_wiki_pages() -> list[dict]:
    """Load all scraped wiki pages from JSON files."""
    pages = []
    
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Wiki data directory not found: {DATA_DIR}")
    
    json_files = list(DATA_DIR.glob("*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {DATA_DIR}")
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            pages.append(json.load(f))
    
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Chunk text into fixed-size pieces with overlap."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start = end - overlap
    
    return chunks


def create_chunks_from_pages(pages: list[dict]) -> tuple[list[str], list[dict], list[str]]:
    """
    Create chunks from wiki pages with metadata.
    
    Returns:
        tuple: (chunks, metadatas, ids)
    """
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    chunk_id_counter = 0
    
    for page in pages:
        page_title = page["title"]
        page_id = page["page_id"]
        page_url = page["url"]
        
        for section in page["sections"]:
            section_header = section["header"]
            section_content = section["content"]
            
            if not section_content.strip():
                continue
            
            chunks = chunk_text(section_content)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source_page": page_title,
                    "page_id": page_id,
                    "page_url": page_url,
                    "section_header": section_header,
                    "chunk_index": i
                })
                all_ids.append(f"chunk_{chunk_id_counter}")
                chunk_id_counter += 1
    
    return all_chunks, all_metadatas, all_ids


def index_to_chromadb(chunks: list[str], metadatas: list[dict], ids: list[str]):
    """Index chunks into ChromaDB."""
    
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    
    try:
        client.delete_collection(name=COLLECTION_NAME)
        logger.info("Deleted existing collection: %s", COLLECTION_NAME)
    except:
        pass
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Tunic Wiki knowledge base"}
    )
    
    batch_size = 100
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        collection.add(
            documents=batch_chunks,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
        
        batch_num = (i // batch_size) + 1
        logger.info("  Indexed batch %d/%d (%d chunks)", batch_num, total_batches, len(batch_chunks))
    
    return collection


def main():
    """Main indexer function."""
    # Setup basic logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logger.info("=" * 60)
    logger.info("Tunic Wiki RAG Indexer")
    logger.info("=" * 60)
    
    logger.info("1. Loading wiki pages...")
    pages = load_wiki_pages()
    logger.info("   Loaded %d pages", len(pages))
    
    logger.info("2. Chunking content...")
    chunks, metadatas, ids = create_chunks_from_pages(pages)
    logger.info("   Created %d chunks", len(chunks))
    
    logger.info("3. Indexing to ChromaDB...")
    collection = index_to_chromadb(chunks, metadatas, ids)
    logger.info("   ✓ Indexed to collection: %s", COLLECTION_NAME)
    
    logger.info("=" * 60)
    logger.info("Indexing complete!")
    logger.info("  Pages: %d", len(pages))
    logger.info("  Chunks: %d", len(chunks))
    logger.info("  Collection: %s", COLLECTION_NAME)
    logger.info("  Database: %s", CHROMA_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
