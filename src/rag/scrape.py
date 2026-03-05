#!/usr/bin/env python3
"""
Tunic Wiki Scraper (tunic.wiki)

Scrapes BookStack-based tunic.wiki using its markdown export feature.
Extracts structured metadata based on the Book.

Usage:
    python -m src.rag.scrape
"""

import logging
import re
import time
from pathlib import Path
from urllib.parse import urljoin
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "https://tunic.wiki"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "wiki"
DELAY_SECONDS = 3.0

# Books to scrape and their corresponding metadata categories
BOOKS = {
    "locations": "location",
    "items": "item",
    "creatures": "creature",
    "secrets": "secret",
    "faq": "general",
    "instruction-booklet": "mechanic",
    "speedrunning": "speedrun",
}


def get_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def clean_markdown(md_text: str) -> str:
    """Basic cleanup of markdown text (remove image links, messy tables)."""
    # Remove image links like [![alt](url)](url) or [![](url)](url)
    md_text = re.sub(r"\[!\[.*?\]\([^)]+\)\]\([^)]+\)", "", md_text)
    # Remove simple images ![](url)
    md_text = re.sub(r"!\[.*?\]\([^)]+\)", "", md_text)

    # Try to convert HTML tables to simple text formats
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(md_text, "html.parser")
    for table in soup.find_all("table"):
        table_text = []
        for row in table.find_all("tr"):
            row_data = []
            for cell in row.find_all(["td", "th"]):
                # Extract text, removing inner HTML tags like links
                cell_text = cell.get_text(separator=" ", strip=True)
                if cell_text and cell_text != "-":
                    row_data.append(cell_text)
            if row_data:
                table_text.append(" | ".join(row_data))

        # Replace the table in the soup with the text representation
        if table_text:
            text_node = soup.new_string("\n" + "\n".join(table_text) + "\n")
            table.replace_with(text_node)
        else:
            table.decompose()

    # Also strip any other stray HTML tags that bs4 finds
    md_text = soup.get_text(separator="\n")

    # Clean up empty lines and trailing spaces
    md_text = re.sub(r"\n\s*\n", "\n\n", md_text)
    return md_text.strip()


def parse_book_markdown(md_text: str, book_slug: str, category: str) -> list[dict]:
    """Parse the giant markdown file into individual pages/sections."""
    pages = []

    # BookStack markdown export concatenates pages with `# Page Title`
    # We split by `# ` (level 1 heading)
    parts = re.split(r"^# ", md_text, flags=re.MULTILINE)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        lines = part.split("\n")
        title = lines[0].strip()
        content = "\n".join(lines[1:]).strip()

        # Skip empty content or just the book title
        if not content or title.lower() == book_slug.lower():
            continue

        pages.append(
            {
                "title": title,
                "url": f"{BASE_URL}/books/{book_slug}/page/{title.lower().replace(' ', '-')}",
                "page_id": f"{book_slug}_{title.lower().replace(' ', '_')}",
                "metadata": {"category": category, "book": book_slug},
                "sections": [{"header": "Content", "content": clean_markdown(content)}],
            }
        )

    return pages


def scrape_book(book_slug: str, category: str, session: requests.Session) -> list[dict]:
    """Download a book's markdown export and parse it."""
    url = f"{BASE_URL}/books/{book_slug}/export/markdown"
    logger.info("Scraping book: %s from %s", book_slug, url)

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()

        md_text = response.text
        pages = parse_book_markdown(md_text, book_slug, category)
        logger.info("  ✓ Extracted %d pages from %s", len(pages), book_slug)
        return pages

    except Exception as e:
        logger.error("  ✗ Error scraping book %s: %s", book_slug, e)
        return []


def main():
    setup_logging("INFO")
    logger.info("=" * 60)
    logger.info("Tunic Wiki Scraper (tunic.wiki)")
    logger.info("=" * 60)

    # Clear old Fandom wiki data
    if DATA_DIR.exists():
        import shutil

        logger.info("Clearing old Fandom wiki data...")
        shutil.rmtree(DATA_DIR)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    session = get_session()

    for book_slug, category in BOOKS.items():
        url = f"{BASE_URL}/books/{book_slug}/export/markdown"
        logger.info("Scraping book: %s from %s", book_slug, url)

        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()

            md_text = clean_markdown(response.text)

            # Save the entire book as a single markdown file
            filename = f"{book_slug}.md"
            filepath = DATA_DIR / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# {book_slug.capitalize()}\n\n")
                f.write(md_text)

            logger.info("  ✓ Saved %s", filename)
            total_pages += 1  # Counting books as "pages" for simplicity in log

        except Exception as e:
            logger.error("  ✗ Error scraping book %s: %s", book_slug, e)

        time.sleep(DELAY_SECONDS)

    logger.info("=" * 60)
    logger.info("Scraping complete! Saved %d pages.", total_pages)
    logger.info("Data saved to: %s", DATA_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
