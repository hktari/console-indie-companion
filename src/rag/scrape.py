#!/usr/bin/env python3
"""
Tunic Wiki Scraper

Scrapes content from the Tunic Fandom wiki and saves it as JSON files.
Target: https://tunic.fandom.com/wiki/

Usage:
    python -m src.rag.scrape
"""

import logging
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# Configuration
BASE_URL = "https://tunic.fandom.com"
WIKI_BASE = f"{BASE_URL}/wiki/"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "wiki"
DELAY_SECONDS = 0.5

# Pages to skip (non-content)
SKIP_PREFIXES = [
    "User:",
    "Talk:",
    "Special:",
    "File:",
    "Template:",
    "Category:",
    "Help:",
    "Forum:",
    "MediaWiki:",
    "Thread:",
    "Message_Wall:",
    "User_blog:",
    "Blog:",
]


def is_valid_content_page(title: str) -> bool:
    """Check if a page title represents valid content."""
    return not any(title.startswith(prefix) for prefix in SKIP_PREFIXES)


def extract_page_content(soup: BeautifulSoup) -> list[dict]:
    """Extract structured content from a wiki page."""
    sections = []
    
    # Find the main content area
    content_div = soup.find("div", {"class": "mw-parser-output"})
    if not content_div:
        return sections
    
    current_section = {"header": "Introduction", "content": []}
    
    for element in content_div.children:
        if not hasattr(element, 'name'):
            continue
            
        # Section headers
        if element.name in ['h2', 'h3', 'h4']:
            # Save previous section if it has content
            if current_section["content"]:
                sections.append({
                    "header": current_section["header"],
                    "content": "\n".join(current_section["content"]).strip()
                })
            
            # Start new section
            header_text = element.get_text(strip=True)
            # Remove [edit] links
            header_text = re.sub(r'\[edit\]', '', header_text).strip()
            current_section = {"header": header_text, "content": []}
        
        # Paragraphs and lists
        elif element.name in ['p', 'ul', 'ol', 'dl']:
            text = element.get_text(separator=' ', strip=True)
            if text:
                current_section["content"].append(text)
    
    # Add final section
    if current_section["content"]:
        sections.append({
            "header": current_section["header"],
            "content": "\n".join(current_section["content"]).strip()
        })
    
    return sections


def fetch_all_pages() -> list[str]:
    """Fetch all wiki page URLs from Special:AllPages."""
    logger.info("Fetching page list from Special:AllPages...")
    all_pages = []
    
    # Fandom wikis use Special:AllPages with pagination
    url = f"{BASE_URL}/wiki/Special:AllPages"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Find all links in the allpages list
        allpages_content = soup.find("div", {"class": "mw-allpages-body"}) or soup.find("ul", {"class": "mw-allpages-chunk"})
        
        if allpages_content:
            links = allpages_content.find_all("a")
            for link in links:
                href = link.get("href")
                if href and href.startswith("/wiki/"):
                    page_title = href.replace("/wiki/", "")
                    if is_valid_content_page(page_title):
                        all_pages.append(href)
        
        # Also crawl from main page to find more pages
        main_response = requests.get(f"{BASE_URL}/wiki/Tunic_Wiki", timeout=10)
        main_soup = BeautifulSoup(main_response.content, "html.parser")
        content_links = main_soup.find_all("a", href=re.compile(r"^/wiki/[^:]+$"))
        
        for link in content_links:
            href = link.get("href")
            if href and href not in all_pages:
                page_title = href.replace("/wiki/", "")
                if is_valid_content_page(page_title):
                    all_pages.append(href)
        
    except Exception as e:
        logger.error("Error fetching page list: %s", e)
        # Fallback: use a seed list of important pages
        all_pages = [
            "/wiki/Tunic_Wiki",
            "/wiki/Bosses",
            "/wiki/Enemies",
            "/wiki/Items",
            "/wiki/Locations",
            "/wiki/Characters",
            "/wiki/Gameplay",
            "/wiki/Story",
        ]
    
    # Remove duplicates
    all_pages = list(set(all_pages))
    logger.info("Found %d pages to scrape", len(all_pages))
    
    return all_pages


def scrape_page(page_url: str) -> dict | None:
    """Scrape a single wiki page."""
    full_url = urljoin(BASE_URL, page_url)
    page_title = page_url.replace("/wiki/", "")
    
    logger.info("Scraping: %s", page_title)
    
    try:
        response = requests.get(full_url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Get the page title
        title_element = soup.find("h1", {"class": "page-header__title"}) or soup.find("h1", {"id": "firstHeading"})
        title = title_element.get_text(strip=True) if title_element else page_title
        
        # Extract content sections
        sections = extract_page_content(soup)
        
        if not sections:
            logger.warning("  ⚠ No content found for %s", page_title)
            return None
        
        return {
            "title": title,
            "url": full_url,
            "page_id": page_title,
            "sections": sections
        }
        
    except Exception as e:
        logger.error("  ✗ Error scraping %s: %s", page_title, e)
        return None


def save_page(page_data: dict):
    """Save page data as JSON."""
    # Create a safe filename from page_id
    filename = re.sub(r'[^\w\-_]', '_', page_data["page_id"])
    filepath = DATA_DIR / f"{filename}.json"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(page_data, f, ensure_ascii=False, indent=2)


def main():
    """Main scraper function."""
    # Setup basic logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logger.info("=" * 60)
    logger.info("Tunic Wiki Scraper")
    logger.info("=" * 60)
    
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Fetch all pages
    pages = fetch_all_pages()
    
    # Scrape each page
    scraped_count = 0
    for i, page_url in enumerate(pages, 1):
        page_data = scrape_page(page_url)
        
        if page_data:
            save_page(page_data)
            scraped_count += 1
            logger.info("  ✓ Saved [%d/%d]", scraped_count, len(pages))
        
        # Be respectful: delay between requests
        if i < len(pages):
            time.sleep(DELAY_SECONDS)
    
    logger.info("=" * 60)
    logger.info("Scraping complete! Scraped %d pages.", scraped_count)
    logger.info("Data saved to: %s", DATA_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
