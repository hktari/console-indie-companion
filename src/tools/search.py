import os
import requests
import logging
from dotenv import load_dotenv


load_dotenv()
logger = logging.getLogger(__name__)


def exa_search(query: str):
    """Performs a web search using the Exa API."""
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        raise ValueError("EXA_API_KEY environment variable not set.")

    url = "https://api.exa.ai/search"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "num_results": 5,
        "use_autoprompt": True,
        "contents": {"text": True},
    }

    try:
        logger.debug(f"Exa search query: {query}")
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred: {e}")
        return None
