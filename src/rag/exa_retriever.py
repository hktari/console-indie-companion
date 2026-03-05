"""Exa AI web search retriever for fallback knowledge retrieval."""

import logging
import os
from typing import Optional

from .orchestrator import RetrievalResult

logger = logging.getLogger(__name__)


class ExaRetriever:
    """Retrieves knowledge from Exa AI web search."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        """Initialize the Exa retriever.

        Args:
            api_key: Exa API key (defaults to EXA_API_KEY env var).
            openai_api_key: OpenAI API key for query enhancement (optional).
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or os.environ.get("EXA_API_KEY")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout

        if not self.api_key:
            logger.warning(
                "EXA_API_KEY not set. Exa retriever will return empty results."
            )

    def query(self, text: str, game_id: str, n_results: int = 5) -> list[RetrievalResult]:
        """Query Exa AI for web search results.

        Args:
            text: Query text.
            game_id: Game identifier for context.
            n_results: Number of results to return.

        Returns:
            List of retrieval results.
        """
        if not self.api_key:
            return []

        search_query = text

        # Enhance query with LLM if OpenAI key is available
        if self.openai_api_key:
            try:
                import openai

                client = openai.OpenAI(api_key=self.openai_api_key)
                prompt = f"""You are an expert search query optimizer for the game '{game_id}'.
User Query: {text}

Task: Create a single, highly effective search query that combines the game name '{game_id}' and the user's intent. Respond ONLY with the optimized search query text."""

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.0,
                    max_tokens=100,
                    timeout=5.0,
                )

                enhanced = response.choices[0].message.content
                if enhanced:
                    enhanced = enhanced.strip().strip('"')
                    logger.info("Enhanced search query: '%s'", enhanced)
                    search_query = enhanced

            except Exception as e:
                logger.warning(
                    "Failed to enhance query with LLM: %s. Using original.", e
                )

        # Perform Exa search
        try:
            import requests

            response = requests.post(
                "https://api.exa.ai/search",
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "query": search_query,
                    "numResults": n_results,
                    "useAutoprompt": True,
                    "contents": {"text": True},
                },
                timeout=self.timeout,
            )

            if not response.ok:
                logger.error(
                    "Exa API error: %s %s", response.status_code, response.text
                )
                return []

            data = response.json()
            results = data.get("results", [])

            retrieval_results: list[RetrievalResult] = []
            for r in results:
                title = r.get("title", "")
                url = r.get("url", "")
                text_content = r.get("text") or r.get("contents", {}).get("text", "")

                content = f"Title: {title}\nURL: {url}\n\n{text_content}"

                retrieval_results.append(
                    RetrievalResult(
                        content=content,
                        source=f"Exa: {url}",
                        confidence=0.8,  # Default confidence for web search
                        metadata={
                            "title": title,
                            "url": url,
                            "id": r.get("id"),
                        },
                    )
                )

            logger.info("Exa search returned %d results", len(retrieval_results))
            return retrieval_results

        except Exception as e:
            logger.error("Exa search failed: %s", e, exc_info=True)
            return []
