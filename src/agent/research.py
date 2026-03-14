"""Research subagent for isolated web search and knowledge synthesis."""

import logging
import os
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agent.tools import knowledge_base_search, web_search

logger = logging.getLogger(__name__)

# TODO: use this example if issues with langchain implementation: https://eu.smith.langchain.com/o/33684ea1-a5d9-41fd-9c0c-6cf051178c16/projects/p/653180ad-ce95-4ff0-9bb6-61523e43344c?onboarding=game-companion&timeModel=%7B%22duration%22%3A%221d%22%7D


class ResearchSubagent:
    """Isolated research subagent that searches and synthesizes findings into a compact memo."""

    def __init__(
        self,
        qmd_url: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        api_key: Optional[str] = None,
    ):
        """Initialize the research subagent.

        Args:
            qmd_url: QMD server URL for knowledge base access
            model: OpenAI model for synthesis
            api_key: OpenAI API key
        """
        self._qmd_url = qmd_url
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OpenAI API key required for research subagent")

        # Use LangChain for cleaner message handling
        self._llm = ChatOpenAI(
            model=model,
            api_key=self._api_key,  # type: ignore
            temperature=0.3,
        )

    def research(self, query: str, game_id: str, use_web: bool = True) -> str:
        """Conduct research and return a compact memo.

        Args:
            query: Research query
            game_id: Game identifier
            use_web: Whether to include web search (default: True)

        Returns:
            Compact research memo with key findings and sources
        """
        logger.info("Research subagent starting for query: %s", query[:100])

        # Step 1: Gather raw search results
        kb_results = []
        web_results = []

        try:
            kb_results = knowledge_base_search.invoke(
                {"query": query, "game_id": game_id, "qmd_url": self._qmd_url}
            )
            logger.info("KB search returned %d results", len(kb_results))
        except Exception as e:
            logger.error("KB search failed in research subagent: %s", e)

        if use_web:
            try:
                web_results = web_search.invoke({"query": query, "game_id": game_id})
                logger.info("Web search returned %d results", len(web_results))
            except Exception as e:
                logger.error("Web search failed in research subagent: %s", e)

        # Step 2: Synthesize findings into a compact memo
        if not kb_results and not web_results:
            return "No relevant information found."

        memo = self._synthesize_memo(query, kb_results[:3], web_results[:2])
        logger.info("Research memo generated (%d chars)", len(memo))
        return memo

    def _synthesize_memo(
        self,
        query: str,
        kb_results: list,
        web_results: list,
    ) -> str:
        """Synthesize search results into a compact research memo.

        Args:
            query: Original research query
            kb_results: Knowledge base results
            web_results: Web search results

        Returns:
            Compact research memo
        """
        # Build context from results
        context_parts = []

        if kb_results:
            kb_text = "\n\n".join(
                f"[KB: {r.source}]\n{r.content[:300]}" for r in kb_results
            )
            context_parts.append(f"Game Knowledge Base:\n{kb_text}")

        if web_results:
            web_text = "\n\n".join(
                f"[Web: {r.source}]\n{r.content[:300]}" for r in web_results
            )
            context_parts.append(f"Web Sources:\n{web_text}")

        context = "\n\n---\n\n".join(context_parts)

        # Synthesize using LLM
        system_prompt = """You are a research assistant that synthesizes information into compact, actionable memos.

Your task:
1. Extract the most relevant facts that answer the user's question
2. Cite sources clearly (use [KB] for knowledge base, [Web] for web sources)
3. Keep the memo under 200 words
4. Focus on actionable information
5. Note any uncertainty or conflicting information

Format:
- Start with a direct answer if possible
- List key findings as bullet points
- End with sources cited"""

        user_prompt = f"""Research query: {query}

Available information:
{context}

Synthesize this into a compact research memo that answers the query."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            logger.info("Synthesizing research memo for query: %s", query[:100])
            response = self._llm.invoke(messages)
            content = response.content
            if isinstance(content, str):
                memo = content.strip()
            else:
                memo = str(content).strip()
            logger.info("Research memo synthesized (%d chars)", len(memo))
            return memo
        except Exception as e:
            logger.error("Memo synthesis failed: %s", e, exc_info=True)
            # Fallback: return raw results
            fallback = []
            if kb_results:
                fallback.append(f"KB: {kb_results[0].content[:150]}")
            if web_results:
                fallback.append(f"Web: {web_results[0].content[:150]}")
            return "\n\n".join(fallback) if fallback else "Research synthesis failed."
