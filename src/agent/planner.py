"""Request planner for routing user queries to appropriate tools."""

import logging
import os
from typing import Any, Optional

import openai

from src.agent.models import AgentDecision, EvidenceBundle, RouteType
from src.agent.research import ResearchSubagent
from src.agent.tools import knowledge_base_search, memory_search

logger = logging.getLogger(__name__)


class RequestPlanner:
    """Lightweight planner that routes user requests to appropriate tools."""

    def __init__(
        self,
        qmd_url: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        api_key: Optional[str] = None,
    ):
        """Initialize the request planner.

        Args:
            qmd_url: QMD server URL for retrieval
            model: OpenAI model for classification
            api_key: OpenAI API key
        """
        self._qmd_url = qmd_url
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OpenAI API key required for planner")
        self._client = openai.OpenAI(api_key=self._api_key)

        # Initialize research subagent for web research delegation
        self._research_subagent = ResearchSubagent(
            qmd_url=qmd_url,
            model=model,
            api_key=self._api_key,
        )

    def plan(
        self,
        transcript: str,
        scene: Optional[dict[str, Any]],
        narrative: str,
    ) -> AgentDecision:
        """Decide how to route the user request.

        Args:
            transcript: User's spoken input
            scene: Current scene analysis
            narrative: Current narrative context

        Returns:
            AgentDecision with routing information
        """
        # Simple heuristic-based routing for Phase 1
        # Phase 2 will add LLM-based classification if needed

        transcript_lower = transcript.lower()

        # Check for memory-related queries
        if any(
            phrase in transcript_lower
            for phrase in [
                "earlier",
                "before",
                "remember",
                "what did i",
                "where was",
                "when did",
                "last time",
                "previously",
            ]
        ):
            return AgentDecision(
                route=RouteType.MEMORY_SEARCH,
                reasoning="Query references past events or memory",
                confidence=0.8,
                tools_to_call=["memory_search"],
            )

        # Check for web research indicators
        if any(
            phrase in transcript_lower
            for phrase in [
                "look up",
                "search for",
                "find out",
                "what is",
                "who is",
                "how do i",
                "where can i find",
                "wiki",
                "guide",
            ]
        ):
            # Only use web search if it seems like external knowledge is needed
            # For game-specific queries, prefer KB first
            if any(
                game_term in transcript_lower
                for game_term in ["tunic", "boss", "enemy", "item", "puzzle", "secret"]
            ):
                return AgentDecision(
                    route=RouteType.COMBINED,
                    reasoning="Game-specific query - check KB first, web if needed",
                    confidence=0.7,
                    tools_to_call=["knowledge_base_search"],
                )
            else:
                return AgentDecision(
                    route=RouteType.WEB_RESEARCH,
                    reasoning="General knowledge query requiring web search",
                    confidence=0.7,
                    tools_to_call=["web_search"],
                )

        # Check for knowledge base queries (game-specific)
        if scene and any(
            key in scene for key in ["location", "activity", "notable_items", "enemies"]
        ):
            # If user is asking about something visible or game-related
            if any(
                word in transcript_lower
                for word in ["what", "where", "how", "this", "that", "here"]
            ):
                return AgentDecision(
                    route=RouteType.KNOWLEDGE_BASE,
                    reasoning="Game context query - search knowledge base",
                    confidence=0.75,
                    tools_to_call=["knowledge_base_search"],
                )

        # Default: direct answer (no retrieval needed)
        return AgentDecision(
            route=RouteType.DIRECT_ANSWER,
            reasoning="Simple conversational response, no retrieval needed",
            confidence=0.6,
            tools_to_call=[],
        )

    def gather_evidence(
        self,
        decision: AgentDecision,
        query: str,
        game_id: str,
    ) -> EvidenceBundle:
        """Execute the planned tools and gather evidence.

        Args:
            decision: Routing decision from planner
            query: Query text for retrieval
            game_id: Game identifier

        Returns:
            EvidenceBundle with collected results
        """
        bundle = EvidenceBundle()

        for tool_name in decision.tools_to_call:
            try:
                if tool_name == "knowledge_base_search":
                    results = knowledge_base_search.invoke(
                        {"query": query, "game_id": game_id, "qmd_url": self._qmd_url}
                    )
                    bundle.kb_results = results[:3]
                    bundle.sources.append("knowledge_base")

                elif tool_name == "memory_search":
                    results = memory_search.invoke(
                        {"query": query, "game_id": game_id, "qmd_url": self._qmd_url}
                    )
                    bundle.memory_results = results[:2]
                    bundle.sources.append("memory")

                elif tool_name == "web_search":
                    # Delegate to research subagent for isolated web research
                    logger.info("Delegating to research subagent for web research")
                    research_memo = self._research_subagent.research(
                        query=query,
                        game_id=game_id,
                        use_web=True,
                    )
                    bundle.research_memo = research_memo
                    bundle.sources.append("web_research")

            except Exception as e:
                logger.error("Tool %s failed: %s", tool_name, e, exc_info=True)

        bundle.metadata["decision"] = {
            "route": decision.route.value,
            "reasoning": decision.reasoning,
            "confidence": decision.confidence,
        }

        return bundle
