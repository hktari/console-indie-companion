"""Request planner for routing user queries to appropriate tools."""

import logging
import os
from typing import Any, Callable, Optional

import openai

from src.agent.job_manager import ResearchJobManager
from src.agent.models import AgentDecision, EvidenceBundle, RouteType
from src.agent.tools import knowledge_base_search, memory_search
from src.utils.performance import get_performance_tracker

logger = logging.getLogger(__name__)


class RequestPlanner:
    """Lightweight planner that routes user requests to appropriate tools."""

    def __init__(
        self,
        job_manager: ResearchJobManager,
        qmd_url: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        api_key: Optional[str] = None,
        on_research_start: Optional[Callable[[], None]] = None,
    ):
        """Initialize the request planner.

        Args:
            job_manager: ResearchJobManager for async research (required)
            qmd_url: QMD server URL for retrieval
            model: OpenAI model for classification
            api_key: OpenAI API key
            on_research_start: Optional callback to notify user before research starts
        """
        self._qmd_url = qmd_url
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OpenAI API key required for planner")
        self._client = openai.OpenAI(api_key=self._api_key)
        self._perf = get_performance_tracker()
        self._on_research_start = on_research_start
        self._job_manager = job_manager

    def plan(
        self,
        transcript: str,
    ) -> AgentDecision:
        """Decide how to route the user request.

        Args:
            transcript: User's spoken input

        Returns:
            AgentDecision with routing information
        """
        logger.info("Planning route for transcript: '%s'", transcript[:100])
        # Simple heuristic-based routing for Phase 1
        # Phase 2 will add LLM-based classification if needed

        transcript_lower = transcript.lower()

        # Check for iterative research triggers
        if any(
            phrase in transcript_lower
            for phrase in [
                "look this up",
                "look up online",
                "research",
                "find information about",
                "deep dive",
            ]
        ):
            decision = AgentDecision(
                route=RouteType.ITERATIVE_RESEARCH,
                reasoning="Query requires iterative research with multiple sources",
                confidence=0.85,
                tools_to_call=["iterative_research"],
                metadata={"timeout": 120, "enable_progress": True},
            )
            logger.info(
                "Decision: %s (Reason: %s)", decision.route.value, decision.reasoning
            )
            return decision

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
            decision = AgentDecision(
                route=RouteType.MEMORY_SEARCH,
                reasoning="Query references past events or memory",
                confidence=0.8,
                tools_to_call=["memory_search"],
            )
            logger.info(
                "Decision: %s (Reason: %s)", decision.route.value, decision.reasoning
            )
            return decision

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
                decision = AgentDecision(
                    route=RouteType.COMBINED,
                    reasoning="Game-specific query - check KB first, web if needed",
                    confidence=0.7,
                    tools_to_call=["knowledge_base_search"],
                )
            else:
                decision = AgentDecision(
                    route=RouteType.WEB_RESEARCH,
                    reasoning="General knowledge query requiring web search",
                    confidence=0.7,
                    tools_to_call=["web_search"],
                )
            logger.info(
                "Decision: %s (Reason: %s)", decision.route.value, decision.reasoning
            )
            return decision

        # If user is asking about something visible or game-related
        if any(
            word in transcript_lower
            for word in ["what", "where", "how", "this", "that", "here"]
        ):
            decision = AgentDecision(
                route=RouteType.KNOWLEDGE_BASE,
                reasoning="Game context query - search knowledge base",
                confidence=0.75,
                tools_to_call=["knowledge_base_search"],
            )
            logger.info(
                "Decision: %s (Reason: %s)",
                decision.route.value,
                decision.reasoning,
            )
            return decision

        # Default: direct answer (no retrieval needed)
        decision = AgentDecision(
            route=RouteType.DIRECT_ANSWER,
            reasoning="Simple conversational response, no retrieval needed",
            confidence=0.6,
            tools_to_call=[],
        )
        logger.info(
            "Decision: %s (Reason: %s)", decision.route.value, decision.reasoning
        )
        return decision

    async def gather_evidence_async(
        self,
        decision: AgentDecision,
        query: str,
        game_id: str,
        transcript: str,
        scene: Optional[dict[str, Any]] = None,
        on_research_complete: Optional[Callable] = None,
    ) -> EvidenceBundle:
        """Execute planned tools with async research delegation.

        Args:
            decision: Routing decision from planner
            query: Query text for retrieval
            game_id: Game identifier
            transcript: Full user transcript
            scene: Optional scene context
            on_research_complete: Callback for research completion

        Returns:
            EvidenceBundle with sync results and job_id for async research
        """
        bundle = EvidenceBundle()

        for tool_name in decision.tools_to_call:
            try:
                logger.info("Executing tool: %s", tool_name)
                if tool_name == "knowledge_base_search":
                    with self._perf.measure(
                        "tool.knowledge_base_search", log_threshold=1.0
                    ):
                        results = knowledge_base_search.invoke(
                            {
                                "query": query,
                                "game_id": game_id,
                                "qmd_url": self._qmd_url,
                            }
                        )
                    bundle.kb_results = results[:3]
                    bundle.sources.append("knowledge_base")

                elif tool_name == "memory_search":
                    with self._perf.measure("tool.memory_search", log_threshold=1.0):
                        results = memory_search.invoke(
                            {
                                "query": query,
                                "game_id": game_id,
                                "qmd_url": self._qmd_url,
                            }
                        )
                    bundle.memory_results = results[:2]
                    bundle.sources.append("memory")

                elif tool_name == "web_search" or tool_name == "iterative_research":
                    # Queue async iterative research
                    logger.info("Queueing async iterative research job")
                    if self._on_research_start:
                        self._on_research_start()
                    job_id = await self._job_manager.queue_research(
                        query=query,
                        transcript=transcript,
                        game_id=game_id,
                        scene=scene,
                        on_complete=on_research_complete,
                    )
                    bundle.job_id = job_id
                    bundle.sources.append("iterative_research_pending")
                    logger.info("Research job queued: %s", job_id[:8])

            except Exception as e:
                logger.error("Tool %s failed: %s", tool_name, e, exc_info=True)

        bundle.metadata["decision"] = {
            "route": decision.route.value,
            "reasoning": decision.reasoning,
            "confidence": decision.confidence,
        }

        return bundle
