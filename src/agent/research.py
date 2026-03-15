"""Iterative research subagent with self-directed reasoning loop."""

import json
import logging
import os
from typing import Callable, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agent.models import ResearchConfig, ResearchResult, ToolResult
from src.agent.tools import knowledge_base_search, memory_search, web_search

logger = logging.getLogger(__name__)


class ResearchSubagent:
    """Iterative research agent with self-directed reasoning loop."""

    def __init__(
        self,
        qmd_url: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        api_key: Optional[str] = None,
        config: Optional[ResearchConfig] = None,
    ):
        """Initialize the research subagent.

        Args:
            qmd_url: QMD server URL for knowledge base access
            model: OpenAI model for synthesis
            api_key: OpenAI API key
            config: Research configuration
        """
        self._qmd_url = qmd_url
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OpenAI API key required for research subagent")
        self._config = config or ResearchConfig()

        # Reasoning model for planning and assessment
        self._reasoning_llm = ChatOpenAI(
            model="gpt-5-mini",
            reasoning={"effort": "low"},
            api_key=self._api_key,  # type: ignore
            temperature=0.3,
        )

        # Synthesis model for final memo
        self._synthesis_llm = ChatOpenAI(
            model="gpt-4.1",
            api_key=self._api_key,  # type: ignore
            temperature=0.3,
        )

    async def research_iterative(
        self,
        query: str,
        game_id: str,
        max_tool_calls: Optional[int] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> ResearchResult:
        """Conduct iterative research with self-directed reasoning loop.

        Args:
            query: Research query
            game_id: Game identifier
            max_tool_calls: Maximum tool calls (overrides config)
            on_progress: Progress callback (step, max_steps)

        Returns:
            ResearchResult with memo, findings, and metadata
        """
        max_calls = max_tool_calls or self._config.max_tool_calls
        logger.info(
            "Starting iterative research for query: %s (max_calls=%d)",
            query[:100],
            max_calls,
        )

        findings: list[ToolResult] = []
        sources_used: list[str] = []

        for step in range(max_calls):
            logger.info("Research step %d/%d", step + 1, max_calls)

            # Plan next action
            action = await self._plan_next_action(query, findings)
            logger.info("Planned action: %s", action.get("action_type"))

            if action["action_type"] == "complete":
                logger.info("Agent decided research is complete")
                break

            # Execute tool
            result = await self._execute_tool(action, game_id)
            if result:
                findings.append(result)
                if result.tool_name not in sources_used:
                    sources_used.append(result.tool_name)

            # Progress callback
            if on_progress:
                on_progress(step + 1, max_calls)

            # Assess completeness after minimum steps
            if step >= self._config.min_tool_calls - 1:
                is_complete = await self._assess_completeness(query, findings)
                if is_complete:
                    logger.info("Self-assessment: research complete")
                    break

        # Synthesize final memo
        memo = await self._synthesize_findings(query, findings)
        confidence = await self._calculate_confidence(query, findings)

        result = ResearchResult(
            memo=memo,
            findings=findings,
            steps_taken=len(findings),
            confidence=confidence,
            sources_used=sources_used,
        )

        logger.info(
            "Research complete: %d steps, %d sources, confidence=%.2f",
            result.steps_taken,
            len(result.sources_used),
            result.confidence,
        )
        return result

    async def research(self, query: str, game_id: str, use_web: bool = True) -> str:
        """Legacy sync research method for backward compatibility.

        Args:
            query: Research query
            game_id: Game identifier
            use_web: Whether to include web search

        Returns:
            Compact research memo
        """
        logger.info("Legacy research method called, using iterative agent")
        result = await self.research_iterative(query, game_id)
        return result.memo

    async def _plan_next_action(self, query: str, findings: list[ToolResult]) -> dict:
        """Plan the next research action using reasoning model.

        Args:
            query: Research query
            findings: Current findings

        Returns:
            Action dict with action_type, tool, and query
        """
        system_prompt = """You are a research planning assistant. Analyze the query and current findings to decide the next action.

Available tools:
- knowledge_base_search: Game-specific knowledge (Tunic)
- memory_search: Player's past gameplay sessions
- web_search: External web sources

Decide:
1. If research is complete, return {"action_type": "complete"}
2. Otherwise, return {"action_type": "search", "tool": "<tool_name>", "query": "<search_query>"}

Strategy:
- Check KB and memory first (faster)
- Use web search for external info
- Stop when you have sufficient coverage

Respond with JSON only."""

        findings_summary = "\n".join(
            f"- {f.tool_name}: {f.content[:100]}..." for f in findings
        )

        user_prompt = f"""Query: {query}

Findings so far ({len(findings)}):
{findings_summary if findings else "None yet"}

What should I do next?"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = await self._reasoning_llm.ainvoke(messages)
            content = str(response.content).strip()

            # Parse JSON response
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()

            action = json.loads(content)
            return action
        except Exception as e:
            logger.error("Action planning failed: %s", e, exc_info=True)
            # Fallback: search KB if no findings yet
            if not findings:
                return {
                    "action_type": "search",
                    "tool": "knowledge_base_search",
                    "query": query,
                }
            return {"action_type": "complete"}

    async def _execute_tool(self, action: dict, game_id: str) -> Optional[ToolResult]:
        """Execute a tool based on action plan.

        Args:
            action: Action dict from planner
            game_id: Game identifier

        Returns:
            ToolResult if successful, None otherwise
        """
        tool_name = action.get("tool")
        tool_query = action.get("query", "")

        if not tool_name:
            return None

        try:
            if tool_name == "knowledge_base_search":
                results = knowledge_base_search.invoke(
                    {"query": tool_query, "game_id": game_id, "qmd_url": self._qmd_url}
                )
                content = "\n\n".join(
                    f"[{r.source}] {r.content[:200]}" for r in results[:3]
                )
                return ToolResult(
                    tool_name=tool_name,
                    query=tool_query,
                    content=content or "No results found",
                    metadata={"count": len(results)},
                )

            elif tool_name == "memory_search":
                results = memory_search.invoke(
                    {"query": tool_query, "game_id": game_id}
                )
                content = "\n\n".join(
                    f"[{r.source}] {r.content[:200]}" for r in results[:2]
                )
                return ToolResult(
                    tool_name=tool_name,
                    query=tool_query,
                    content=content or "No results found",
                    metadata={"count": len(results)},
                )

            elif tool_name == "web_search":
                results = web_search.invoke({"query": tool_query, "game_id": game_id})
                content = "\n\n".join(
                    f"[{r.source}] {r.content[:200]}" for r in results[:2]
                )
                return ToolResult(
                    tool_name=tool_name,
                    query=tool_query,
                    content=content or "No results found",
                    metadata={"count": len(results)},
                )

        except Exception as e:
            logger.error("Tool %s failed: %s", tool_name, e, exc_info=True)
            return None

        return None

    async def _assess_completeness(
        self, query: str, findings: list[ToolResult]
    ) -> bool:
        """Assess if research is complete.

        Args:
            query: Research query
            findings: Current findings

        Returns:
            True if research is sufficient
        """
        if len(findings) < self._config.min_tool_calls:
            return False

        system_prompt = """You are a research quality assessor. Determine if the findings sufficiently answer the query.

Respond with JSON: {"is_sufficient": true/false, "reason": "explanation"}"""

        findings_summary = "\n".join(
            f"- {f.tool_name}: {f.content[:150]}..." for f in findings
        )

        user_prompt = f"""Query: {query}

Findings:
{findings_summary}

Is this sufficient to answer the query?"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = await self._reasoning_llm.ainvoke(messages)
            content = str(response.content).strip()

            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()

            assessment = json.loads(content)
            is_sufficient = assessment.get("is_sufficient", False)
            logger.info("Completeness assessment: %s", assessment.get("reason"))
            return is_sufficient
        except Exception as e:
            logger.error("Completeness assessment failed: %s", e, exc_info=True)
            return False

    async def _synthesize_findings(self, query: str, findings: list[ToolResult]) -> str:
        """Synthesize findings into final memo.

        Args:
            query: Research query
            findings: All research findings

        Returns:
            Synthesized memo
        """
        if not findings:
            return "No relevant information found."

        findings_text = "\n\n".join(
            f"Source: {f.tool_name}\nQuery: {f.query}\nContent: {f.content}"
            for f in findings
        )

        system_prompt = """You are a research assistant that synthesizes findings into compact memos.

Your task:
1. Extract key facts that answer the query
2. Cite sources clearly
3. Keep under 200 words
4. Focus on actionable information
5. Note any uncertainty

Format:
- Direct answer first
- Key findings as bullets
- Sources cited"""

        user_prompt = f"""Query: {query}

Findings:
{findings_text}

Synthesize into a compact memo."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = await self._synthesis_llm.ainvoke(messages)
            content = response.content
            memo = str(content).strip() if content else "Synthesis failed"
            logger.info("Synthesized memo (%d chars)", len(memo))
            return memo
        except Exception as e:
            logger.error("Synthesis failed: %s", e, exc_info=True)
            return findings[0].content[:200] if findings else "Synthesis failed"

    async def _calculate_confidence(
        self, query: str, findings: list[ToolResult]
    ) -> float:
        """Calculate confidence in research completeness.

        Args:
            query: Research query
            findings: Research findings

        Returns:
            Confidence score (0.0-1.0)
        """
        if not findings:
            return 0.0

        # Simple heuristic: more sources and steps = higher confidence
        unique_sources = len(set(f.tool_name for f in findings))
        steps = len(findings)

        # Base confidence on coverage
        confidence = min(1.0, (unique_sources * 0.3) + (steps * 0.1))
        return round(confidence, 2)
