"""Agent pipeline for voice interaction - planning, context building, and response generation."""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import openai

from src.agent.models import EvidenceBundle
from src.agent.planner import RequestPlanner
from src.context.manager import ContextManager
from src.memory.manager import ConversationMemoryManager
from src.prompts.tunic_companion import SYSTEM_INSTRUCTIONS
from src.rag.orchestrator import KnowledgeOrchestrator
from src.utils.performance import get_performance_tracker

logger = logging.getLogger(__name__)


@dataclass
class PromptContext:
    """Context bundle for agent prompt generation."""

    transcript: str
    scene: Optional[dict[str, Any]]
    narrative: str
    evidence: EvidenceBundle


class AgentPipeline:
    """Centralized agent logic for voice sessions.

    Handles:
    - Request planning and routing (KB, memory, web search)
    - Evidence gathering
    - Context building (scene, narrative, retrieval)
    - LLM response generation
    - Memory management
    """

    def __init__(
        self,
        context_manager: ContextManager,
        orchestrator: KnowledgeOrchestrator,
        system_instructions: str = SYSTEM_INSTRUCTIONS,
        model: str = "gpt-4.1-mini",
        game_id: str = "tunic",
        memory_dir: Optional[Path] = None,
        qmd_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cost_tracker: Optional[Any] = None,
        on_research_start: Optional[Any] = None,
    ) -> None:
        self._context_manager = context_manager
        self._orchestrator = orchestrator
        self._system_instructions = system_instructions
        self._model = model
        self._game_id = game_id
        self._cost_tracker = cost_tracker

        api_key = api_key or openai.api_key
        if not api_key:
            raise ValueError("OpenAI API key required for agent pipeline.")

        self._client = openai.OpenAI(api_key=api_key)

        # Initialize planner for routing decisions
        self._planner = RequestPlanner(
            qmd_url=qmd_url,
            model=model,
            api_key=api_key,
            on_research_start=on_research_start,
        )

        # Initialize memory manager
        if memory_dir is None:
            memory_dir = Path("var/memory")
        self._memory_manager = ConversationMemoryManager(
            game_id=game_id,
            memory_dir=memory_dir,
            api_key=api_key,
        )

        self._perf = get_performance_tracker()

    async def process_request(
        self,
        transcript: str,
        frame_provider: Any,
        scene_analyzer: Any,
    ) -> str:
        """Process a user request through the full agent pipeline.

        Args:
            transcript: User's transcribed speech
            frame_provider: Provider for game screenshots
            scene_analyzer: VLM for scene analysis

        Returns:
            Agent's response text
        """
        started_at = time.perf_counter()

        with self._perf.measure("agent.total_response_time", log_threshold=5.0):
            with self._perf.measure("agent.build_prompt_context", log_threshold=2.0):
                prompt_context = await self._build_prompt_context(
                    transcript, frame_provider, scene_analyzer
                )

            with self._perf.measure("agent.generate_reply", log_threshold=3.0):
                reply = await asyncio.to_thread(self._generate_reply, prompt_context)

            if reply:
                logger.info("Agent response: %s", reply)

                with self._perf.measure("agent.memory_update", log_threshold=0.5):
                    await asyncio.to_thread(
                        self._memory_manager.add_turn,
                        user_input=transcript,
                        assistant_response=reply,
                        scene_context=prompt_context.scene,
                        is_event_triggered=False,
                    )

                    if self._memory_manager.should_summarize():
                        await asyncio.to_thread(self._memory_manager.create_summary)

        if self._cost_tracker:
            self._cost_tracker.log_call(
                service="openai",
                model=self._model,
                duration_seconds=time.perf_counter() - started_at,
            )

        return reply

    async def inject_event(self, context_text: str) -> str:
        """Process a system event (e.g., detector trigger) and generate response.

        Args:
            context_text: Event context text

        Returns:
            Agent's response text
        """
        reply = await asyncio.to_thread(
            self._generate_reply_from_event,
            context_text,
        )
        if reply:
            logger.info("Agent response (event): %s", reply)

            await asyncio.to_thread(
                self._memory_manager.add_turn,
                user_input=None,
                assistant_response=reply,
                scene_context=None,
                is_event_triggered=True,
            )

            if self._memory_manager.should_summarize():
                await asyncio.to_thread(self._memory_manager.create_summary)

        return reply

    def flush_memory(self) -> None:
        """Flush conversation memory to disk."""
        self._memory_manager.flush()

    async def _build_prompt_context(
        self,
        transcript: str,
        frame_provider: Any,
        scene_analyzer: Any,
    ) -> PromptContext:
        """Build context for prompt generation."""
        with self._perf.measure("context.capture_frame", log_threshold=0.5):
            frame = frame_provider.capture_once()
            if frame is None:
                frame = frame_provider.get_latest_frame()

        scene: Optional[dict[str, Any]] = None
        if frame is not None:
            try:
                with self._perf.measure("context.vlm_analysis", log_threshold=2.0):
                    scene = await asyncio.to_thread(
                        scene_analyzer.analyze_screenshot, frame, "image/jpeg"
                    )
                if scene and isinstance(scene, dict) and "error" not in scene:
                    self._context_manager.update_scene(scene)
                else:
                    scene = None
            except Exception:
                logger.exception("Prompt-time VLM analysis failed")
                scene = None

        narrative = self._context_manager.get_current_narrative()

        # Use planner to decide routing and gather evidence
        evidence = EvidenceBundle()
        try:
            with self._perf.measure("planner.plan", log_threshold=0.5):
                decision = await asyncio.to_thread(
                    self._planner.plan, transcript, scene, narrative
                )
            logger.info(
                "Planner decision: %s (confidence: %.2f) - %s",
                decision.route.value,
                decision.confidence,
                decision.reasoning,
            )

            if decision.tools_to_call:
                query = (
                    self._build_retrieval_query(transcript, scene)
                    if scene
                    else transcript
                )
                with self._perf.measure("planner.gather_evidence", log_threshold=3.0):
                    evidence = await asyncio.to_thread(
                        self._planner.gather_evidence, decision, query, self._game_id
                    )
                logger.info(
                    "Evidence gathered: %d KB results, %d memory results, sources: %s",
                    len(evidence.kb_results),
                    len(evidence.memory_results),
                    ", ".join(evidence.sources),
                )
        except Exception:
            logger.exception("Planner execution failed, continuing without evidence")

        return PromptContext(
            transcript=transcript,
            scene=scene,
            narrative=narrative,
            evidence=evidence,
        )

    def _build_retrieval_query(self, transcript: str, scene: dict[str, Any]) -> str:
        """Build enhanced query for retrieval from transcript and scene."""
        scene_bits = [
            transcript,
            str(scene.get("location", "")),
            str(scene.get("activity", "")),
            str(scene.get("notable_items", "")),
        ]
        return " ".join(bit for bit in scene_bits if bit and bit != "None")

    def _generate_reply(self, prompt_context: PromptContext) -> str:
        """Generate LLM reply from prompt context."""
        scene_text = "No fresh scene analysis available."
        if prompt_context.scene:
            scene = prompt_context.scene
            scene_text = (
                f"Scene description: {scene.get('description', 'unknown')}\n"
                f"Location: {scene.get('location', 'unknown')}\n"
                f"Activity: {scene.get('activity', 'unknown')}\n"
                f"Visible enemies: {scene.get('enemies', 'none')}\n"
                f"Player health: {scene.get('health_status', 'unknown')}\n"
                f"UI elements: {scene.get('ui_elements', 'none')}\n"
                f"Notable items: {scene.get('notable_items', 'none')}"
            )

        retrieval_text = "No additional retrieval context."
        if prompt_context.evidence.has_evidence():
            all_results = prompt_context.evidence.get_all_results()
            if all_results:
                retrieval_text = "\n\n".join(
                    f"[{result.source}]\n{result.content}" for result in all_results[:3]
                )
            if prompt_context.evidence.research_memo:
                retrieval_text = (
                    f"Research findings:\n{prompt_context.evidence.research_memo}"
                )

        user_prompt = (
            f"Player said: {prompt_context.transcript}\n\n"
            f"Recent narrative: {prompt_context.narrative}\n\n"
            f"{scene_text}\n\n"
            f"Retrieved knowledge:\n{retrieval_text}"
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_instructions},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.8,
            max_tokens=160,
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return ""

    def _generate_reply_from_event(self, context_text: str) -> str:
        """Generate LLM reply from system event."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_instructions},
                {"role": "user", "content": context_text},
            ],
            temperature=0.7,
            max_tokens=60,
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return ""
