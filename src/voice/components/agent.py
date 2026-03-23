"""Agent pipeline for voice interaction - planning, context building, and response generation."""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import openai

from src.agent.job_manager import ResearchJob, ResearchJobManager
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
        job_manager: ResearchJobManager,
        system_instructions: str = SYSTEM_INSTRUCTIONS,
        model: str = "gpt-4.1-mini",
        game_id: str = "tunic",
        memory_dir: Optional[Path] = None,
        qmd_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cost_tracker: Optional[Any] = None,
        on_research_start: Optional[Any] = None,
        on_research_complete: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._context_manager = context_manager
        self._orchestrator = orchestrator
        self._system_instructions = system_instructions
        self._model = model
        self._game_id = game_id
        self._cost_tracker = cost_tracker
        self._job_manager = job_manager
        self._on_research_complete_tts = on_research_complete
        self._tts_lock = asyncio.Lock()

        api_key = api_key or openai.api_key
        if not api_key:
            raise ValueError("OpenAI API key required for agent pipeline.")

        self._client = openai.OpenAI(api_key=api_key)

        # Initialize planner for routing decisions
        self._planner = RequestPlanner(
            job_manager=job_manager,
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

    async def _on_research_complete(self, job: ResearchJob) -> None:
        """Handle research job completion with TTS notification.

        Args:
            job: Completed research job
                assistant_response=reply,
                scene_context=None,
                is_event_triggered=True,
        """
        if not job.result or not job.result.research_memo:
            logger.warning("Research job %s completed without result", job.job_id[:8])
            return

        async with self._tts_lock:
            if self._on_research_complete_tts:
                memo_preview = job.result.research_memo[:150]
                notification = f"Regarding your question: {memo_preview}"
                logger.info(
                    "Notifying user of research completion: %s", notification[:50]
                )
                try:
                    if asyncio.iscoroutinefunction(self._on_research_complete_tts):
                        await self._on_research_complete_tts(notification)
                    else:
                        self._on_research_complete_tts(notification)
                except Exception as e:
                    logger.error("Research completion TTS failed: %s", e, exc_info=True)

    async def _build_prompt_context(
        self,
        transcript: str,
        frame_provider: Any,
        scene_analyzer: Any,
    ) -> PromptContext:
        """Build minimal context for fast initial response (transcript only).

        VLM analysis, narrative, and evidence gathering are skipped for speed.
        Agent can request these via tools if needed for follow-up responses.
        """
        return PromptContext(
            transcript=transcript,
            scene=None,
            narrative="",
            evidence=EvidenceBundle(),
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
        """Generate LLM reply with conversation history context.

        Includes recent turns verbatim plus summary of older conversation.
        """
        # Get conversation history from memory manager
        history = self._memory_manager.get_context_for_llm()

        # Build messages: system + history + current user input
        messages = [
            {"role": "system", "content": self._system_instructions},
        ]
        messages.extend(history)
        messages.append(
            {"role": "user", "content": f"Player said: {prompt_context.transcript}"}
        )

        logger.debug(
            "[LLM] Sending %d messages (system + %d history + 1 current)",
            len(messages),
            len(history),
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
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
