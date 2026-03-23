"""Realtime transcription voice session implementation."""

import asyncio
import logging
import os
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Optional

from src.agent.job_manager import ResearchJobManager
from src.agent.research import ResearchSubagent
from src.context.manager import ContextManager
from src.prompts.tunic_companion import SYSTEM_INSTRUCTIONS
from src.rag.orchestrator import KnowledgeOrchestrator
from src.voice.components.agent import AgentPipeline
from src.voice.components.output import OpenAITTSPlayer
from src.voice.realtime_transcriber import RealtimeTranscriber

logger = logging.getLogger(__name__)


class RealtimeTranscriptionSession:
    """Realtime transcription voice session.

    Continuous transcription with manual submit (primary use case).
    Uses OpenAI Realtime API for VAD-based transcription, but processes
    requests through local agent pipeline (not full-duplex conversation).
    """

    def __init__(
        self,
        frame_provider: Any,
        scene_analyzer: Any,
        context_manager: ContextManager,
        orchestrator: KnowledgeOrchestrator,
        cost_tracker: Optional[Any] = None,
        system_instructions: str = SYSTEM_INSTRUCTIONS,
        model: str = "gpt-4.1-mini",
        tts_model: str = "tts-1",
        tts_voice: str = "alloy",
        game_id: str = "tunic",
        memory_dir: Optional[Path] = None,
        qmd_url: Optional[str] = None,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required.")

        self._frame_provider = frame_provider
        self._scene_analyzer = scene_analyzer
        self._cost_tracker = cost_tracker

        # Initialize TTS player early so agent pipeline callback can use it
        self._tts_player = OpenAITTSPlayer(
            api_key=api_key,
            model=tts_model,
            voice=tts_voice,
        )

        # Initialize research subagent for iterative research
        self._research_subagent = ResearchSubagent(
            qmd_url=qmd_url,
            model=model,
            api_key=api_key,
        )

        # Initialize research job manager with iterative research executor
        self._job_manager = ResearchJobManager(
            research_executor=self._research_subagent.research_iterative,
        )

        # Initialize agent pipeline
        self._agent = AgentPipeline(
            context_manager=context_manager,
            orchestrator=orchestrator,
            job_manager=self._job_manager,
            system_instructions=system_instructions,
            model=model,
            game_id=game_id,
            memory_dir=memory_dir,
            qmd_url=qmd_url,
            api_key=api_key,
            cost_tracker=cost_tracker,
            on_research_start=lambda: self._tts_player.speak("Let me look that up."),
            on_research_complete=lambda msg: self._tts_player.speak(msg),
        )

        # Initialize realtime transcriber with auto-submit callback on final transcript
        # Note: We use on_final_transcript instead of on_speech_stopped because
        # transcription.completed arrives AFTER speech_stopped, so this avoids the race condition
        # where we'd submit the previous utterance's transcript.
        self._realtime_transcriber = RealtimeTranscriber(
            api_key=api_key,
            on_final_transcript=self._on_final_transcript_ready,
        )

        self._active_lock = asyncio.Lock()
        self._last_response: Optional[str] = None
        self._pending_submit: Optional[Future[Optional[str]]] = None

    async def start_transcription(self) -> None:
        """Start the transcription session."""
        await self._job_manager.start()
        await self._realtime_transcriber.start()

    async def stop_transcription(self) -> None:
        """Stop the transcription session."""
        await self._realtime_transcriber.stop()
        await self._job_manager.stop()

    async def submit_transcript_and_respond(self) -> Optional[str]:
        """Submit buffered transcript to agent.

        Returns:
            Agent response text if successful, None otherwise.
        """
        transcript = self._realtime_transcriber.get_buffered_transcript()
        if not transcript:
            logger.info("No transcript buffered to submit")
            return None

        # Clear buffer after retrieving
        self._realtime_transcriber.clear_buffer()
        logger.debug("[Queue] Buffer cleared, processing request")

        async with self._active_lock:
            reply = await self._agent.process_request(
                transcript, self._frame_provider, self._scene_analyzer
            )
            if reply:
                self._last_response = reply
                await asyncio.to_thread(self._tts_player.speak, reply)

            return reply

    def _on_final_transcript_ready(self, transcript: str) -> None:
        """Callback triggered when a final transcript is ready.

        n        This is the correct time to auto-submit because the transcript is now
                in the buffer. Speech_stopped fires BEFORE transcription completes,
                causing the race condition where we'd submit the previous utterance.
        """
        if not self._realtime_transcriber:
            return

        # Check if there's already a pending submit - don't queue another
        if self._pending_submit and not self._pending_submit.done():
            logger.debug(
                "[Queue] Final transcript ready but previous submit still processing, skipping"
            )
            return

        logger.info("[Queue] Final transcript ready, auto-submitting: '%s'", transcript)

        # Schedule the async submit_transcript_and_respond in the event loop
        try:
            loop = asyncio.get_event_loop()
            self._pending_submit = asyncio.run_coroutine_threadsafe(
                self.submit_transcript_and_respond(), loop
            )
            logger.debug(
                "[Queue] Submit scheduled, pending=%s", not self._pending_submit.done()
            )
        except Exception as e:
            logger.error("Failed to schedule auto-submit: %s", e)

    def is_transcribing(self) -> bool:
        """Check if transcription session is active."""
        return self._realtime_transcriber.is_connected()

    async def inject_context(self, context_text: str) -> None:
        """Inject system event context (e.g., from detector)."""
        reply = await self._agent.inject_event(context_text)
        if reply:
            self._last_response = reply
            await asyncio.to_thread(self._tts_player.speak, reply)

    def get_last_response(self) -> Optional[str]:
        """Get the last agent response."""
        return self._last_response

    async def shutdown(self) -> None:
        """Shutdown the voice session and flush memory."""
        logger.info("Shutting down realtime transcription session, flushing memory...")

        # Stop realtime transcription if active
        if self._realtime_transcriber and self._realtime_transcriber.is_connected():
            await self._realtime_transcriber.stop()

        # Stop job manager
        await self._job_manager.stop()

        self._agent.flush_memory()
        logger.info("Memory flushed, shutdown complete")
