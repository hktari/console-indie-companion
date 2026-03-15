"""Push-to-talk voice session implementation."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Optional

from src.context.manager import ContextManager
from src.prompts.tunic_companion import SYSTEM_INSTRUCTIONS
from src.rag.orchestrator import KnowledgeOrchestrator
from src.voice.components.agent import AgentPipeline
from src.voice.components.input import BatchTranscriber, PTTRecorder
from src.voice.components.output import OpenAITTSPlayer

logger = logging.getLogger(__name__)


class PTTVoiceSession:
    """Push-to-talk voice session.

    Supports two modes:
    - push-to-talk: Hold key to record, release to stop and process
    - toggle: Press once to start, press again to stop and process
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
        stt_model: str = "whisper-1",
        tts_model: str = "tts-1",
        tts_voice: str = "alloy",
        game_id: str = "tunic",
        memory_dir: Optional[Path] = None,
        qmd_url: Optional[str] = None,
        recorder_mode: str = "push-to-talk",
        enable_recorder_sounds: bool = True,
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

        # Initialize agent pipeline
        self._agent = AgentPipeline(
            context_manager=context_manager,
            orchestrator=orchestrator,
            system_instructions=system_instructions,
            model=model,
            game_id=game_id,
            memory_dir=memory_dir,
            qmd_url=qmd_url,
            api_key=api_key,
            cost_tracker=cost_tracker,
            on_research_start=lambda: self._tts_player.speak("Let me look that up."),
        )

        # Initialize PTT recorder and transcriber
        self._recorder = PTTRecorder(
            mode=recorder_mode,
            enable_sounds=enable_recorder_sounds,
        )
        self._transcriber = BatchTranscriber(api_key=api_key, model=stt_model)

        self._active_lock = asyncio.Lock()
        self._last_response: Optional[str] = None

    def start_recording(self) -> None:
        """Start recording (for push-to-talk mode)."""
        self._recorder.start_recording()

    async def stop_recording_and_respond(self) -> Optional[str]:
        """Stop recording and process (PTT mode)."""
        pcm_bytes = await asyncio.to_thread(self._recorder.stop_recording)
        if not pcm_bytes:
            return None

        async with self._active_lock:
            transcript = await asyncio.to_thread(
                self._transcriber.transcribe_pcm16, pcm_bytes
            )
            if not transcript:
                logger.info("No transcript captured from push-to-talk audio")
                return None

            reply = await self._agent.process_request(
                transcript, self._frame_provider, self._scene_analyzer
            )
            if reply:
                self._last_response = reply
                await asyncio.to_thread(self._tts_player.speak, reply)

            return reply

    async def toggle_recording_and_respond(self) -> Optional[str]:
        """Toggle recording state - starts if stopped, stops and processes if recording.

        Best for controller gameplay where holding a button is awkward.
        """
        pcm_bytes = await asyncio.to_thread(self._recorder.toggle_recording)

        # If we got audio bytes back, we just stopped recording - process it
        if pcm_bytes:
            async with self._active_lock:
                transcript = await asyncio.to_thread(
                    self._transcriber.transcribe_pcm16, pcm_bytes
                )
                if not transcript:
                    logger.info("No transcript captured from toggle audio")
                    return None

                reply = await self._agent.process_request(
                    transcript, self._frame_provider, self._scene_analyzer
                )
                if reply:
                    self._last_response = reply
                    await asyncio.to_thread(self._tts_player.speak, reply)

                return reply

        # Started recording, no response yet
        return None

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recorder.is_recording()

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
        logger.info("Shutting down PTT voice session, flushing memory...")
        self._agent.flush_memory()
        logger.info("Memory flushed, shutdown complete")
