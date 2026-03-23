"""Realtime transcription with OpenAI VAD, no auto-response."""

import asyncio
import base64
import json
import logging
import os
import threading
from typing import Optional, Callable

from src.voice.audio import AudioManager
from src.voice.websocket import RealtimeWebSocket

logger = logging.getLogger(__name__)


class RealtimeTranscriber:
    """Handles realtime transcription with OpenAI VAD but no automatic assistant responses.

    This class streams microphone audio to OpenAI, receives VAD events and transcription,
    but does NOT trigger automatic model responses. Instead, it buffers finalized transcripts
    and exposes a manual commit mechanism for the application to decide when to invoke
    the local agent.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        on_partial_transcript: Optional[Callable[[str], None]] = None,
        on_final_transcript: Optional[Callable[[str], None]] = None,
        on_speech_stopped: Optional[Callable[[], None]] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OpenAI API key required for realtime transcription")

        self._on_partial_transcript = on_partial_transcript
        self._on_final_transcript = on_final_transcript
        self._on_speech_stopped = on_speech_stopped

        # Components
        self._audio = AudioManager(on_audio_data=self._on_mic_data)
        self._ws = RealtimeWebSocket(self.api_key, on_event=self._handle_server_event)

        # State
        self._tasks: list[asyncio.Task] = []
        self._transcript_buffer: list[str] = []
        self._current_partial: str = ""
        self._buffer_lock = threading.Lock()
        self._connected = False

    def _on_mic_data(self, pcm_bytes: bytes) -> None:
        """Callback for AudioManager when mic data is ready."""
        if self._ws.is_connected():
            b64 = base64.b64encode(pcm_bytes).decode("ascii")
            asyncio.create_task(
                self._ws.send_event({"type": "input_audio_buffer.append", "audio": b64})
            )

    async def start(self) -> None:
        """Connect to OpenAI Realtime API and start transcription."""
        logger.info("Connecting to OpenAI Realtime API for transcription...")
        await self._ws.connect()

        # Wait for session.created
        raw = await asyncio.wait_for(self._ws.recv(), timeout=10)
        event = json.loads(raw)
        if event.get("type") == "session.created":
            logger.info(
                "Transcription session created (id=%s)",
                event.get("session", {}).get("id", "?"),
            )
        else:
            logger.warning("Expected session.created, got %s", event.get("type"))

        # Configure session for transcription-only mode
        # Key: create_response=False means VAD will detect speech but NOT auto-answer
        session_config = {
            "modalities": ["text"],  # Only text, no audio output
            "input_audio_format": "pcm16",
            "input_audio_transcription": {"model": "whisper-1", "language": "en"},
            "input_audio_noise_reduction": {
                "type": "near_field",
            },
            "tracing": "auto",
            "turn_detection": {
                "type": "semantic_vad",
                "create_response": False,  # Critical: do NOT auto-respond
            },
        }

        await self._ws.send_event({"type": "session.update", "session": session_config})

        # Start audio input
        self._tasks = [
            asyncio.create_task(self._ws.receive_loop(), name="receive_loop"),
            asyncio.create_task(
                self._audio.start_input(
                    asyncio.get_running_loop(), self._ws.is_connected
                ),
                name="mic_input",
            ),
        ]

        self._connected = True
        logger.info("Realtime transcription started")

    async def stop(self) -> None:
        """Stop the transcription session."""
        logger.info("Stopping realtime transcription...")
        self._connected = False

        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

        self._audio.stop_all()
        await self._ws.disconnect()
        logger.info("Realtime transcription stopped")

    def _handle_server_event(self, event_type: str, event: dict) -> None:
        """Process incoming server events."""
        if event_type == "session.updated":
            logger.debug("Session updated")

        elif event_type == "input_audio_buffer.speech_started":
            logger.debug("Speech started")
            with self._buffer_lock:
                self._current_partial = ""

        elif event_type == "input_audio_buffer.speech_stopped":
            logger.debug("Speech stopped - waiting for transcription")
            # Note: We don't trigger submit here - the transcription.completed
            # event arrives AFTER speech_stopped, so we auto-submit there instead.
            if self._on_speech_stopped:
                try:
                    self._on_speech_stopped()
                except Exception as e:
                    logger.warning("Speech stopped callback failed: %s", e)

        elif event_type == "conversation.item.input_audio_transcription.completed":
            # Final transcript for this utterance
            transcript = event.get("transcript", "").strip()
            if transcript:
                logger.info("[Transcribed] %s", transcript)
                with self._buffer_lock:
                    self._transcript_buffer.append(transcript)
                    self._current_partial = ""
                    buffer_state = list(self._transcript_buffer)  # snapshot for logging
                logger.debug(
                    "[Buffer] After append: %s (count=%d)",
                    buffer_state,
                    len(buffer_state),
                )

                if self._on_final_transcript:
                    try:
                        self._on_final_transcript(transcript)
                    except Exception as e:
                        logger.warning("Final transcript callback failed: %s", e)

        elif event_type == "conversation.item.input_audio_transcription.delta":
            # Partial transcript update
            delta = event.get("delta", "")
            if delta:
                logger.debug("[Transcription delta] %s", delta)
                with self._buffer_lock:
                    self._current_partial += delta

                if self._on_partial_transcript:
                    try:
                        with self._buffer_lock:
                            self._on_partial_transcript(self._current_partial)
                    except Exception as e:
                        logger.warning("Partial transcript callback failed: %s", e)

        elif event_type == "error":
            error_data = event.get("error", {})
            logger.error(
                "API Error: %s (Type: %s, Code: %s)",
                error_data.get("message"),
                error_data.get("type"),
                error_data.get("code"),
            )

    def get_buffered_transcript(self) -> str:
        """Get all finalized transcripts accumulated since last clear.

        Returns:
            Combined transcript text.
        """
        with self._buffer_lock:
            result = " ".join(self._transcript_buffer)
            logger.debug(
                "[Buffer] get_buffered_transcript: '%s' (count=%d)",
                result,
                len(self._transcript_buffer),
            )
            return result

    def clear_buffer(self) -> None:
        """Clear the transcript buffer."""
        with self._buffer_lock:
            old_buffer = list(self._transcript_buffer)  # snapshot for logging
            self._transcript_buffer.clear()
            self._current_partial = ""
        logger.debug(
            "[Buffer] Cleared (was: %s, count=%d)", old_buffer, len(old_buffer)
        )

    def get_current_partial(self) -> str:
        """Get the current partial transcript (in-progress utterance).

        Returns:
            Partial transcript text.
        """
        with self._buffer_lock:
            return self._current_partial

    def is_connected(self) -> bool:
        """Check if transcription session is active."""
        return self._connected and self._ws.is_connected()
