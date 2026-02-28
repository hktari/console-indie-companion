"""OpenAI Realtime API voice session module.

Connects to OpenAI's Realtime API via WebSocket, handles bidirectional audio
(mic input → API → speaker output), and supports injecting game context as
system messages mid-conversation.
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import time
from typing import Optional, Any

from dotenv import load_dotenv

from src.voice.config import DEFAULT_SESSION_CONFIG, MODEL
from src.voice.audio import AudioManager
from src.voice.websocket import RealtimeWebSocket
from src.voice.utils import check_api_quota
from src.utils.logging_config import setup_logging

load_dotenv()

logger = logging.getLogger(__name__)


class VoiceSession:
    """Bidirectional voice session with OpenAI's Realtime API.

    Audio flows:
        Microphone → PCM16 24 kHz → base64 → WebSocket → API
        API → base64 → PCM16 24 kHz → Speakers/headphones

    Game context is injected via ``inject_context()`` which sends a system
    message into the running conversation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        system_instructions: str = "",
        cost_tracker: Optional[Any] = None,
        context_manager: Optional[Any] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OpenAI API key required.")

        self._cost_tracker = cost_tracker
        self._context_manager = context_manager
        self.session_config = DEFAULT_SESSION_CONFIG.copy()

        # Components
        self._audio = AudioManager(on_audio_data=self._on_mic_data)
        self._ws = RealtimeWebSocket(self.api_key, on_event=self._handle_server_event)

        # State
        self._tasks: list[asyncio.Task] = []
        self._session_start_time: Optional[float] = None
        self._audio_session_start: Optional[float] = None
        self._conversation_context: list[str] = []
        self._transcript_buffer: list[str] = []
        self._active_item_id: Optional[str] = None
        self._active_item_received_bytes: int = 0

    def _on_mic_data(self, pcm_bytes: bytes) -> None:
        """Callback for AudioManager when mic data is ready."""
        if self._ws.is_connected():
            b64 = base64.b64encode(pcm_bytes).decode("ascii")
            asyncio.create_task(
                self._ws.send_event({"type": "input_audio_buffer.append", "audio": b64})
            )

    async def start(self) -> None:
        """Connect to the Realtime API and start audio I/O."""
        logger.info("Connecting to OpenAI Realtime API (%s) …", MODEL)
        await self._ws.connect()

        # Wait for session.created
        raw = await asyncio.wait_for(self._ws.recv(), timeout=10)
        event = json.loads(raw)
        if event.get("type") == "session.created":
            logger.info(
                "Session created (id=%s)", event.get("session", {}).get("id", "?")
            )
        else:
            logger.warning("Expected session.created, got %s", event.get("type"))

        # Configure session
        config_payload = self.session_config.copy()

        await self._ws.send_event({"type": "session.update", "session": config_payload})

        self._audio.start_output()
        self._tasks = [
            asyncio.create_task(self._ws.receive_loop(), name="receive_loop"),
            asyncio.create_task(
                self._audio.start_input(
                    asyncio.get_running_loop(), self._ws.is_connected
                ),
                name="mic_input",
            ),
        ]

        self._session_start_time = time.time()
        self._audio_session_start = time.time()
        logger.info("Voice session started")

    async def stop(self) -> None:
        """Gracefully stop the session."""
        logger.info("Stopping voice session …")

        if self._cost_tracker and self._audio_session_start:
            duration = time.time() - self._audio_session_start
            self._cost_tracker.log_call(
                service="openai", model=MODEL, duration_seconds=duration
            )

        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

        self._audio.stop_all()
        await self._ws.disconnect()
        logger.info("Voice session stopped")

    async def inject_context(self, context_text: str) -> None:
        """Inject game context."""
        if not self._ws.is_connected():
            return

        logger.info("Injecting context: %s", context_text)
        await self._ws.send_event(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": context_text}],
                },
            }
        )
        self._conversation_context.append(context_text)
        await self._check_rotation()

    def _handle_server_event(self, event_type: str, event: dict) -> None:
        """Process incoming server events."""
        if event_type == "session.updated":
            logger.info("Session updated: %s", json.dumps(event, indent=2))
        elif event_type == "response.audio.delta":
            delta = event.get("delta", "")
            if delta:
                self._audio.enqueue_playback(base64.b64decode(delta))
        elif event_type == "response.audio_transcript.delta":
            fragment = event.get("delta", "")
            if fragment:
                self._transcript_buffer.append(fragment)

        elif event_type == "response.audio_transcript.done":
            full_transcript = "".join(self._transcript_buffer).strip()
            if full_transcript:
                logger.info(f"Assistant: {full_transcript}")
            self._transcript_buffer.clear()

        elif event_type == "input_audio_buffer.speech_started":
            # Handle interruption: truncate the current assistant item
            if self._active_item_id:
                # Calculate how much was actually played
                buffered_bytes = self._audio.get_playback_buffer_size()
                played_bytes = max(0, self._active_item_received_bytes - buffered_bytes)

                # Convert bytes to ms: 24kHz, 16-bit mono = 48000 bytes/sec
                # ms = (bytes / 48000) * 1000 = bytes / 48
                played_ms = played_bytes // 48

                logger.info(
                    "Interruption detected. Truncating item %s at %d ms",
                    self._active_item_id,
                    played_ms,
                )

                asyncio.create_task(
                    self._ws.send_event(
                        {
                            "type": "conversation.item.truncate",
                            "item_id": self._active_item_id,
                            "content_index": 0,
                            "audio_end_ms": played_ms,
                        }
                    )
                )
                # Also cancel the current response if one is in progress
                # TODO: test without canceling. VAD + item.truncate should be enough
                # asyncio.create_task(self._ws.send_event({"type": "response.cancel"}))

            self._audio.clear_playback()

        elif event_type == "input_audio_buffer.speech_stopped":
            # Handle end of speech
            if self._context_manager:
                context = self._context_manager.get_current_narrative()
                if context:
                    asyncio.create_task(self.inject_context(context))

        elif event_type == "response.done":
            # Clean up active item tracking
            self._active_item_id = None
            self._active_item_received_bytes = 0

            if self._context_manager:
                context = self._context_manager.get_current_narrative()
                if context:
                    asyncio.create_task(self.inject_context(context))

        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            if transcript:
                logger.info(f"[You] {transcript}")

        elif event_type == "response.function_call_arguments.done":
            asyncio.create_task(self._handle_tool_call(event))

        elif event_type == "error":
            error_data = event.get("error", {})
            logger.error(
                "API Error: %s (Type: %s, Code: %s, ID: %s)",
                error_data.get("message"),
                error_data.get("type"),
                error_data.get("code"),
                event.get("event_id"),
            )

    async def _handle_tool_call(self, event: dict) -> None:
        logger.debug("Handling tool call: %s", event)
        call_id = event.get("call_id")
        name = event.get("name")
        args_str = event.get("arguments", "{}")

        if name == "query_knowledge_base":
            try:
                from src.rag.query import query_tunic_knowledge

                args = json.loads(args_str)
                search_query = args.get("search_query", "")
                category_filter = args.get("metadata_category")

                logger.info(f"TAG query: '{search_query}' (filter: {category_filter})")

                results = query_tunic_knowledge(
                    question=search_query,
                    category_filter=category_filter,
                    n_results=3,
                )
                response_text = "\n\n".join(results) if results else "No info found."

                await self._ws.send_event(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": response_text,
                        },
                    }
                )
                await self._ws.send_event({"type": "response.create"})
            except Exception as e:
                logger.error("Tool error: %s", e)
                # Send error back to model so it's not stuck
                try:
                    await self._ws.send_event(
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": f"Error executing tool {name}: {str(e)}",
                            },
                        }
                    )
                    await self._ws.send_event({"type": "response.create"})
                except Exception as ws_err:
                    logger.error("Failed to send tool error back to WS: %s", ws_err)

    async def _check_rotation(self) -> None:
        if not self._session_start_time:
            return
        if time.time() - self._session_start_time > 55 * 60:
            logger.info("Rotating session...")
            # Simplified rotation logic for now
            await self.stop()
            await self.start()


async def _run_session(duration: int, instructions: str) -> None:
    await check_api_quota()
    session = VoiceSession(system_instructions=instructions)
    try:
        await session.start()
        await asyncio.sleep(duration)
    finally:
        await session.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--instructions", type=str, default="")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    try:
        setup_logging(log_level=args.log_level)
        asyncio.run(_run_session(args.duration, args.instructions))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
