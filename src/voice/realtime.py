"""OpenAI Realtime API voice session module.

Connects to OpenAI's Realtime API via WebSocket, handles bidirectional audio
(mic input → API → speaker output), and supports injecting game context as
system messages mid-conversation.

Usage:
    python -m src.voice.realtime --duration 30
    python -m src.voice.realtime --instructions "You are a helpful guide."
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import threading
import time
from typing import Optional, Any

import numpy as np

try:
    import sounddevice as sd
except (ImportError, OSError) as _sd_err:
    sd = None
    _sd_import_error = _sd_err

import websockets
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 24_000  # 24 kHz required by OpenAI Realtime API
CHANNELS = 1  # Mono
CHUNK_DURATION_MS = 100  # ms per mic capture chunk
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # samples/chunk
BYTES_PER_SAMPLE = 2  # PCM16 = 2 bytes per sample

MODEL = "gpt-realtime"
WS_URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"

DEFAULT_INSTRUCTIONS = (
    "You are a friendly and knowledgeable gaming companion for the game TUNIC. "
    "You've beaten the game and love helping other players. Be casual and conversational. "
    "When the player asks for help, give graduated hints - start vague, get more specific only if asked. "
    "Keep responses concise - 1-2 sentences max since this is a voice conversation. "
    "Respond ONLY in English, regardless of the language the player uses."
)


# ---------------------------------------------------------------------------
# VoiceSession
# ---------------------------------------------------------------------------


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
    ) -> None:
        """Initialise with OpenAI API key and initial system instructions.

        Args:
            api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.
            system_instructions: Initial instructions sent via ``session.update``.
            cost_tracker: Optional CostTracker instance to log API usage.
        """
        load_dotenv()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )

        self.system_instructions = system_instructions or DEFAULT_INSTRUCTIONS
        self._cost_tracker = cost_tracker

        # Connection state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._connected: bool = False
        self._tasks: list[asyncio.Task] = []

        # Audio output buffer (shared between asyncio receive loop and
        # sounddevice output callback thread → protected by a lock).
        self._playback_buf = bytearray()
        self._playback_lock = threading.Lock()

        # Sounddevice streams
        self._input_stream: Optional[object] = None
        self._output_stream: Optional[object] = None

        # Session rotation tracking
        self._session_start_time: Optional[float] = None
        self._conversation_context: list[str] = []  # Recent context for rotation
        
        # Audio cost tracking
        self._audio_session_start: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect to the Realtime API, configure the session, start audio I/O."""
        logger.info("Connecting to OpenAI Realtime API (%s) …", MODEL)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        try:
            self._ws = await websockets.connect(
                WS_URL,
                additional_headers=headers,
                max_size=2**24,  # 16 MiB – audio deltas can be large
            )
            self._connected = True
            logger.info("WebSocket connected")
        except Exception:
            logger.exception("Failed to connect to Realtime API")
            raise

        # Wait for the mandatory ``session.created`` event.
        try:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=10)
            event = json.loads(raw)
            if event.get("type") == "session.created":
                sid = event.get("session", {}).get("id", "?")
                logger.info("Session created (id=%s)", sid)
            else:
                logger.warning("Expected session.created, got %s", event.get("type"))
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for session.created")
            await self.stop()
            raise ConnectionError("Timeout waiting for session.created from API")

        # Configure the session (voice, VAD, audio format, instructions).
        await self._send_event(
            {
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"],
                    "instructions": self.system_instructions,
                    "voice": "ballad",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {"model": "whisper-1"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.3,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500
                    },
                    "temperature": 0.7,
                },
            }
        )

        # Start audio output stream *before* launching loops so playback is
        # ready when the first audio delta arrives.
        self._start_audio_output()

        # Launch background tasks.
        self._tasks = [
            asyncio.create_task(self._receive_loop(), name="receive_loop"),
            asyncio.create_task(self._mic_input_loop(), name="mic_input"),
        ]

        # Initialize session rotation and cost tracking timer
        self._session_start_time = time.time()
        if self._audio_session_start is None:
            self._audio_session_start = time.time()

        logger.info("Voice session started – speak into your microphone")

    async def stop(self) -> None:
        """Gracefully close the WebSocket connection and stop audio."""
        logger.info("Stopping voice session …")
        self._connected = False
        
        # Log cost if tracker available
        if self._cost_tracker and self._audio_session_start:
            duration = time.time() - self._audio_session_start
            self._cost_tracker.log_call(
                service="openai",
                model=MODEL,
                duration_seconds=duration
            )
            self._audio_session_start = None

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()

        self._stop_audio()

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                logger.debug("Error closing WebSocket (ignored)", exc_info=True)
            self._ws = None

        logger.info("Voice session stopped")

    async def inject_context(self, context_text: str) -> None:
        """Inject a game context update as a system message.

        This is the integration point for the VLM scene descriptions: the
        ContextManager calls this method to feed new information into the
        ongoing conversation.

        Args:
            context_text: Plain-text description of what's happening in the game.
        """
        if not self._connected:
            logger.warning("Cannot inject context – not connected")
            return

        logger.debug("Injecting context (%d chars): %.100s…", len(context_text), context_text)

        await self._send_event(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": context_text,
                        }
                    ],
                },
            }
        )

        # Store context for rotation summary
        self._conversation_context.append(context_text)
        if len(self._conversation_context) > 10:
            self._conversation_context = self._conversation_context[-10:]

        # Check if rotation is needed
        await self._check_rotation()

    def is_connected(self) -> bool:
        """Return *True* if the WebSocket connection is alive."""
        return self._connected and self._ws is not None

    async def _check_rotation(self) -> None:
        """Check if session needs rotation (55 min limit) and rotate if needed.

        Rotation process:
        1. Create a condensed summary of recent conversation context
        2. Close the current WebSocket connection
        3. Open a new WebSocket connection
        4. Inject the condensed context as the initial system message

        Handles rotation failures gracefully (retry once, then log and continue).
        """
        if self._session_start_time is None:
            return

        elapsed = time.time() - self._session_start_time
        rotation_threshold = 55 * 60  # 55 minutes in seconds

        if elapsed < rotation_threshold:
            return

        logger.info(
            "Session rotation triggered (%.0f min elapsed, limit is 55 min)",
            elapsed / 60,
        )

        # Create condensed context summary from recent conversation
        context_summary = self._create_context_summary()

        # Attempt rotation with one retry
        for attempt in range(2):
            try:
                logger.info("Rotating session (attempt %d/2)...", attempt + 1)

                # Close current connection
                self._connected = False
                if self._ws is not None:
                    try:
                        await self._ws.close()
                    except Exception:
                        logger.debug("Error closing WebSocket during rotation", exc_info=True)
                    self._ws = None

                # Cancel background tasks
                for task in self._tasks:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                self._tasks.clear()

                # Stop audio
                self._stop_audio()

                # Brief pause before reconnecting
                await asyncio.sleep(1)

                # Reconnect with fresh session
                await self._reconnect_with_context(context_summary)
                logger.info("Session rotation completed successfully")
                return

            except Exception:
                logger.exception("Session rotation attempt %d failed", attempt + 1)
                if attempt == 1:
                    logger.error("Session rotation failed after 2 attempts, continuing with current session")
                    return
                # Retry on first failure
                await asyncio.sleep(2)

    def _create_context_summary(self) -> str:
        """Create a condensed summary of recent conversation context.

        Returns:
            A brief text summary of the conversation so far.
        """
        if not self._conversation_context:
            return "(No prior context)"

        # Keep last 3 context updates for summary
        recent = self._conversation_context[-3:]
        summary = "Recent context: " + " | ".join(recent)
        return summary[:500]  # Limit to 500 chars

    async def _reconnect_with_context(self, context_summary: str) -> None:
        """Reconnect to the API and inject the context summary.

        Args:
            context_summary: Condensed context to inject as system message.
        """
        logger.info("Reconnecting to OpenAI Realtime API …")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        try:
            self._ws = await websockets.connect(
                WS_URL,
                additional_headers=headers,
                max_size=2**24,
            )
            self._connected = True
            logger.info("WebSocket reconnected")
        except Exception:
            logger.exception("Failed to reconnect to Realtime API")
            raise

        # Wait for session.created
        try:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=10)
            event = json.loads(raw)
            if event.get("type") == "session.created":
                sid = event.get("session", {}).get("id", "?")
                logger.info("Session created (id=%s)", sid)
            else:
                logger.warning("Expected session.created, got %s", event.get("type"))
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for session.created after rotation")
            await self.stop()
            raise ConnectionError("Timeout waiting for session.created from API")

        # Configure session with context summary
        await self._send_event(
            {
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"],
                    "instructions": self.system_instructions,
                    "voice": "sage",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {"model": "whisper-1"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500
                    },
                    "temperature": 0.7,
                },
            }
        )

        # Inject context summary as system message
        if context_summary:
            await self.inject_context(context_summary)

        # Restart audio and background tasks
        self._start_audio_output()
        self._session_start_time = time.time()  # Reset rotation timer
        self._tasks = [
            asyncio.create_task(self._receive_loop(), name="receive_loop"),
            asyncio.create_task(self._mic_input_loop(), name="mic_input"),
        ]
        logger.info("Session rotation complete – voice session resumed")

    # ------------------------------------------------------------------
    # WebSocket helpers
    # ------------------------------------------------------------------

    async def _send_event(self, event: dict) -> None:
        """Serialise *event* to JSON and send it over the WebSocket."""
        if self._ws is None:
            logger.warning("WebSocket not connected, cannot send event")
            return
        try:
            await self._ws.send(json.dumps(event))
            logger.debug("→ %s", event.get("type"))
        except Exception:
            logger.error("Error sending %s", event.get("type"), exc_info=True)
            self._connected = False

    # ------------------------------------------------------------------
    # Receive loop & event dispatcher
    # ------------------------------------------------------------------

    async def _receive_loop(self) -> None:
        """Read events from the WebSocket and dispatch them."""
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Non-JSON message received, skipping")
                    continue
                event_type = event.get("type", "")
                logger.debug("← %s", event_type)
                self._handle_server_event(event_type, event)
        except websockets.ConnectionClosed as exc:
            logger.warning("WebSocket closed: %s", exc)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Unexpected error in receive loop")
        finally:
            self._connected = False

    def _handle_server_event(self, event_type: str, event: dict) -> None:
        """Process a single server event (runs on the asyncio thread)."""

        # -- Session lifecycle -------------------------------------------
        if event_type == "session.updated":
            logger.debug("Session configuration updated")

        # -- Audio output ------------------------------------------------
        elif event_type == "response.audio.delta":
            delta = event.get("delta", "")
            if delta:
                logger.debug("Received audio delta: %d bytes", len(delta))
                pcm_bytes = base64.b64decode(delta)
                self._enqueue_playback(pcm_bytes)

        elif event_type == "response.audio.done":
            logger.debug("Audio response stream complete")

        # -- Transcript (log assistant words as INFO) ---------------
        elif event_type == "response.audio_transcript.delta":
            fragment = event.get("delta", "")
            if fragment:
                # We still print to stdout for real-time feedback, 
                # but could also log to a file if needed.
                print(fragment, end="", flush=True)

        elif event_type == "response.audio_transcript.done":
            print()  # newline after full transcript
            logger.info("Assistant response complete")

        # -- VAD / turn detection ----------------------------------------
        elif event_type == "input_audio_buffer.speech_started":
            logger.debug("User speech started – clearing playback buffer")
            self._clear_playback()
            # If the user interrupts, we also want to stop the current response
            # But the Realtime API handles this automatically with server_vad

        elif event_type == "input_audio_buffer.speech_stopped":
            logger.debug("User speech stopped")

        elif event_type == "input_audio_buffer.committed":
            logger.debug("Audio buffer committed by server VAD")

        # -- Conversation items ------------------------------------------
        elif event_type == "conversation.item.created":
            item = event.get("item", {})
            logger.debug(
                "Item created: role=%s type=%s",
                item.get("role"),
                item.get("type"),
            )
            
        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            if transcript:
                print(f"\n[You] {transcript}\n")
                logger.info("User transcript: %s", transcript)

        # -- Response lifecycle ------------------------------------------
        elif event_type == "response.created":
            logger.debug("Response generation started")

        elif event_type == "response.done":
            usage = event.get("response", {}).get("usage", {})
            if usage:
                logger.debug(
                    "Response done – tokens in=%s out=%s",
                    usage.get("input_tokens", "?"),
                    usage.get("output_tokens", "?"),
                )

        # -- Errors ------------------------------------------------------
        elif event_type == "error":
            err = event.get("error", {})
            logger.error(
                "API error [%s/%s]: %s",
                err.get("type", "?"),
                err.get("code", "?"),
                err.get("message", "?"),
            )

        # -- Rate limits -------------------------------------------------
        elif event_type == "rate_limits.updated":
            pass  # noisy, suppress

        # -- Catch-all ---------------------------------------------------
        else:
            logger.debug("← %s (unhandled)", event_type)

    # ------------------------------------------------------------------
    # Audio input (microphone)
    # ------------------------------------------------------------------

    async def _mic_input_loop(self) -> None:
        """Capture audio from the default microphone, base64-encode, and send."""
        if sd is None:
            logger.warning(
                "sounddevice unavailable (%s) – mic input disabled", _sd_import_error
            )
            # Keep task alive so it can be cancelled cleanly.
            while self._connected:
                await asyncio.sleep(1)
            return

        loop = asyncio.get_running_loop()
        audio_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=50)

        # PyAudio configuration
        import pyaudio
        p = pyaudio.PyAudio()
        
        def _mic_callback(in_data, frame_count, time_info, status_flags):
            if status_flags:
                logger.warning("Mic status: %s", status_flags)
            
            # Apply input gain (e.g. 100x) and clip to [-32768, 32767]
            # Since PyAudio provides raw bytes, we'll convert to numpy array to amplify
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            # Use float32 for calculations to avoid overflow during multiplication
            amplified = np.clip(audio_data.astype(np.float32) * 100.0, -32768, 32767)
            pcm16 = amplified.astype(np.int16).tobytes()
            
            try:
                loop.call_soon_threadsafe(audio_q.put_nowait, pcm16)
            except asyncio.QueueFull:
                pass  # drop frame rather than block the audio thread
            
            return (None, pyaudio.paContinue)

        try:
            self._input_stream = p.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SAMPLES,
                stream_callback=_mic_callback
            )
            self._input_stream.start_stream()
            logger.debug("Microphone capture started (24 kHz mono PCM16)")

            while self._connected:
                try:
                    pcm_bytes = await asyncio.wait_for(audio_q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                b64 = base64.b64encode(pcm_bytes).decode("ascii")
                await self._send_event(
                    {"type": "input_audio_buffer.append", "audio": b64}
                )

        except OSError as exc:
            logger.warning("Audio device error – continuing without mic: %s", exc)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error in mic input loop")
        finally:
            if self._input_stream is not None:
                try:
                    self._input_stream.stop_stream()
                    self._input_stream.close()
                except Exception:
                    pass
                self._input_stream = None

    # ------------------------------------------------------------------
    # Audio output (speakers / headphones)
    # ------------------------------------------------------------------

    def _start_audio_output(self) -> None:
        """Open an ``sd.OutputStream`` whose callback drains ``_playback_buf``."""
        if sd is None:
            logger.warning("sounddevice unavailable – audio output disabled")
            return

        def _speaker_callback(
            outdata: np.ndarray, frames: int, time_info: object, status: object
        ) -> None:
            if status:
                logger.warning("Speaker status: %s", status)

            need = frames * BYTES_PER_SAMPLE
            with self._playback_lock:
                available = len(self._playback_buf)
                if available >= need:
                    raw = bytes(self._playback_buf[:need])
                    del self._playback_buf[:need]
                elif available > 0:
                    raw = bytes(self._playback_buf) + b"\x00" * (need - available)
                    self._playback_buf.clear()
                else:
                    outdata.fill(0)
                    return

            # logger.debug("Playing back %d bytes (%d remaining in buffer)", len(raw), len(self._playback_buf))
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
            outdata[:, 0] = samples[:frames]

        try:
            self._output_stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=CHUNK_SAMPLES,
                callback=_speaker_callback,
            )
            self._output_stream.start()
            logger.debug("Audio output started (24 kHz mono)")
        except OSError as exc:
            logger.warning("Cannot open audio output – continuing without: %s", exc)

    def _enqueue_playback(self, pcm_bytes: bytes) -> None:
        """Append decoded PCM16 bytes to the playback buffer (thread-safe)."""
        with self._playback_lock:
            self._playback_buf.extend(pcm_bytes)

    def _clear_playback(self) -> None:
        """Clear the playback buffer (e.g. on user interruption)."""
        with self._playback_lock:
            self._playback_buf.clear()

    def _stop_audio(self) -> None:
        """Stop and close all sounddevice streams."""
        for stream_attr in ("_input_stream", "_output_stream"):
            stream = getattr(self, stream_attr, None)
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
                setattr(self, stream_attr, None)

        self._clear_playback()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


async def _check_api_quota() -> None:
    """Perform a pre-flight check to ensure the OpenAI API key is valid and has quota."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)
        
    logger.info("Performing pre-flight API check...")
    try:
        # We use a simple models list request to verify auth
        # It's lightweight and fails if the key is invalid or quota is exceeded
        import urllib.request
        import json
        req = urllib.request.Request(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                logger.info("API key is valid. Checking quota by initializing a minimal text completion...")
        
        # To truly check quota for a model, we try a minimal 1-token completion
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps({
                "model": "gpt-4o-mini", # Use a cheap model for the check
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1
            }).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        )
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                logger.info("Pre-flight check passed! API quota is available.")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            error_json = json.loads(error_body)
            error_msg = error_json.get("error", {}).get("message", e.reason)
            error_code = error_json.get("error", {}).get("code", "unknown")
            
            logger.error(f"API Error [{error_code}]: {error_msg}")
            
            if error_code == "insufficient_quota":
                print("\n❌ ERROR: Insufficient OpenAI API Quota")
                print("Your API key is valid, but you have run out of credits or hit your billing limit.")
                print("Please check your billing details at: https://platform.openai.com/account/billing")
            elif e.code == 401:
                print("\n❌ ERROR: Invalid OpenAI API Key")
                print("Please ensure your OPENAI_API_KEY is correct.")
            else:
                print(f"\n❌ ERROR: API Check Failed ({e.code})")
                print(f"Details: {error_msg}")
        except Exception:
            logger.error(f"HTTP Error {e.code}: {e.reason}")
            print(f"\n❌ ERROR: Pre-flight check failed (HTTP {e.code})")
            
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to perform pre-flight check: {e}")
        print(f"\n❌ ERROR: Could not connect to OpenAI API: {e}")
        sys.exit(1)

async def _run_session(duration: int, instructions: str) -> None:
    """Run an interactive voice session for *duration* seconds."""
    await _check_api_quota()
    
    session = VoiceSession(system_instructions=instructions)

    try:
        await session.start()
        print("\n🎙️  Voice session active — speak into your microphone!")
        print(f"   Duration : {duration}s")
        print(f"   Model    : {MODEL}")
        print("   Press Ctrl+C to stop early\n")
        await asyncio.sleep(duration)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        await session.stop()
        print("Session ended.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI Realtime API – interactive voice session",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Session duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default="",
        help="System instructions (default: TUNIC companion prompt)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        asyncio.run(_run_session(duration=args.duration, instructions=args.instructions))
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
