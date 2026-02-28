import asyncio
import json
import logging
from typing import Optional, Any, Callable

import websockets

from src.voice.config import WS_URL

logger = logging.getLogger(__name__)


class RealtimeWebSocket:
    """Manages the WebSocket connection to the OpenAI Realtime API."""

    def __init__(self, api_key: str, on_event: Callable[[str, dict], None]):
        self.api_key = api_key
        self.on_event = on_event
        self._ws: Optional[Any] = None
        self._connected: bool = False

    async def connect(self) -> None:
        """Connect to the Realtime API."""
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

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        self._connected = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                logger.debug("Error closing WebSocket (ignored)", exc_info=True)
            self._ws = None

    async def send_event(self, event: dict) -> None:
        """Serialize event to JSON and send it."""
        if self._ws is None or not self._connected:
            logger.warning(
                "WebSocket not connected, cannot send event: %s", event.get("type")
            )
            return
        try:
            await self._ws.send(json.dumps(event))
            event_type = event.get("type", "")
            if event_type != "input_audio_buffer.append":
                logger.debug("→ %s", event_type)
        except Exception:
            logger.error("Error sending %s", event.get("type"), exc_info=True)
            self._connected = False

    async def receive_loop(self) -> None:
        """Read events from the WebSocket and dispatch them via on_event callback."""
        if self._ws is None:
            return

        try:
            async for raw in self._ws:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Non-JSON message received, skipping")
                    continue

                event_type = event.get("type", "")
                if event_type not in [
                    "input_audio_buffer.append",
                    "response.audio.delta",
                    "response.audio_transcript.delta",
                ]:
                    logger.debug("← %s", event_type)
                self.on_event(event_type, event)

        except websockets.ConnectionClosed as exc:
            logger.warning("WebSocket closed: %s", exc)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Unexpected error in receive loop")
        finally:
            self._connected = False

    def is_connected(self) -> bool:
        """Return True if the connection is alive."""
        return self._connected and self._ws is not None

    async def recv(self) -> str:
        """Receive a single message from the WebSocket."""
        if self._ws is None:
            raise ConnectionError("WebSocket not connected")
        return await self._ws.recv()
