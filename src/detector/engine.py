"""Detector engine for processing game frames and emitting events."""

import logging
from dataclasses import dataclass
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class DetectorEvent:
    """Event emitted by a detector when a game condition is detected."""

    id: str
    game_id: str
    timestamp: float
    type: str
    data: dict[str, Any]
    confidence: float


class FrameDetector(Protocol):
    """Protocol for frame-based game event detectors."""

    id: str
    game_id: str

    def probe(self, frame_bytes: bytes) -> Optional[DetectorEvent]:
        """Analyze a frame and return an event if detected.

        Args:
            frame_bytes: Raw image bytes (JPEG/PNG format).

        Returns:
            DetectorEvent if a condition is detected, None otherwise.
        """
        ...


class DetectorEngine:
    """Engine for running multiple detectors on game frames."""

    def __init__(self, logger_instance: Optional[logging.Logger] = None) -> None:
        """Initialize the detector engine.

        Args:
            logger_instance: Optional logger instance for detector output.
        """
        self._detectors: list[FrameDetector] = []
        self._logger = logger_instance or logger

    def register_detector(self, detector: FrameDetector) -> None:
        """Register a detector to run on each frame.

        Args:
            detector: Detector instance implementing FrameDetector protocol.
        """
        self._detectors.append(detector)
        self._logger.info(
            "Registered detector: %s (game: %s)", detector.id, detector.game_id
        )

    def process_frame(self, frame_bytes: bytes) -> list[DetectorEvent]:
        """Process a frame through all registered detectors.

        Args:
            frame_bytes: Raw image bytes (JPEG/PNG format).

        Returns:
            List of events detected in the frame (may be empty).
        """
        events: list[DetectorEvent] = []

        for detector in self._detectors:
            try:
                event = detector.probe(frame_bytes)
                if event:
                    events.append(event)
                    self._logger.debug(
                        "Detector %s emitted event: %s (confidence: %.2f)",
                        detector.id,
                        event.type,
                        event.confidence,
                    )
            except Exception as e:
                self._logger.error(
                    "Detector %s failed: %s", detector.id, e, exc_info=True
                )

        return events
