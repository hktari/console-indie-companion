"""Replay mode capture module for testing without a live window.

Reads pre-captured screenshots from a directory and cycles through them
at a configurable interval. Provides the same interface as CaptureService.
"""

import io
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


class ReplayCapture:
    """Replays pre-captured screenshots from a directory instead of capturing live."""

    def __init__(
        self, screenshot_dir: str, interval: float = 3.0, jpeg_quality: int = 85
    ):
        """Initialize replay from a directory of screenshots.

        Args:
            screenshot_dir: Path to directory containing PNG/JPG screenshots.
            interval: Seconds between captures in periodic mode.
            jpeg_quality: JPEG compression quality (1-100). Lower = smaller files.
        """
        self.screenshot_dir = screenshot_dir
        self.interval = interval
        self.jpeg_quality = jpeg_quality

        self._image_files: list[str] = []
        self._current_index = 0
        self._latest_frame: Optional[bytes] = None
        self._frame_lock = threading.Lock()

        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def find_window(self) -> bool:
        """Validate that the screenshot directory exists and contains images.

        Returns:
            True if directory exists and has image files, False otherwise.
        """
        if not os.path.isdir(self.screenshot_dir):
            logger.error("Screenshot directory not found: %s", self.screenshot_dir)
            return False

        # Scan for PNG and JPG files, sorted alphabetically
        image_extensions = {".png", ".jpg", ".jpeg"}
        files = []
        try:
            for filename in sorted(os.listdir(self.screenshot_dir)):
                if Path(filename).suffix.lower() in image_extensions:
                    files.append(filename)
        except Exception:
            logger.exception(
                "Failed to scan screenshot directory: %s", self.screenshot_dir
            )
            return False

        if not files:
            logger.error("No PNG/JPG files found in %s", self.screenshot_dir)
            return False

        self._image_files = files
        self._current_index = 0
        logger.info(
            "Found %d image files in %s", len(self._image_files), self.screenshot_dir
        )
        return True

    def capture_once(self) -> Optional[bytes]:
        """Read the next screenshot from the directory, convert to JPEG bytes.

        Cycles back to the beginning when all files are exhausted.

        Returns:
            JPEG bytes of the loaded image, or None on failure.
        """
        if not self._image_files:
            logger.error("No image files loaded. Call find_window() first.")
            return None

        # Get the current file and advance the index
        filename = self._image_files[self._current_index]
        self._current_index = (self._current_index + 1) % len(self._image_files)

        filepath = os.path.join(self.screenshot_dir, filename)

        try:
            # Open the image and convert to JPEG bytes
            img = Image.open(filepath)
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.jpeg_quality)
            jpeg_bytes = buf.getvalue()
        except Exception:
            logger.exception("Failed to load/convert image: %s", filepath)
            return None

        with self._frame_lock:
            self._latest_frame = jpeg_bytes

        logger.debug("Loaded %d bytes JPEG from %s", len(jpeg_bytes), filename)
        return jpeg_bytes

    def get_latest_frame(self) -> Optional[bytes]:
        """Get the most recently loaded frame (thread-safe).

        Returns:
            JPEG bytes of the last successful capture, or None if no capture yet.
        """
        with self._frame_lock:
            return self._latest_frame

    def start(self) -> None:
        """Start cycling through screenshots in a background daemon thread."""
        if self._capture_thread is not None and self._capture_thread.is_alive():
            logger.warning("Replay thread already running")
            return

        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="replay-capture",
            daemon=True,
        )
        self._capture_thread.start()
        logger.info("Started replay capture every %.1fs", self.interval)

    def stop(self) -> None:
        """Stop the replay loop and wait for thread to finish."""
        if self._capture_thread is None or not self._capture_thread.is_alive():
            return

        self._stop_event.set()
        self._capture_thread.join(timeout=self.interval + 2)
        if self._capture_thread.is_alive():
            logger.warning("Replay thread did not stop cleanly")
        else:
            logger.info("Replay thread stopped")
        self._capture_thread = None

    def _capture_loop(self) -> None:
        """Internal loop for periodic capture. Runs in background thread."""
        while not self._stop_event.is_set():
            try:
                self.capture_once()
            except Exception:
                logger.exception("Unexpected error in replay loop")
            self._stop_event.wait(timeout=self.interval)
