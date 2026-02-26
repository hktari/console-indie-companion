"""Screen capture module using X11/xdotool and mss.

Captures a window by ID, gets its screen region, and returns JPEG bytes.
Designed for Linux with X11 (Wayland not supported).
"""

import argparse
import io
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

import mss
from PIL import Image

from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class WindowGeometry:
    """Position and size of a window on screen."""

    window_id: str
    x: int
    y: int
    width: int
    height: int


class CaptureService:
    """Captures screenshots of a specific window by ID using xdotool + mss."""

    def __init__(self, window_id: Optional[str] = None, window_name: Optional[str] = None, interval: float = 3.0, jpeg_quality: int = 85):
        """Initialize capture for a specific window.

        Args:
            window_id: X11 window ID (hex or decimal). If provided, takes precedence.
            window_name: Name of the window to search for if window_id is not provided.
            interval: Seconds between captures in periodic mode.
            jpeg_quality: JPEG compression quality (1-100). Lower = smaller files.
        """
        self.window_id = window_id
        self.window_name = window_name
        self.interval = interval
        self.jpeg_quality = jpeg_quality

        self._geometry: Optional[WindowGeometry] = None
        self._latest_frame: Optional[bytes] = None
        self._frame_lock = threading.Lock()

        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def find_window(self) -> bool:
        """Find the window ID based on process or name if not already set.

        Returns:
            True if a window was found or already set.
        """
        if self.window_id:
            # Validate if it's a valid hex or dec string and refresh
            return self.refresh_geometry()

        # Technique 1: Try finding by PID (most reliable for Chiaki/AppImage)
        try:
            # Look for Chiaki's AppRun.wrapped or similar process
            ps_result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if ps_result.returncode == 0:
                # Target the specific Chiaki AppRun process which usually has high CPU/Memory
                for line in ps_result.stdout.splitlines():
                    if "AppRun.wrapped" in line or "chiaki-ng" in line:
                        parts = line.split()
                        if len(parts) > 1:
                            pid = parts[1]
                            # Try xdotool search --pid
                            xd_result = subprocess.run(
                                ["xdotool", "search", "--pid", pid],
                                capture_output=True,
                                text=True,
                                timeout=5,
                            )
                            if xd_result.returncode == 0 and xd_result.stdout.strip():
                                ids = xd_result.stdout.strip().split("\n")
                                self.window_id = ids[0] # Usually the first one for PID search
                                logger.debug("Found window ID %s via PID %s", self.window_id, pid)
                                if self.refresh_geometry():
                                    return True
        except Exception as e:
            logger.debug("PID search failed: %s", e)

        if not self.window_name:
            logger.error("No window_id, window_name, and PID discovery failed")
            return False

        logger.debug("Searching for window with name: %s", self.window_name)
        try:
            # Try searching by name exactly first
            result = subprocess.run(
                ["xdotool", "search", "--name", self.window_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Take the last one, as it's often the actual UI window vs hidden ones
                ids = result.stdout.strip().split("\n")
                self.window_id = ids[-1]
                logger.debug("Found window ID: %s", self.window_id)
                return self.refresh_geometry()

            # Fallback: search by class
            result = subprocess.run(
                ["xdotool", "search", "--class", self.window_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                ids = result.stdout.strip().split("\n")
                self.window_id = ids[-1]
                logger.debug("Found window ID by class: %s", self.window_id)
                return self.refresh_geometry()

        except Exception as e:
            logger.error("Error searching for window: %s", e)

        return False

    def refresh_geometry(self) -> bool:
        """Refresh window geometry from the window ID.

        Returns:
            True if geometry was successfully retrieved.
        """
        if not self.window_id:
            return self.find_window()
        return self._update_geometry(self.window_id)

    def _update_geometry(self, window_id: str) -> bool:
        """Get window position and size via xdotool getwindowgeometry.

        Args:
            window_id: X11 window ID string.

        Returns:
            True if geometry was successfully retrieved.
        """
        try:
            result = subprocess.run(
                ["xdotool", "getwindowgeometry", window_id],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            logger.error("xdotool getwindowgeometry timed out")
            return False

        if result.returncode != 0:
            logger.warning(
                "Failed to get geometry for window %s: %s",
                window_id,
                result.stderr.strip(),
            )
            self._geometry = None
            return False

        # Parse output like:
        #   Window 12345678
        #   Position: 100,200 (screen: 0)
        #   Geometry: 800x600
        output = result.stdout
        pos_match = re.search(r"Position:\s*(\d+),(\d+)", output)
        geo_match = re.search(r"Geometry:\s*(\d+)x(\d+)", output)

        if not pos_match or not geo_match:
            logger.error("Could not parse xdotool geometry output:\n%s", output)
            self._geometry = None
            return False

        self._geometry = WindowGeometry(
            window_id=window_id,
            x=int(pos_match.group(1)),
            y=int(pos_match.group(2)),
            width=int(geo_match.group(1)),
            height=int(geo_match.group(2)),
        )
        logger.debug(
            "Window %s at (%d,%d) size %dx%d",
            self.window_id,
            self._geometry.x,
            self._geometry.y,
            self._geometry.width,
            self._geometry.height,
        )
        return True

    def capture_once(self) -> Optional[bytes]:
        """Capture one screenshot of the target window.

        Refreshes geometry each time to handle moved/resized windows.

        Returns:
            JPEG bytes of the captured region, or None on failure.
        """
        if not self.refresh_geometry():
            return None

        geo = self._geometry
        if geo is None:
            return None

        if geo.width <= 0 or geo.height <= 0:
            logger.warning(
                "Window has invalid dimensions: %dx%d", geo.width, geo.height
            )
            return None

        try:
            with mss.mss() as sct:
                # Clamp capture region to virtual screen bounds
                screen = sct.monitors[0]  # monitors[0] is the combined virtual screen
                left = max(geo.x, screen["left"])
                top = max(geo.y, screen["top"])
                right = min(geo.x + geo.width, screen["left"] + screen["width"])
                bottom = min(geo.y + geo.height, screen["top"] + screen["height"])
                clamped_w = right - left
                clamped_h = bottom - top

                if clamped_w <= 0 or clamped_h <= 0:
                    logger.warning(
                        "Window region outside screen bounds: (%d,%d) %dx%d vs screen %s",
                        geo.x,
                        geo.y,
                        geo.width,
                        geo.height,
                        screen,
                    )
                    return None

                monitor = {
                    "left": left,
                    "top": top,
                    "width": clamped_w,
                    "height": clamped_h,
                }
                screenshot = sct.grab(monitor)
        except Exception:
            logger.exception("mss screen grab failed for window %s", self.window_id)
            return None

        # Convert to JPEG bytes via Pillow
        try:
            img = Image.frombytes(
                "RGB", screenshot.size, screenshot.bgra, "raw", "BGRX"
            )
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.jpeg_quality)
            jpeg_bytes = buf.getvalue()
        except Exception:
            logger.exception("Image conversion to JPEG failed")
            return None

        with self._frame_lock:
            self._latest_frame = jpeg_bytes

        logger.debug(
            "Captured %d bytes JPEG from window %s", len(jpeg_bytes), self.window_id
        )
        return jpeg_bytes

    def get_latest_frame(self) -> Optional[bytes]:
        """Get the most recently captured frame (thread-safe).

        Returns:
            JPEG bytes of the last successful capture, or None if no capture yet.
        """
        with self._frame_lock:
            return self._latest_frame

    def start(self) -> None:
        """Start periodic capture in a background daemon thread."""
        if self._capture_thread is not None and self._capture_thread.is_alive():
            logger.warning("Capture thread already running")
            return

        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"capture-{self.window_id}",
            daemon=True,
        )
        self._capture_thread.start()
        logger.debug(
            "Started periodic capture for window %s every %.1fs",
            self.window_id,
            self.interval,
        )

    def stop(self) -> None:
        """Stop periodic capture and wait for thread to finish."""
        if self._capture_thread is None or not self._capture_thread.is_alive():
            return

        self._stop_event.set()
        self._capture_thread.join(timeout=self.interval + 2)
        if self._capture_thread.is_alive():
            logger.warning("Capture thread did not stop cleanly")
        else:
            logger.debug("Capture thread stopped")
        self._capture_thread = None

    def _capture_loop(self) -> None:
        """Internal loop for periodic capture. Runs in background thread."""
        while not self._stop_event.is_set():
            try:
                self.capture_once()
            except Exception:
                logger.exception("Unexpected error in capture loop")
            self._stop_event.wait(timeout=self.interval)

    @property
    def geometry(self) -> Optional[WindowGeometry]:
        """Current window geometry, or None if window not found."""
        return self._geometry


def main() -> None:
    """CLI entry point for testing screen capture."""
    parser = argparse.ArgumentParser(
        description="Capture screenshots of a window by ID or name.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--window-id",
        help="X11 window ID (hex or decimal) to capture.",
    )
    group.add_argument(
        "--window-name",
        help="Window name/title to search for via xdotool.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of captures to take (default: 10).",
    )
    parser.add_argument(
        "--output",
        default="/tmp/captures/",
        help="Output directory for captured images (default: /tmp/captures/).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Seconds between captures (default: 3).",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        help="JPEG quality 1-100 (default: 85).",
    )
    args = parser.parse_args()

    setup_logging("INFO")

    os.makedirs(args.output, exist_ok=True)

    service = CaptureService(
        window_id=args.window_id,
        window_name=args.window_name,
        interval=args.interval,
        jpeg_quality=args.quality,
    )

    if args.window_id:
        logger.info("Capturing window ID %s...", args.window_id)
    else:
        logger.info("Searching for window '%s'...", args.window_name)

    if not service.find_window():
        logger.error("Could not locate window. Exiting.")
        return

    geo = service.geometry
    if geo:
        logger.info(
            "Found window at (%d,%d) size %dx%d", geo.x, geo.y, geo.width, geo.height
        )

    captured = 0
    for i in range(1, args.count + 1):
        frame = service.capture_once()
        if frame is not None:
            filename = f"capture_{i:03d}.jpg"
            filepath = os.path.join(args.output, filename)
            with open(filepath, "wb") as f:
                f.write(frame)
            captured += 1
            logger.info("Saved %s (%d bytes)", filepath, len(frame))
        else:
            logger.warning("Capture %d/%d failed", i, args.count)

        if i < args.count:
            time.sleep(args.interval)

    logger.info("Done. Captured %d/%d frames to %s", captured, args.count, args.output)


if __name__ == "__main__":
    main()
