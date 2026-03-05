"""Tunic-specific frame detectors using OpenCV."""

import logging
import time
from typing import Optional

import cv2
import numpy as np

from .engine import DetectorEvent

logger = logging.getLogger(__name__)


class TunicDeathDetector:
    """Detects player death in Tunic by analyzing screen color patterns.

    Death indicators:
    - Screen turns predominantly black (death animation)
    - Bottom-left corner shows red vignette (critical health/hit)
    """

    id = "tunic-death"
    game_id = "tunic"

    # Thresholds
    BLACK_SCREEN_THRESHOLD = 15  # Mean RGB below this = black screen
    RED_CORNER_MIN_R = 100  # Minimum red channel value
    RED_CORNER_R_TO_G_RATIO = 2.0  # R must be > G * ratio
    RED_CORNER_R_TO_B_RATIO = 2.0  # R must be > B * ratio
    CORNER_SIZE_RATIO = 0.1  # Corner is 10% of width

    def probe(self, frame_bytes: bytes) -> Optional[DetectorEvent]:
        """Analyze frame for death indicators.

        Args:
            frame_bytes: Raw image bytes (JPEG/PNG).

        Returns:
            DetectorEvent if death is detected, None otherwise.
        """
        try:
            # Decode frame
            nparr = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None

            height, width = img.shape[:2]

            # Check 1: Black screen (death animation)
            screen_mean = img.mean()
            is_screen_black = screen_mean < self.BLACK_SCREEN_THRESHOLD

            # Check 2: Red corner (critical health/hit)
            corner_size = int(width * self.CORNER_SIZE_RATIO)
            corner_region = img[height - corner_size : height, 0:corner_size]
            corner_mean = corner_region.mean(axis=(0, 1))  # [B, G, R] in OpenCV
            b, g, r = corner_mean

            is_corner_red = (
                r > self.RED_CORNER_MIN_R
                and r > g * self.RED_CORNER_R_TO_G_RATIO
                and r > b * self.RED_CORNER_R_TO_B_RATIO
            )

            if is_screen_black or is_corner_red:
                return DetectorEvent(
                    id=f"death-{int(time.time() * 1000)}",
                    game_id=self.game_id,
                    timestamp=time.time(),
                    type="player_death",
                    data={
                        "reason": "black_screen" if is_screen_black else "red_corner",
                        "is_screen_black": is_screen_black,
                        "is_corner_red": is_corner_red,
                        "corner_r": float(r),
                        "screen_mean": float(screen_mean),
                    },
                    confidence=0.95 if is_screen_black else 0.8,
                )

            return None

        except Exception as e:
            logger.error("TunicDeathDetector error: %s", e, exc_info=True)
            return None


class TunicHealthDetector:
    """Detects player health state in Tunic by analyzing HUD elements.

    Health indicators:
    - Health bar in top-left corner (red bar)
    - Red corner vignette for critical health
    """

    id = "tunic-health"
    game_id = "tunic"

    # Thresholds for pixel checks (scaled to 1920x1080 reference)
    REFERENCE_WIDTH = 1920
    REFERENCE_HEIGHT = 1080
    HEALTH_CHECK_X = 80  # X coordinate from left
    HEALTH_FULL_Y = 110  # Y coordinate from top for full health check
    HEALTH_LOW_Y = 80  # Y coordinate from top for low health check

    # Pixel intensity thresholds
    LOW_INTENSITY_THRESHOLD = 50  # R and G below this
    BLUE_MAX_THRESHOLD = 130  # B below this
    SATURATION_DIFF_THRESHOLD = 30  # R-G difference below this

    # Red corner thresholds
    RED_CORNER_MIN_R = 100
    RED_CORNER_R_TO_G_RATIO = 1.5
    RED_CORNER_R_TO_B_RATIO = 1.5
    CORNER_SIZE_RATIO = 0.05

    def probe(self, frame_bytes: bytes) -> Optional[DetectorEvent]:
        """Analyze frame for health indicators.

        Args:
            frame_bytes: Raw image bytes (JPEG/PNG).

        Returns:
            DetectorEvent with health state, or None on error.
        """
        try:
            # Decode frame
            nparr = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None

            height, width = img.shape[:2]

            # Scale coordinates to actual resolution
            scale_x = width / self.REFERENCE_WIDTH
            scale_y = height / self.REFERENCE_HEIGHT

            x_left = int(self.HEALTH_CHECK_X * scale_x)
            y_full = int(self.HEALTH_FULL_Y * scale_y)
            y_low = int(self.HEALTH_LOW_Y * scale_y)

            # Check pixels (OpenCV uses BGR)
            def is_pixel_black(x: int, y: int) -> bool:
                """Check if pixel at (x, y) is dark/black."""
                x = max(0, min(width - 1, x))
                y = max(0, min(height - 1, y))
                b, g, r = img[y, x]
                is_low_intensity = (
                    r < self.LOW_INTENSITY_THRESHOLD
                    and g < self.LOW_INTENSITY_THRESHOLD
                )
                is_low_saturation = (
                    abs(int(r) - int(g)) < self.SATURATION_DIFF_THRESHOLD
                )
                return is_low_intensity and is_low_saturation

            is_full_health_pixel_black = is_pixel_black(x_left, y_full)
            is_low_health_pixel_black = is_pixel_black(x_left, y_low)

            # Check red corner for critical health
            corner_size = int(width * self.CORNER_SIZE_RATIO)
            corner_region = img[height - corner_size : height, 0:corner_size]
            corner_mean = corner_region.mean(axis=(0, 1))
            b, g, r = corner_mean

            is_corner_red = (
                r > self.RED_CORNER_MIN_R
                and r > g * self.RED_CORNER_R_TO_G_RATIO
                and r > b * self.RED_CORNER_R_TO_B_RATIO
            )

            # Determine health state
            health_state = "good"
            if is_corner_red:
                health_state = "critical"
            elif is_low_health_pixel_black:
                health_state = "low"

            return DetectorEvent(
                id=f"health-{int(time.time() * 1000)}",
                game_id=self.game_id,
                timestamp=time.time(),
                type="player_health",
                data={
                    "state": health_state,
                    "is_full_health_pixel_black": is_full_health_pixel_black,
                    "is_low_health_pixel_black": is_low_health_pixel_black,
                    "is_corner_red": is_corner_red,
                    "corner_r": float(r),
                    "metadata": {"width": width, "height": height},
                    "scaled_coords": {
                        "x_left": x_left,
                        "y_full": y_full,
                        "y_low": y_low,
                    },
                },
                confidence=0.9,
            )

        except Exception as e:
            logger.error("TunicHealthDetector error: %s", e, exc_info=True)
            return None
