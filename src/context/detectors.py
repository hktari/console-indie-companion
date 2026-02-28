"""Pluggable scene detectors for Identifying specific game events from VLM analysis."""

import logging
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


class SceneDetector(Protocol):
    """Protocol for pluggable scene detectors."""

    def detect(self, scene: dict[str, Any]) -> Optional[str]:
        """Analyze a scene and return a system event message if detected.

        Args:
            scene: The latest VLM scene description dictionary.

        Returns:
            A system event string (e.g., "[SYSTEM EVENT] ...") or None.
        """
        ...


class DeathDetector:
    """Detects when the player dies based on location and activity transitions."""

    def __init__(self) -> None:
        self._is_dying = False

    def detect(self, scene: dict[str, Any]) -> Optional[str]:
        """Detect death event.

        Death pattern:
        - Activity is "fighting"
        - Location is null (None)
        - Triggers once when transition to null occurs.
        - Resets when location becomes valid again.
        """
        location = scene.get("location")
        activity = scene.get("activity", "").lower()

        # If location is null and we are fighting, we are likely in the death animation/void
        if location is None or (isinstance(location, str) and location.lower() == "none"):
            if activity == "fighting" and not self._is_dying:
                self._is_dying = True
                logger.info("Death detected! (location=null, activity=fighting)")
                return "[SYSTEM EVENT] The player has died."
        else:
            # Reset when we have a valid location again (respawned)
            if self._is_dying:
                logger.info("Player respawned (location=%s)", location)
                self._is_dying = False

        return None
