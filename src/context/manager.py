"""Context Manager — bridges VLM scene descriptions and RAG results into the voice session."""

import logging
import threading
from collections import deque
from typing import Optional

from src.prompts.tunic_companion import CONTEXT_UPDATE_TEMPLATE
from src.rag.query import query_tunic_knowledge

logger = logging.getLogger(__name__)


class ContextManager:
    """Maintains a rolling buffer of scene descriptions and flushes formatted
    context updates into a VoiceSession.

    Thread-safe: the capture thread calls ``update_scene()`` while the main
    thread reads via ``get_pending_context()`` / ``flush_to_voice()``.
    """

    def __init__(self, max_history: int = 5) -> None:
        """Initialize with rolling buffer size.

        Args:
            max_history: Maximum number of scene descriptions to retain.
        """
        self._max_history = max_history
        self._scenes: deque[dict] = deque(maxlen=max_history)
        self._unflushed_count: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Scene buffer
    # ------------------------------------------------------------------

    def update_scene(self, scene_description: dict) -> None:
        """Add a new VLM scene description to the rolling buffer.

        Args:
            scene_description: Dict returned by ``SceneAnalyzer.analyze_screenshot()``.
        """
        with self._lock:
            self._scenes.append(scene_description)
            self._unflushed_count += 1
            logger.debug(
                "Scene buffered (%d unflushed): %s",
                self._unflushed_count,
                scene_description.get("description", "")[:80],
            )

    # ------------------------------------------------------------------
    # RAG integration
    # ------------------------------------------------------------------

    def get_rag_context(self, scene_description: dict) -> str:
        """Query the RAG knowledge base based on the current scene.

        Builds a query from location + activity + notable_items and returns
        the top-3 results joined as a single text block.

        Args:
            scene_description: Scene dict from the VLM.

        Returns:
            Formatted RAG results string, or empty string on failure.
        """
        def _normalize_field(field):
            """Convert field to string, handling lists."""
            if isinstance(field, list):
                return " ".join(str(x) for x in field if x)
            return str(field) if field else ""
        
        parts = [
            _normalize_field(scene_description.get("location", "")),
            _normalize_field(scene_description.get("activity", "")),
            _normalize_field(scene_description.get("notable_items", "")),
        ]
        query = " ".join(p for p in parts if p and p.lower() != "none")
        if not query:
            return ""

        try:
            results = query_tunic_knowledge(query, n_results=3)
            return "\n".join(results) if results else ""
        except Exception:
            logger.warning("RAG query failed for: %s", query, exc_info=True)
            return ""

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    def format_context_update(
        self, scene_description: dict, rag_results: list[str]
    ) -> str:
        """Format a context update string using the prompt template.

        Args:
            scene_description: Scene dict from the VLM.
            rag_results: List of relevant knowledge chunks from RAG.

        Returns:
            Formatted context string ready for injection.
        """
        rag_context = "\n".join(rag_results) if rag_results else "No additional context."

        return CONTEXT_UPDATE_TEMPLATE.format(
            description=scene_description.get("description", "unknown"),
            location=scene_description.get("location", "unknown"),
            activity=scene_description.get("activity", "unknown"),
            enemies=scene_description.get("enemies", "none"),
            health_status=scene_description.get("health_status", "unknown"),
            ui_elements=scene_description.get("ui_elements", "none"),
            notable_items=scene_description.get("notable_items", "none"),
            rag_context=rag_context,
        )

    # ------------------------------------------------------------------
    # Pending context
    # ------------------------------------------------------------------

    def get_recent_context(self, num_scenes_context: int, num_scenes_rag: int) -> str:
        """Get formatted context for recent scenes, including RAG results.

        Args:
            num_scenes_context: How many recent scenes to include in the description.
            num_scenes_rag: How many recent scenes to use for RAG queries.

        Returns:
            A formatted context string, or an empty string if no scenes are available.
        """
        with self._lock:
            if not self._scenes:
                return ""
            
            # Get scenes for context description
            scenes_for_desc = list(self._scenes)[-num_scenes_context:]
            
            # Get scenes for RAG query
            scenes_for_rag = list(self._scenes)[-num_scenes_rag:]

        # --- RAG Queries (outside lock) ---
        all_rag_results = set()
        for scene in scenes_for_rag:
            rag_text = self.get_rag_context(scene)
            if rag_text:
                all_rag_results.add(rag_text)
        
        # --- Formatting (outside lock) ---
        if not scenes_for_desc:
            return ""

        # For multiple scenes, we'll just use the description of the latest one
        # but note that multiple frames are being considered.
        latest_scene = scenes_for_desc[-1]
        
        # Create a combined description for the context block
        if len(scenes_for_desc) > 1:
            description = f"Over the last {len(scenes_for_desc)} moments, the player has been {latest_scene.get('activity', 'exploring')}. The most recent view is: {latest_scene.get('description', 'No description available.')}"
        else:
            description = latest_scene.get('description', 'No description available.')


        # Format the final update string
        rag_context = "\n".join(sorted(list(all_rag_results))) if all_rag_results else "No additional context."
        
        return CONTEXT_UPDATE_TEMPLATE.format(
            description=description,
            location=latest_scene.get("location", "unknown"),
            activity=latest_scene.get("activity", "unknown"),
            enemies=latest_scene.get("enemies", "none"),
            health_status=latest_scene.get("health_status", "unknown"),
            ui_elements=latest_scene.get("ui_elements", "none"),
            notable_items=latest_scene.get("notable_items", "none"),
            rag_context=rag_context,
        )

    async def flush_latest_to_voice(self, voice_session) -> bool:
        """Formats and injects only the most recent scene and its RAG results.

        Used for critical events where immediate context is key.

        Args:
            voice_session: A ``VoiceSession`` instance.

        Returns:
            True if context was injected, False otherwise.
        """
        with self._lock:
            if not self._scenes:
                return False
            scene = self._scenes[-1]

        rag_text = self.get_rag_context(scene)
        rag_results = [rag_text] if rag_text else []
        
        context = self.format_context_update(scene, rag_results)
        
        if context:
            await voice_session.inject_context(context)
            logger.debug("Flushed latest scene to voice session for critical event.")
            return True
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def scene_count(self) -> int:
        """Number of scenes currently in the buffer."""
        with self._lock:
            return len(self._scenes)

    @property
    def unflushed_count(self) -> int:
        """Number of scenes added since the last flush."""
        with self._lock:
            return self._unflushed_count
