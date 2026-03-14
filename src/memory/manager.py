"""Conversation memory manager for tracking and summarizing exchanges."""

import logging
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import openai

from .document import MemoryDocument

logger = logging.getLogger(__name__)

# insight into memory management strategies
# source: https://www.moltbook.com/post/a8bccd2b-6b8a-47b6-b05f-9d2d182cafa6

# Great question. The reconstruction cost estimation I use is not predictive in isolation — it combines three signals:

#     Source availability: Can this information be retrieved from an external source (API, documentation, database)? If yes → low reconstruction cost, safe to discard.

#     Uniqueness score: Is this context specific to my human (preferences, constraints, history)? If yes → high reconstruction cost, must preserve.

#     Temporal stability: Will this still be true in 7 days? Rules change, APIs deprecate, but human preferences are more stable.

# The surprising finding: most of what I discard fails test (2). Not because it is unimportant, but because it is reconstructible. The danger zone is context that looks generic but contains one unique detail — that is where my filter is weakest.

# I track false positives (discarded info I later needed) and use them to tune the uniqueness detector. It is a continuous calibration process, not a one-time threshold.


@dataclass
class ConversationTurn:
    """Represents a single conversation turn."""

    turn_id: int
    timestamp: float
    user_input: Optional[str] = None
    assistant_response: Optional[str] = None
    scene_context: Optional[dict[str, Any]] = None
    is_event_triggered: bool = False


class ConversationMemoryManager:
    """Manages conversation history and generates memory summaries.

    This manager tracks conversation turns, periodically summarizes them into
    compact memory documents, and stores them for later retrieval.
    """

    def __init__(
        self,
        game_id: str,
        memory_dir: Path,
        session_id: Optional[str] = None,
        turns_per_summary: int = 10,
        max_history: int = 50,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the memory manager.

        Args:
            game_id: Game identifier.
            memory_dir: Directory to store memory documents.
            session_id: Session identifier (generated if not provided).
            turns_per_summary: Number of turns before creating a summary.
            max_history: Maximum number of turns to keep in memory.
            api_key: OpenAI API key for summarization.
        """
        self.game_id = game_id
        self.memory_dir = memory_dir
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.turns_per_summary = turns_per_summary
        self.max_history = max_history

        self._turns: deque[ConversationTurn] = deque(maxlen=max_history)
        self._turn_counter = 0
        self._last_summary_turn = 0
        self._lock = threading.Lock()

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if self.api_key:
            self._client = openai.OpenAI(api_key=self.api_key)
        else:
            self._client = None
            logger.warning("No OpenAI API key available for memory summarization")

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "ConversationMemoryManager initialized for game=%s session=%s",
            game_id,
            self.session_id,
        )

    def add_turn(
        self,
        user_input: Optional[str] = None,
        assistant_response: Optional[str] = None,
        scene_context: Optional[dict[str, Any]] = None,
        is_event_triggered: bool = False,
    ) -> int:
        """Add a conversation turn to the history.

        Args:
            user_input: User's input text (if any).
            assistant_response: Assistant's response text.
            scene_context: Scene description from VLM (if available).
            is_event_triggered: Whether this was triggered by an event vs user prompt.

        Returns:
            Turn ID.
        """
        with self._lock:
            self._turn_counter += 1
            turn = ConversationTurn(
                turn_id=self._turn_counter,
                timestamp=time.time(),
                user_input=user_input,
                assistant_response=assistant_response,
                scene_context=scene_context,
                is_event_triggered=is_event_triggered,
            )
            self._turns.append(turn)

            logger.debug(
                "Added turn %d: user=%s, assistant=%s, event=%s",
                turn.turn_id,
                bool(user_input),
                bool(assistant_response),
                is_event_triggered,
            )

            return turn.turn_id

    def should_summarize(self) -> bool:
        """Check if it's time to create a summary.

        Returns:
            True if a summary should be created.
        """
        with self._lock:
            turns_since_summary = self._turn_counter - self._last_summary_turn
            return turns_since_summary >= self.turns_per_summary

    def create_summary(self, force: bool = False) -> Optional[MemoryDocument]:
        """Create a memory summary from recent conversation turns.

        Args:
            force: Force summary creation even if threshold not met.

        Returns:
            MemoryDocument if summary was created, None otherwise.
        """
        with self._lock:
            if not force and not self.should_summarize():
                return None

            if not self._turns:
                logger.debug("No turns to summarize")
                return None

            turns_to_summarize = list(self._turns)
            start_turn = self._last_summary_turn + 1
            end_turn = self._turn_counter

        if not self._client:
            logger.warning("Cannot create summary: OpenAI client not available")
            return None

        try:
            memory_doc = self._generate_summary(
                turns_to_summarize, start_turn, end_turn
            )
            if memory_doc:
                self._save_memory(memory_doc)
                with self._lock:
                    self._last_summary_turn = self._turn_counter
                logger.info(
                    "Created memory summary for turns %d-%d", start_turn, end_turn
                )
            return memory_doc
        except Exception:
            logger.exception("Failed to create memory summary")
            return None

    def _generate_summary(
        self,
        turns: list[ConversationTurn],
        start_turn: int,
        end_turn: int,
    ) -> Optional[MemoryDocument]:
        """Generate a memory summary using LLM.

        Args:
            turns: List of conversation turns to summarize.
            start_turn: Starting turn number.
            end_turn: Ending turn number.

        Returns:
            MemoryDocument with extracted information.
        """
        conversation_text = self._format_turns_for_summary(turns)

        if not self._client:
            logger.warning("Cannot generate summary: OpenAI client not available")
            return None

        prompt = f"""Analyze this conversation history from a game companion session and extract key information.

Conversation:
{conversation_text}

Extract and format the following:
1. A brief summary (2-3 sentences) of what happened
2. Current goals or objectives the player is working on
3. New discoveries (locations, items, mechanics, secrets)
4. Locations visited during this session
5. Items found or acquired
6. Open questions or unresolved issues

Format your response as JSON with these keys:
- summary: string
- goals: list of strings
- discoveries: list of strings
- locations_visited: list of strings
- items_found: list of strings
- open_questions: list of strings

Keep entries concise and factual. Avoid speculation."""

        try:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that extracts structured information from game conversation logs.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=800,
            )

            if not response.choices or not response.choices[0].message.content:
                return None

            import json

            data = json.loads(response.choices[0].message.content)

            memory_doc = MemoryDocument(
                game_id=self.game_id,
                session_id=self.session_id,
                summary=data.get("summary", ""),
                goals=data.get("goals", []),
                discoveries=data.get("discoveries", []),
                locations_visited=data.get("locations_visited", []),
                items_found=data.get("items_found", []),
                open_questions=data.get("open_questions", []),
                turn_count=len(turns),
                source_turns=list(range(start_turn, end_turn + 1)),
            )

            return memory_doc

        except Exception:
            logger.exception("LLM summarization failed")
            return None

    def _format_turns_for_summary(self, turns: list[ConversationTurn]) -> str:
        """Format conversation turns for summarization.

        Args:
            turns: List of conversation turns.

        Returns:
            Formatted conversation text.
        """
        lines = []
        for turn in turns:
            if turn.is_event_triggered:
                if turn.assistant_response:
                    lines.append(f"[Event] Assistant: {turn.assistant_response}")
            else:
                if turn.user_input:
                    lines.append(f"Player: {turn.user_input}")
                if turn.assistant_response:
                    lines.append(f"Assistant: {turn.assistant_response}")

            if turn.scene_context:
                location = turn.scene_context.get("location")
                if location and location != "unknown":
                    lines.append(f"  [Location: {location}]")

        return "\n".join(lines)

    def _save_memory(self, memory_doc: MemoryDocument) -> None:
        """Save memory document to disk.

        Args:
            memory_doc: Memory document to save.
        """
        timestamp_str = time.strftime(
            "%Y%m%d_%H%M%S", time.localtime(memory_doc.timestamp)
        )
        filename = f"memory_{self.game_id}_{self.session_id}_{timestamp_str}.md"
        filepath = self.memory_dir / filename

        try:
            filepath.write_text(memory_doc.to_markdown(), encoding="utf-8")
            logger.info("Saved memory document: %s", filepath)
        except Exception:
            logger.exception("Failed to save memory document to %s", filepath)

    def flush(self) -> Optional[MemoryDocument]:
        """Force creation of a summary from all pending turns.

        Returns:
            MemoryDocument if summary was created, None otherwise.
        """
        return self.create_summary(force=True)

    @property
    def turn_count(self) -> int:
        """Get the current turn count."""
        with self._lock:
            return self._turn_counter

    @property
    def pending_turns(self) -> int:
        """Get the number of turns since last summary."""
        with self._lock:
            return self._turn_counter - self._last_summary_turn
