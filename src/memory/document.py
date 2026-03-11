"""Memory document model for conversation history."""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryDocument:
    """Represents a summarized memory from conversation history.

    This document captures useful context from recent exchanges, including
    progress, discoveries, goals, and assistant guidance.
    """

    game_id: str
    session_id: str
    timestamp: float = field(default_factory=time.time)
    summary: str = ""
    goals: list[str] = field(default_factory=list)
    discoveries: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    locations_visited: list[str] = field(default_factory=list)
    items_found: list[str] = field(default_factory=list)
    turn_count: int = 0
    source_turns: list[int] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Convert memory document to markdown format for storage.

        Returns:
            Markdown-formatted memory document.
        """
        lines = [
            f"# Memory: {self.game_id} - Session {self.session_id}",
            "",
            f"**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}",
            f"**Turn Count:** {self.turn_count}",
            "",
        ]

        if self.summary:
            lines.extend(
                [
                    "## Summary",
                    "",
                    self.summary,
                    "",
                ]
            )

        if self.goals:
            lines.extend(
                [
                    "## Current Goals",
                    "",
                ]
            )
            for goal in self.goals:
                lines.append(f"- {goal}")
            lines.append("")

        if self.discoveries:
            lines.extend(
                [
                    "## Discoveries",
                    "",
                ]
            )
            for discovery in self.discoveries:
                lines.append(f"- {discovery}")
            lines.append("")

        if self.locations_visited:
            lines.extend(
                [
                    "## Locations Visited",
                    "",
                ]
            )
            for location in self.locations_visited:
                lines.append(f"- {location}")
            lines.append("")

        if self.items_found:
            lines.extend(
                [
                    "## Items Found",
                    "",
                ]
            )
            for item in self.items_found:
                lines.append(f"- {item}")
            lines.append("")

        if self.open_questions:
            lines.extend(
                [
                    "## Open Questions",
                    "",
                ]
            )
            for question in self.open_questions:
                lines.append(f"- {question}")
            lines.append("")

        if self.source_turns:
            lines.extend(
                [
                    f"**Source Turns:** {', '.join(str(t) for t in self.source_turns)}",
                    "",
                ]
            )

        return "\n".join(lines)

    @classmethod
    def from_markdown(
        cls, markdown: str, game_id: str, session_id: str
    ) -> Optional["MemoryDocument"]:
        """Parse a markdown memory document.

        Args:
            markdown: Markdown-formatted memory document.
            game_id: Game identifier.
            session_id: Session identifier.

        Returns:
            MemoryDocument instance or None if parsing fails.
        """
        doc = cls(game_id=game_id, session_id=session_id)

        lines = markdown.split("\n")
        current_section = None
        summary_lines = []

        for line in lines:
            line = line.strip()

            if line.startswith("**Timestamp:**"):
                continue
            elif line.startswith("**Turn Count:**"):
                try:
                    doc.turn_count = int(line.split(":")[-1].strip())
                except ValueError:
                    pass
            elif line.startswith("**Source Turns:**"):
                try:
                    turns_str = line.split(":")[-1].strip()
                    doc.source_turns = [
                        int(t.strip()) for t in turns_str.split(",") if t.strip()
                    ]
                except ValueError:
                    pass
            elif line.startswith("## Summary"):
                current_section = "summary"
            elif line.startswith("## Current Goals"):
                current_section = "goals"
            elif line.startswith("## Discoveries"):
                current_section = "discoveries"
            elif line.startswith("## Locations Visited"):
                current_section = "locations"
            elif line.startswith("## Items Found"):
                current_section = "items"
            elif line.startswith("## Open Questions"):
                current_section = "questions"
            elif line.startswith("#"):
                current_section = None
            elif line.startswith("- ") and current_section:
                item = line[2:].strip()
                if current_section == "goals":
                    doc.goals.append(item)
                elif current_section == "discoveries":
                    doc.discoveries.append(item)
                elif current_section == "locations":
                    doc.locations_visited.append(item)
                elif current_section == "items":
                    doc.items_found.append(item)
                elif current_section == "questions":
                    doc.open_questions.append(item)
            elif current_section == "summary" and line:
                summary_lines.append(line)

        if summary_lines:
            doc.summary = "\n".join(summary_lines)

        return doc
