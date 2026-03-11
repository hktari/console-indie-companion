"""Data models for agent decision and evidence artifacts."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from src.rag.orchestrator import RetrievalResult


class RouteType(str, Enum):
    """Routing decision types."""

    DIRECT_ANSWER = "direct_answer"
    KNOWLEDGE_BASE = "knowledge_base"
    MEMORY_SEARCH = "memory_search"
    WEB_RESEARCH = "web_research"
    COMBINED = "combined"


@dataclass
class AgentDecision:
    """Structured decision from the planner."""

    route: RouteType
    reasoning: str
    confidence: float
    tools_to_call: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceBundle:
    """Collected evidence from various sources with provenance."""

    kb_results: list[RetrievalResult] = field(default_factory=list)
    memory_results: list[RetrievalResult] = field(default_factory=list)
    research_memo: Optional[str] = None
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_all_results(self) -> list[RetrievalResult]:
        """Get all retrieval results combined."""
        return self.kb_results + self.memory_results

    def has_evidence(self) -> bool:
        """Check if any evidence was collected."""
        return bool(self.kb_results or self.memory_results or self.research_memo)
