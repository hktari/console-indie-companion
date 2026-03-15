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
    ITERATIVE_RESEARCH = "iterative_research"


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
    job_id: Optional[str] = None

    def get_all_results(self) -> list[RetrievalResult]:
        """Get all retrieval results combined."""
        return self.kb_results + self.memory_results

    def has_evidence(self) -> bool:
        """Check if any evidence was collected."""
        return bool(self.kb_results or self.memory_results or self.research_memo)


@dataclass
class ToolResult:
    """Result from a single tool execution."""

    tool_name: str
    query: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchResult:
    """Result from iterative research agent."""

    memo: str
    findings: list[ToolResult] = field(default_factory=list)
    steps_taken: int = 0
    confidence: float = 0.0
    sources_used: list[str] = field(default_factory=list)


@dataclass
class ResearchConfig:
    """Configuration for iterative research agent."""

    max_tool_calls: int = 5
    min_tool_calls: int = 2
    model: str = "o3-mini"
    synthesis_model: str = "gpt-4.1-mini"
    enable_progress_tts: bool = True
