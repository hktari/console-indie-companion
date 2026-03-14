"""Performance measurement utilities for identifying bottlenecks."""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TimingStats:
    """Statistics for a timed operation."""

    name: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    last_time: float = 0.0

    @property
    def avg_time(self) -> float:
        """Average time per operation."""
        return self.total_time / self.count if self.count > 0 else 0.0

    def update(self, duration: float) -> None:
        """Update statistics with a new timing."""
        self.count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.last_time = duration

    def __str__(self) -> str:
        """Format statistics as a readable string."""
        if self.count == 0:
            return f"{self.name}: no data"
        return (
            f"{self.name}: "
            f"count={self.count}, "
            f"avg={self.avg_time:.3f}s, "
            f"min={self.min_time:.3f}s, "
            f"max={self.max_time:.3f}s, "
            f"last={self.last_time:.3f}s"
        )


class PerformanceTracker:
    """Tracks performance metrics for various operations."""

    def __init__(self) -> None:
        self._stats: dict[str, TimingStats] = {}
        self._enabled = True

    def enable(self) -> None:
        """Enable performance tracking."""
        self._enabled = True

    def disable(self) -> None:
        """Disable performance tracking."""
        self._enabled = False

    @contextmanager
    def measure(self, operation_name: str, log_threshold: Optional[float] = None):
        """Context manager to measure operation duration.

        Args:
            operation_name: Name of the operation being measured
            log_threshold: If set, log a warning if duration exceeds this threshold (in seconds)

        Example:
            with perf_tracker.measure("knowledge_retrieval"):
                results = search_knowledge_base(query)
        """
        if not self._enabled:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time

            if operation_name not in self._stats:
                self._stats[operation_name] = TimingStats(name=operation_name)

            self._stats[operation_name].update(duration)

            if log_threshold and duration > log_threshold:
                logger.warning(
                    "⚠️  SLOW: %s took %.3fs (threshold: %.3fs)",
                    operation_name,
                    duration,
                    log_threshold,
                )
            else:
                logger.debug("⏱️  %s: %.3fs", operation_name, duration)

    def get_stats(self, operation_name: str) -> Optional[TimingStats]:
        """Get statistics for a specific operation."""
        return self._stats.get(operation_name)

    def get_all_stats(self) -> dict[str, TimingStats]:
        """Get all tracked statistics."""
        return self._stats.copy()

    def get_summary(self) -> str:
        """Get a formatted summary of all statistics."""
        if not self._stats:
            return "No performance data collected"

        lines = ["=== Performance Summary ==="]

        # Sort by total time (descending) to show biggest time consumers first
        sorted_stats = sorted(
            self._stats.values(), key=lambda s: s.total_time, reverse=True
        )

        for stat in sorted_stats:
            lines.append(str(stat))

        # Calculate total time across all operations
        total_time = sum(s.total_time for s in self._stats.values())
        lines.append(f"\nTotal measured time: {total_time:.3f}s")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all statistics."""
        self._stats.clear()

    def log_summary(self) -> None:
        """Log the performance summary."""
        logger.info("\n" + self.get_summary())


# Global performance tracker instance
_global_tracker: Optional[PerformanceTracker] = None


def get_performance_tracker() -> PerformanceTracker:
    """Get the global performance tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PerformanceTracker()
    return _global_tracker
