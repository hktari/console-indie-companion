"""Background research job manager for async web research."""

import asyncio
import logging
import uuid
from collections.abc import Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from src.agent.models import EvidenceBundle, ResearchResult

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Research job status states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class ResearchJob:
    """Represents a single background research task."""

    job_id: str
    query: str
    transcript: str
    game_id: str
    scene: Optional[dict[str, Any]] = None
    status: JobStatus = JobStatus.PENDING
    result: Optional[EvidenceBundle] = None
    research_result: Optional[ResearchResult] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: tuple[int, int] = (0, 0)
    on_complete: Optional[Callable[["ResearchJob"], None]] = None
    on_progress: Optional[Callable[[int, int], None]] = None
    _task: Optional[asyncio.Task] = field(default=None, repr=False)

    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration if completed."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class ResearchJobConfig:
    """Configuration for research job manager."""

    max_concurrent: int = 3
    timeout_seconds: float = 60.0
    enable_notifications: bool = True


class ResearchJobManager:
    """Manages background research jobs with async execution."""

    def __init__(
        self,
        research_executor: Callable[
            [str, str, Optional[int], Optional[Callable[[int, int], None]]],
            Awaitable[ResearchResult],
        ],
        config: Optional[ResearchJobConfig] = None,
    ):
        """Initialize the research job manager.

        Args:
            research_executor: Async function that performs iterative research (query, game_id, max_tool_calls, on_progress) -> ResearchResult
            config: Job manager configuration
        """
        self._research_executor = research_executor
        self._config = config or ResearchJobConfig()
        self._jobs: dict[str, ResearchJob] = {}
        self._queue: asyncio.Queue[ResearchJob] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._workers: list[asyncio.Task] = []
        self._running = False

        logger.info(
            "ResearchJobManager initialized (max_concurrent=%d, timeout=%ds)",
            self._config.max_concurrent,
            self._config.timeout_seconds,
        )

    async def start(self) -> None:
        """Start background worker tasks."""
        if self._running:
            logger.warning("ResearchJobManager already running")
            return

        self._running = True
        logger.info("Starting %d research workers", self._config.max_concurrent)

        for i in range(self._config.max_concurrent):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)

    async def stop(self) -> None:
        """Stop all workers and cancel pending jobs."""
        if not self._running:
            return

        logger.info("Stopping ResearchJobManager")
        self._running = False

        # Cancel all pending jobs
        async with self._lock:
            for job in self._jobs.values():
                if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
                    await self._cancel_job_internal(job)

        # Wait for workers to finish
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        logger.info("ResearchJobManager stopped")

    async def queue_research(
        self,
        query: str,
        transcript: str,
        game_id: str,
        scene: Optional[dict[str, Any]] = None,
        on_complete: Optional[Callable[[ResearchJob], None]] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """Queue a new research job.

        Args:
            query: Research query
            transcript: Full user transcript
            game_id: Game identifier
            scene: Optional scene context
            on_complete: Callback invoked when job completes
            on_progress: Callback invoked on research progress (step, max_steps)

        Returns:
            Job ID for tracking
        """
        job_id = str(uuid.uuid4())
        job = ResearchJob(
            job_id=job_id,
            query=query,
            transcript=transcript,
            game_id=game_id,
            scene=scene,
            on_complete=on_complete,
            on_progress=on_progress,
        )

        async with self._lock:
            self._jobs[job_id] = job

        await self._queue.put(job)
        logger.info("Queued research job %s: '%s'", job_id[:8], query[:50])

        return job_id

    async def get_job(self, job_id: str) -> Optional[ResearchJob]:
        """Get job by ID.

        Args:
            job_id: Job identifier

        Returns:
            ResearchJob if found, None otherwise
        """
        async with self._lock:
            return self._jobs.get(job_id)

    async def list_active_jobs(self) -> list[ResearchJob]:
        """List all pending or running jobs.

        Returns:
            List of active jobs
        """
        async with self._lock:
            return [
                job
                for job in self._jobs.values()
                if job.status in (JobStatus.PENDING, JobStatus.RUNNING)
            ]

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running or pending job.

        Args:
            job_id: Job identifier

        Returns:
            True if cancelled, False if not found or already completed
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
                return False
            await self._cancel_job_internal(job)
            return True

    async def has_active_research(self) -> bool:
        """Check if any jobs are pending or running.

        Returns:
            True if active jobs exist
        """
        async with self._lock:
            return any(
                job.status in (JobStatus.PENDING, JobStatus.RUNNING)
                for job in self._jobs.values()
            )

    async def _worker(self, worker_id: int) -> None:
        """Background worker that processes jobs from queue.

        Args:
            worker_id: Worker identifier for logging
        """
        logger.info("Research worker %d started", worker_id)

        while self._running:
            try:
                # Wait for next job with timeout to allow graceful shutdown
                job = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Worker %d queue error: %s", worker_id, e)
                continue

            try:
                await self._execute_job(job, worker_id)
            except Exception as e:
                logger.error(
                    "Worker %d failed to execute job %s: %s",
                    worker_id,
                    job.job_id[:8],
                    e,
                    exc_info=True,
                )
                await self._mark_error(job, str(e))
            finally:
                self._queue.task_done()

        logger.info("Research worker %d stopped", worker_id)

    async def _execute_job(self, job: ResearchJob, worker_id: int) -> None:
        """Execute a research job.

        Args:
            job: Job to execute
            worker_id: Worker identifier for logging
        """
        async with self._lock:
            if job.status != JobStatus.PENDING:
                logger.warning("Job %s already processed, skipping", job.job_id[:8])
                return
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()

        logger.info(
            "Worker %d executing job %s: '%s'",
            worker_id,
            job.job_id[:8],
            job.query[:50],
        )

        try:
            # Create progress callback that updates job progress
            def progress_callback(step: int, max_steps: int) -> None:
                async def update_progress():
                    async with self._lock:
                        job.progress = (step, max_steps)
                    if job.on_progress:
                        try:
                            if asyncio.iscoroutinefunction(job.on_progress):
                                await job.on_progress(step, max_steps)
                            else:
                                job.on_progress(step, max_steps)
                        except Exception as e:
                            logger.error("Progress callback failed: %s", e)

                # Schedule the async update
                asyncio.create_task(update_progress())

            # Execute iterative research with timeout
            research_result = await asyncio.wait_for(
                self._research_executor(
                    job.query, job.game_id, None, progress_callback
                ),
                timeout=self._config.timeout_seconds,
            )

            # Create evidence bundle with research result
            evidence = EvidenceBundle(
                research_memo=research_result.memo,
                sources=research_result.sources_used,
                metadata={
                    "job_id": job.job_id,
                    "steps_taken": research_result.steps_taken,
                    "confidence": research_result.confidence,
                },
            )

            async with self._lock:
                job.status = JobStatus.COMPLETED
                job.result = evidence
                job.research_result = research_result
                job.completed_at = datetime.now()

            duration = job.duration_seconds()
            logger.info(
                "Job %s completed in %.2fs (%d steps, confidence=%.2f)",
                job.job_id[:8],
                duration if duration else 0,
                research_result.steps_taken,
                research_result.confidence,
            )

            # Invoke completion callback if configured
            if self._config.enable_notifications and job.on_complete:
                try:
                    if asyncio.iscoroutinefunction(job.on_complete):
                        await job.on_complete(job)
                    else:
                        job.on_complete(job)
                except Exception as e:
                    logger.error(
                        "Job %s completion callback failed: %s",
                        job.job_id[:8],
                        e,
                        exc_info=True,
                    )

        except asyncio.TimeoutError:
            error_msg = f"Research timed out after {self._config.timeout_seconds}s"
            logger.warning("Job %s: %s", job.job_id[:8], error_msg)
            await self._mark_error(job, error_msg)

        except Exception as e:
            logger.error("Job %s failed: %s", job.job_id[:8], e, exc_info=True)
            await self._mark_error(job, str(e))

    async def _mark_error(self, job: ResearchJob, error: str) -> None:
        """Mark job as failed.

        Args:
            job: Job to mark
            error: Error message
        """
        async with self._lock:
            job.status = JobStatus.ERROR
            job.error = error
            job.completed_at = datetime.now()

    async def _cancel_job_internal(self, job: ResearchJob) -> None:
        """Cancel a job (internal, assumes lock held).

        Args:
            job: Job to cancel
        """
        if job._task and not job._task.done():
            job._task.cancel()

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now()
        logger.info("Cancelled job %s", job.job_id[:8])
