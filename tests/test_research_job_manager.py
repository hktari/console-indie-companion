"""Tests for ResearchJobManager."""

import asyncio

import pytest
import pytest_asyncio

from src.agent.job_manager import (
    JobStatus,
    ResearchJob,
    ResearchJobConfig,
    ResearchJobManager,
)


@pytest.fixture
def mock_research_executor():
    """Create a mock research executor."""

    async def executor(query: str, game_id: str) -> str:
        await asyncio.sleep(0.1)
        return f"Research result for: {query}"

    return executor


@pytest_asyncio.fixture
async def job_manager(mock_research_executor):
    """Create and start a job manager."""
    config = ResearchJobConfig(
        max_concurrent=2,
        timeout_seconds=5.0,
        enable_notifications=True,
    )
    manager = ResearchJobManager(
        research_executor=mock_research_executor,
        config=config,
    )
    await manager.start()
    yield manager
    await manager.stop()


@pytest.mark.asyncio
async def test_job_lifecycle(job_manager):
    """Test basic job lifecycle: queue -> running -> completed."""
    job_id = await job_manager.queue_research(
        query="test query",
        transcript="user said test",
        game_id="tunic",
    )

    assert job_id is not None

    # Job should start as pending
    job = await job_manager.get_job(job_id)
    assert job is not None
    assert job.status in (JobStatus.PENDING, JobStatus.RUNNING)

    # Wait for completion
    await asyncio.sleep(0.3)

    job = await job_manager.get_job(job_id)
    assert job.status == JobStatus.COMPLETED
    assert job.result is not None
    assert job.result.research_memo == f"Research result for: {job.query}"


@pytest.mark.asyncio
async def test_concurrent_jobs(job_manager):
    """Test multiple concurrent jobs."""
    job_ids = []
    for i in range(3):
        job_id = await job_manager.queue_research(
            query=f"query {i}",
            transcript=f"transcript {i}",
            game_id="tunic",
        )
        job_ids.append(job_id)

    # Wait for all to complete
    await asyncio.sleep(0.5)

    for job_id in job_ids:
        job = await job_manager.get_job(job_id)
        assert job.status == JobStatus.COMPLETED
        assert job.result is not None


@pytest.mark.asyncio
async def test_job_timeout():
    """Test job timeout handling."""

    async def slow_executor(query: str, game_id: str) -> str:
        await asyncio.sleep(10)
        return "should timeout"

    config = ResearchJobConfig(
        max_concurrent=1,
        timeout_seconds=0.2,
        enable_notifications=False,
    )
    manager = ResearchJobManager(
        research_executor=slow_executor,
        config=config,
    )
    await manager.start()

    job_id = await manager.queue_research(
        query="slow query",
        transcript="test",
        game_id="tunic",
    )

    # Wait for timeout
    await asyncio.sleep(0.5)

    job = await manager.get_job(job_id)
    assert job is not None
    assert job.status == JobStatus.ERROR
    assert job.error is not None
    assert "timed out" in job.error.lower()

    await manager.stop()


@pytest.mark.asyncio
async def test_completion_callback(job_manager):
    """Test completion callback invocation."""
    callback_called = asyncio.Event()
    callback_job = None

    def on_complete(job: ResearchJob):
        nonlocal callback_job
        callback_job = job
        callback_called.set()

    job_id = await job_manager.queue_research(
        query="test query",
        transcript="test",
        game_id="tunic",
        on_complete=on_complete,
    )

    # Wait for callback
    await asyncio.wait_for(callback_called.wait(), timeout=1.0)

    assert callback_job is not None
    assert callback_job.job_id == job_id
    assert callback_job.status == JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_has_active_research(job_manager):
    """Test active research detection."""
    assert not await job_manager.has_active_research()

    job_id = await job_manager.queue_research(
        query="test",
        transcript="test",
        game_id="tunic",
    )

    # Should have active research while running
    assert await job_manager.has_active_research()

    # Wait for completion
    await asyncio.sleep(0.3)

    # Should not have active research after completion
    assert not await job_manager.has_active_research()


@pytest.mark.asyncio
async def test_list_active_jobs(job_manager):
    """Test listing active jobs."""
    active = await job_manager.list_active_jobs()
    assert len(active) == 0

    # Queue multiple jobs
    job_ids = []
    for i in range(3):
        job_id = await job_manager.queue_research(
            query=f"query {i}",
            transcript=f"test {i}",
            game_id="tunic",
        )
        job_ids.append(job_id)

    # Should have active jobs
    active = await job_manager.list_active_jobs()
    assert len(active) > 0
    assert any(j.job_id in job_ids for j in active)

    # Wait for completion
    await asyncio.sleep(0.5)

    # Should be empty after completion
    active = await job_manager.list_active_jobs()
    assert len(active) == 0
