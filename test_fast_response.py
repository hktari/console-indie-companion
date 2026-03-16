"""Quick test to verify fast agent response without VLM/retrieval."""

import asyncio
import logging
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

from dotenv import load_dotenv

from src.agent.job_manager import ResearchJobManager
from src.context.manager import ContextManager
from src.rag.orchestrator import KnowledgeOrchestrator
from src.voice.components.agent import AgentPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


async def test_fast_response():
    """Test that agent responds quickly without VLM/retrieval."""

    # Create minimal dependencies
    context_manager = ContextManager()
    orchestrator = KnowledgeOrchestrator()

    # Mock research executor
    async def mock_research(query: str, game_id: str, max_calls=None, on_progress=None):
        from src.agent.models import ResearchResult

        return ResearchResult(memo="Mock research result")

    job_manager = ResearchJobManager(
        research_executor=mock_research,
    )

    # Create agent pipeline
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")

    agent = AgentPipeline(
        context_manager=context_manager,
        orchestrator=orchestrator,
        job_manager=job_manager,
        model="gpt-4.1-mini",
        game_id="tunic",
        memory_dir=Path("var/memory"),
        api_key=api_key,
    )

    # Mock frame provider and scene analyzer (should not be called)
    frame_provider = MagicMock()
    scene_analyzer = MagicMock()

    # Test transcript
    transcript = "What's the best strategy for the boss fight?"

    # Measure response time
    start = time.perf_counter()
    response = await agent.process_request(
        transcript=transcript,
        frame_provider=frame_provider,
        scene_analyzer=scene_analyzer,
    )
    duration = time.perf_counter() - start

    logger.info("=" * 70)
    logger.info("FAST RESPONSE TEST RESULTS")
    logger.info("=" * 70)
    logger.info(f"Transcript: {transcript}")
    logger.info(f"Response: {response}")
    logger.info(f"Duration: {duration:.2f}s")
    logger.info("=" * 70)

    # Verify frame provider and scene analyzer were NOT called
    assert not frame_provider.capture_once.called, "Frame capture should be skipped!"
    assert not scene_analyzer.analyze_screenshot.called, (
        "VLM analysis should be skipped!"
    )

    # Verify response is fast (should be < 2s for LLM call only)
    assert duration < 3.0, f"Response too slow: {duration:.2f}s (expected < 3s)"

    logger.info("✓ Test passed: Fast response without VLM/retrieval")
    logger.info(
        f"✓ Frame provider not called: {not frame_provider.capture_once.called}"
    )
    logger.info(
        f"✓ Scene analyzer not called: {not scene_analyzer.analyze_screenshot.called}"
    )
    logger.info(f"✓ Response time: {duration:.2f}s < 3s")


if __name__ == "__main__":
    asyncio.run(test_fast_response())
