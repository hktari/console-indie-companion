"""Test script to measure agent loop performance."""

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.agent.planner import RequestPlanner
from src.capture.replay import ReplayCapture
from src.context.manager import ContextManager
from src.memory.manager import ConversationMemoryManager
from src.rag import KnowledgeOrchestrator, LocalGameRetriever
from src.rag.orchestrator import KnowledgeOrchestrator
from src.utils.logging_config import setup_logging
from src.utils.performance import get_performance_tracker
from src.vlm.analyze import SceneAnalyzer

logger = logging.getLogger(__name__)


async def test_agent_loop():
    """Test the agent loop with performance measurements."""
    load_dotenv()
    setup_logging("DEBUG")
    
    perf = get_performance_tracker()
    
    # Initialize components
    logger.info("Initializing components...")
    
    capture = ReplayCapture("data/screenshots/", interval=15.0)
    if not capture.find_window():
        logger.error("No screenshots found")
        return
    capture.start()
    
    vlm = SceneAnalyzer(model="gemini-2.5-flash-lite")
    
    orchestrator = KnowledgeOrchestrator()
    orchestrator.register_retriever(LocalGameRetriever())
    
    context_mgr = ContextManager(orchestrator=orchestrator)
    
    planner = RequestPlanner(
        qmd_url=os.environ.get("QMD_URL"),
        model="gpt-4.1-mini",
    )
    
    memory_mgr = ConversationMemoryManager(
        game_id="tunic",
        memory_dir=Path("var/memory"),
    )
    
    # Test queries that should trigger different retrieval paths
    test_queries = [
        "What is this place?",  # Should trigger KB search
        "How do I beat the boss here?",  # Should trigger KB search
        "What did I do earlier?",  # Should trigger memory search
    ]
    
    logger.info("Starting performance test with %d queries", len(test_queries))
    
    for i, query in enumerate(test_queries, 1):
        logger.info("\n" + "="*60)
        logger.info("Query %d/%d: %s", i, len(test_queries), query)
        logger.info("="*60)
        
        # Get a frame
        with perf.measure("test.capture_frame"):
            frame = capture.get_latest_frame()
        
        # Analyze scene
        scene = None
        if frame:
            with perf.measure("test.vlm_analysis", log_threshold=2.0):
                scene = await asyncio.to_thread(
                    vlm.analyze_screenshot, frame, "image/jpeg"
                )
        
        # Get narrative
        narrative = context_mgr.get_current_narrative()
        
        # Plan and gather evidence
        with perf.measure("test.plan_and_gather", log_threshold=5.0):
            with perf.measure("test.planner.plan"):
                decision = await asyncio.to_thread(
                    planner.plan, query, scene, narrative
                )
            
            logger.info(
                "Decision: %s (confidence: %.2f) - %s",
                decision.route.value,
                decision.confidence,
                decision.reasoning,
            )
            
            if decision.tools_to_call:
                with perf.measure("test.planner.gather_evidence", log_threshold=3.0):
                    evidence = await asyncio.to_thread(
                        planner.gather_evidence, decision, query, "tunic"
                    )
                
                logger.info(
                    "Evidence: %d KB results, %d memory results, sources: %s",
                    len(evidence.kb_results),
                    len(evidence.memory_results),
                    ", ".join(evidence.sources),
                )
        
        # Small delay between queries
        await asyncio.sleep(0.5)
    
    # Cleanup
    capture.stop()
    orchestrator.shutdown()
    
    # Print performance summary
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE TEST COMPLETE")
    logger.info("="*60)
    perf.log_summary()


if __name__ == "__main__":
    asyncio.run(test_agent_loop())
