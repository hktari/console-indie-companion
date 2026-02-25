"""Main pipeline orchestrator for the Tunic Voice Companion.

Wires together: capture → VLM → context manager → voice session.

Usage:
    # Replay mode (no PS5 needed)
    python -m src.main --replay --screenshot-dir data/screenshots/ --duration 60

    # Replay mode without voice (test VLM pipeline only)
    python -m src.main --replay --no-voice --duration 30

    # Live mode
    python -m src.main --window "PS Remote Play" --interval 3
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from typing import Optional, Union

from dotenv import load_dotenv

from src.capture.capture import CaptureService
from src.capture.replay import ReplayCapture
from src.vlm.analyze import SceneAnalyzer
from src.voice.realtime import VoiceSession
from src.utils import CostTracker
from src.prompts.tunic_companion import (
    CONTEXT_UPDATE_TEMPLATE,
    SYSTEM_INSTRUCTIONS,
)

from src.context.manager import ContextManager

# RAG may not be indexed yet — import but handle failures gracefully.
try:
    from src.rag.query import query_tunic_knowledge
except Exception:
    query_tunic_knowledge = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="Tunic Voice Companion — capture → VLM → context → voice pipeline",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--window",
        help="Window name for live capture (e.g. 'chiaki-ng' or 'PS Remote Play').",
    )
    group.add_argument(
        "--window-id",
        help="X11 window ID in hex (e.g. 0xb60000e) or decimal.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Screenshot interval in seconds (default: 3).",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Use replay mode (read screenshots from disk instead of live capture).",
    )
    parser.add_argument(
        "--screenshot-dir",
        default="data/screenshots/",
        help="Directory of screenshots for replay mode (default: data/screenshots/).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Max runtime in seconds (0 = infinite, default: 0).",
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Skip voice session (useful for testing VLM pipeline only).",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        choices=["gemini-2.5-flash", "gemini-2.5-flash-lite"],
        help="Gemini model to use (default: gemini-2.5-flash). Use 'gemini-2.5-flash-lite' for lower cost.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# API key validation
# ---------------------------------------------------------------------------


def validate_env(need_voice: bool) -> None:
    """Ensure required API keys are present. Exits on failure."""
    missing: list[str] = []
    if not os.environ.get("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    if need_voice and not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if missing:
        logger.error("Missing required environment variables: %s", ", ".join(missing))
        logger.error("Set them in .env or export them before running.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# RAG helper
# ---------------------------------------------------------------------------


def fetch_rag_context(scene: dict) -> str:
    """Build a RAG query from the scene and return formatted results."""
    if query_tunic_knowledge is None:
        return "(RAG not available — run 'python -m src.rag.index' first)"

    parts = [
        str(scene.get("location", "")),
        str(scene.get("activity", "")),
        str(scene.get("notable_items", "")),
    ]
    query = " ".join(p for p in parts if p).strip()
    if not query:
        return "(no query terms)"

    try:
        results = query_tunic_knowledge(query, n_results=3)
        if results:
            return "\n".join(results)
        return "(no relevant results)"
    except Exception:
        logger.debug("RAG query failed", exc_info=True)
        return "(RAG query failed)"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def run_pipeline(args: argparse.Namespace) -> None:
    """Run the capture → VLM → context → voice pipeline."""
    running = True

    # -- Signal handling --------------------------------------------------
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        nonlocal running
        running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _request_stop)

    # -- 1. Initialise modules -------------------------------------------
    capture: Union[ReplayCapture, CaptureService]
    if args.replay:
        capture = ReplayCapture(args.screenshot_dir, interval=args.interval)
        logger.info("Mode: REPLAY from %s", args.screenshot_dir)
    else:
        # Default to "PS Remote Play" if neither is provided
        window_name = args.window or ("PS Remote Play" if not args.window_id else None)
        capture = CaptureService(
            window_id=args.window_id,
            window_name=window_name,
            interval=args.interval
        )
        if args.window_id:
            logger.info("Mode: LIVE capture of window ID %s", args.window_id)
        else:
            logger.info("Mode: LIVE capture of window name '%s'", window_name)

    vlm = SceneAnalyzer(model=args.model)
    logger.info("VLM (Gemini) initialised")

    context_mgr = ContextManager()
    logger.info("Context manager loaded")

    voice: Optional[VoiceSession] = None
    if not args.no_voice:
        voice = VoiceSession(system_instructions=SYSTEM_INSTRUCTIONS)
        logger.info("Voice session created")

    # -- 2. Start capture ------------------------------------------------
    if not capture.find_window():
        logger.error("Could not locate capture source. Exiting.")
        return
    capture.start()

    # -- 3. Start voice session -----------------------------------------
    if voice:
        try:
            await voice.start()
            logger.info("Voice session connected")
        except Exception:
            logger.exception("Failed to start voice session — continuing without voice")
            voice = None

    # -- 4. Main loop ---------------------------------------------------
    logger.info("Pipeline running. Press Ctrl+C to stop.")
    start_time = asyncio.get_event_loop().time()
    last_frame: Optional[bytes] = None
    analysis_count = 0
    cost_tracker = CostTracker()
    logger.info("Cost tracker initialised")

    try:
        while running:
            # Check duration limit
            if args.duration > 0:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= args.duration:
                    logger.info("Duration limit reached (%.0fs). Stopping.", elapsed)
                    break

            # Get latest frame from capture service
            frame = capture.get_latest_frame()

            if frame is None:
                logger.debug("No frame available yet, waiting...")
                await asyncio.sleep(1)
                continue

            if frame == last_frame:
                # Same frame — no new capture yet, skip VLM call.
                await asyncio.sleep(1)
                continue

            # New frame detected
            last_frame = frame

            # Analyse with VLM (synchronous call → run in thread)
            try:
                scene = await asyncio.to_thread(
                    vlm.analyze_screenshot, frame, "image/jpeg"
                )
            except Exception:
                logger.exception("VLM analysis failed — skipping frame")
                await asyncio.sleep(1)
                continue

            if not scene or "error" in scene:
                logger.warning("VLM returned error: %s", scene.get("error") if scene else "empty")
                await asyncio.sleep(1)
                continue

            analysis_count += 1
            logger.debug(
                "[#%d] Scene: %s | Location: %s | Activity: %s",
                analysis_count,
                scene.get("description", "unknown")[:80],
                scene.get("location", "?"),
                scene.get("activity", "?"),
            )

            # Fetch RAG context and attach to scene
            rag_context = await asyncio.to_thread(fetch_rag_context, scene)
            scene["rag_context"] = rag_context

            # Update context manager
            context_mgr.update_scene(scene)

            # Flush to voice session
            if voice and voice.is_connected():
                try:
                    await context_mgr.flush_to_voice(voice)
                    logger.debug("Context injected into voice session")
                except Exception:
                    logger.exception("Failed to inject context into voice session")
            elif voice and not voice.is_connected():
                logger.warning("Voice session disconnected — VLM pipeline continues")

            # Wait before checking for next frame
            await asyncio.sleep(1)

    except asyncio.CancelledError:
        logger.info("Pipeline cancelled")

    # -- 5. Cleanup -----------------------------------------------------
    logger.info("Shutting down... (%d scenes analysed)", analysis_count)
    capture.stop()
    if voice:
        try:
            await voice.stop()
        except Exception:
            logger.exception("Error stopping voice session")
    logger.info("Shutdown complete")
    cost_tracker.print_summary()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    # Logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Suppress noisy library logs unless in DEBUG mode
    if numeric_level > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("google_genai").setLevel(logging.WARNING)

    # Environment
    load_dotenv()
    validate_env(need_voice=not args.no_voice)

    # Run
    try:
        asyncio.run(run_pipeline(args))
    except KeyboardInterrupt:
        # Fallback if signal handler didn't catch it (e.g. Windows)
        logger.info("Interrupted.")


if __name__ == "__main__":
    main()