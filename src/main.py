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
import threading
from typing import Optional, Union

from dotenv import load_dotenv
from pynput import keyboard

from src.capture.capture import CaptureService
from src.capture.replay import ReplayCapture
from src.vlm.analyze import SceneAnalyzer
from src.voice.realtime import VoiceSession
from src.utils import CostTracker
from src.utils.logging_config import setup_logging
logger = logging.getLogger(__name__)

from src.prompts.tunic_companion import (
    CONTEXT_UPDATE_TEMPLATE,
    SYSTEM_INSTRUCTIONS,
)

from src.context.manager import ContextManager
from src.context.synthesizer import ContextSynthesizer

# RAG may not be indexed yet — import but handle failures gracefully.
try:
    from src.rag.query import query_tunic_knowledge
except Exception:
    query_tunic_knowledge = None  # type: ignore[assignment]



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
        default=0.0, # gemini-2.5-flash-lite has around 2s latency anyway
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
        default="gemini-2.5-flash-lite",
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


def setup_key_listener(voice_session: VoiceSession, loop: asyncio.AbstractEventLoop) -> None:
    """Setup and run a non-blocking keyboard listener."""
    def on_press(key):
        pass

    # The listener runs in its own thread, so it's non-blocking
    with keyboard.Listener(on_press=on_press) as listener:
        logger.info("Key listener started.")
        listener.join()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def context_synthesis_loop(
    synthesizer: ContextSynthesizer,
    context_mgr: ContextManager,
    interval_seconds: int = 5,
    num_scenes_for_synthesis: int = 10,
):
    """Periodically synthesizes a narrative from recent context."""
    logger.info(
        f"Starting context synthesis loop (running every {interval_seconds}s)"
    )
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            
            with context_mgr._lock: # type: ignore
                scenes = list(context_mgr._scenes)[-num_scenes_for_synthesis:]
            
            if not scenes:
                continue

            rag_context = ""
            if query_tunic_knowledge:
                rag_context = await asyncio.to_thread(context_mgr.get_rag_context, scenes[-1])

            narrative = await asyncio.to_thread(
                synthesizer.synthesize, scenes, rag_context
            )

            if narrative:
                context_mgr.set_current_narrative(narrative)

        except asyncio.CancelledError:
            logger.info("Context synthesis loop cancelled.")
            break
        except Exception:
            logger.exception("Error in context synthesis loop.")


async def main_pipeline(
    args: argparse.Namespace,
    capture: Union[ReplayCapture, CaptureService],
    vlm: SceneAnalyzer,
    context_mgr: ContextManager,
    cost_tracker: CostTracker,
    voice: Optional[VoiceSession] = None,
) -> None:
    """The main VLM analysis loop."""
    logger.info("Main pipeline running. Press Ctrl+C to stop.")
    start_time = asyncio.get_event_loop().time()
    last_frame: Optional[bytes] = None
    analysis_count = 0

    while True: # Loop is broken by run_pipeline's signal handler
        if args.duration > 0:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= args.duration:
                logger.info("Duration limit reached (%.0fs). Stopping.", elapsed)
                break

        frame = capture.get_latest_frame()
        if frame is None:
            await asyncio.sleep(0.5)
            continue
        if frame == last_frame:
            await asyncio.sleep(0.5)
            continue
        last_frame = frame

        try:
            scene = await asyncio.to_thread(vlm.analyze_screenshot, frame, "image/jpeg")
        except Exception:
            logger.exception("VLM analysis failed — skipping frame")
            continue

        if not scene or not isinstance(scene, dict) or "error" in scene:
            logger.warning("VLM returned error/invalid: %s", scene.get("error") if isinstance(scene, dict) else "empty")
            continue

        analysis_count += 1
        logger.info(
            "[#%d] Scene: %s | Location: %s | Activity: %s | Health: %s",
            analysis_count,
            (scene.get("description") or "unknown")[:80],
            scene.get("location", "?"),
            scene.get("activity", "?"),
            scene.get("health_status", "?"),
        )

        context_mgr.update_scene(scene)
        await asyncio.sleep(args.interval)


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

    cost_tracker = CostTracker()
    logger.info("Cost tracker initialised")

    vlm = SceneAnalyzer(model=args.model, cost_tracker=cost_tracker)
    logger.info("VLM (Gemini) initialised")

    context_mgr = ContextManager()
    logger.info("Context manager loaded")

    synthesizer = ContextSynthesizer(model="gpt-4.1-mini")
    logger.info("Context synthesizer loaded")

    voice: Optional[VoiceSession] = None
    if not args.no_voice:
        voice = VoiceSession(
            system_instructions=SYSTEM_INSTRUCTIONS, 
            cost_tracker=cost_tracker,
            context_manager=context_mgr
        )
        logger.info("Voice session created")

    # -- 2. Start capture ------------------------------------------------
    if not capture.find_window():
        logger.error("Could not locate capture source. Exiting.")
        return
    capture.start()

    # -- 3. Start voice session -----------------------------------------
    if voice:
        # Start keybind listener in a separate thread
        key_listener_thread = threading.Thread(
            target=setup_key_listener,
            args=(voice, loop),
            daemon=True
        )
        key_listener_thread.start()

        try:
            await voice.start()
            logger.info("Voice session connected")
        except Exception:
            logger.exception("Failed to start voice session — continuing without voice")
            voice = None

    # -- 4. Start main pipeline & synthesis loop ------------------------
    main_task = None
    synthesis_task = None
    try:
        main_task = asyncio.create_task(
            main_pipeline(args, capture, vlm, context_mgr, cost_tracker, voice)
        )
        synthesis_task = asyncio.create_task(
            context_synthesis_loop(synthesizer, context_mgr)
        )

        # Wait for the main pipeline to finish (e.g. duration limit)
        # or for a stop signal to be received.
        stop_event = asyncio.Event()
        loop.add_signal_handler(signal.SIGINT, stop_event.set)
        loop.add_signal_handler(signal.SIGTERM, stop_event.set)

        # Create a task that completes when the stop event is set
        stop_waiter = asyncio.create_task(stop_event.wait())

        # Wait for either the main task to complete or the stop event
        done, pending = await asyncio.wait(
            [main_task, stop_waiter],
            return_when=asyncio.FIRST_COMPLETED
        )

        if stop_waiter in done:
            logger.info("Stop signal received, initiating shutdown.")
        else:
            logger.info("Main task completed, initiating shutdown.")

        # Cancel pending tasks
        for task in pending:
            task.cancel()

    except asyncio.CancelledError:
        logger.info("Main run_pipeline task cancelled.")
    finally:
        logger.info("Shutting down tasks...")
        if main_task:
            main_task.cancel()
        if synthesis_task:
            synthesis_task.cancel()
        
        tasks = [t for t in [main_task, synthesis_task] if t is not None]
        await asyncio.gather(*tasks, return_exceptions=True)


    # -- 5. Cleanup -----------------------------------------------------
    capture.stop()
    if voice:
        try:
            await voice.stop()
        except Exception:
            logger.exception("Error stopping voice session")
    logger.info("Shutdown complete")
    logger.info(cost_tracker.get_summary_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    setup_logging(args.log_level)


    # Suppress noisy library logs unless in DEBUG mode
    log_level = args.log_level.upper()
    if log_level != "DEBUG":
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