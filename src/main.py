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
from typing import Any, Optional, Union

from dotenv import load_dotenv
from pynput import keyboard

from src.capture.capture import CaptureService
from src.capture.replay import ReplayCapture
from src.vlm.analyze import SceneAnalyzer
from src.voice.non_realtime import NonRealtimeVoiceSession
from src.voice.realtime import VoiceSession
from src.utils import CostTracker
from src.utils.logging_config import setup_logging

from src.prompts.tunic_companion import SYSTEM_INSTRUCTIONS

from src.context.manager import ContextManager
from src.context.synthesizer import ContextSynthesizer
from src.detector.engine import DetectorEngine
from src.memory.retriever import MemoryRetriever
from src.rag import ExaRetriever, KnowledgeOrchestrator, LocalGameRetriever

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
        default=0.0,  # gemini-2.5-flash-lite has around 2s latency anyway
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
        "--voice-mode",
        default="non-realtime",
        choices=["non-realtime", "realtime"],
        help="Voice interaction mode (default: non-realtime).",
    )
    parser.add_argument(
        "--ptt-key",
        default="ctrl",
        help="Push-to-talk key for non-realtime mode (default: ctrl).",
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
    if not os.environ.get("QMD_URL"):
        logger.warning(
            "QMD_URL not set. Fallback to CLI mode. Will be slow on CPU only (no GPU acceleration)."
        )
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


def setup_key_listener(
    voice_session: Any, loop: asyncio.AbstractEventLoop, ptt_key: str
) -> None:
    """Setup and run a non-blocking keyboard listener."""

    target_key = ptt_key.lower()

    def _key_matches(key: keyboard.Key | keyboard.KeyCode | None) -> bool:
        if key is None:
            return False
        if isinstance(key, keyboard.KeyCode):
            if key.char:
                return key.char.lower() == target_key
            return False
        return getattr(key, "name", "").lower() == target_key

    def on_press(key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if not _key_matches(key):
            return

        if (
            hasattr(voice_session, "start_recording")
            and not voice_session.is_recording()
        ):
            try:
                voice_session.start_recording()
            except Exception:
                logger.exception("Failed to start push-to-talk recording")

    def on_release(key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if not _key_matches(key):
            return

        if (
            hasattr(voice_session, "stop_recording_and_respond")
            and voice_session.is_recording()
        ):
            future = asyncio.run_coroutine_threadsafe(
                voice_session.stop_recording_and_respond(),
                loop,
            )

            def _done_callback(result_future):
                try:
                    reply = result_future.result()
                    if reply:
                        logger.info("Assistant: %s", reply)
                except Exception:
                    logger.exception("Push-to-talk response failed")

            future.add_done_callback(_done_callback)

    # The listener runs in its own thread, so it's non-blocking
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        logger.info("Key listener started for push-to-talk key '%s'.", ptt_key)
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
    logger.info(f"Starting context synthesis loop (running every {interval_seconds}s)")
    while True:
        try:
            await asyncio.sleep(interval_seconds)

            with context_mgr._lock:  # type: ignore
                scenes = list(context_mgr._scenes)[-num_scenes_for_synthesis:]

            if not scenes:
                continue

            rag_context = ""
            if query_tunic_knowledge:
                rag_context = await asyncio.to_thread(
                    context_mgr.get_rag_context, scenes[-1]
                )

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
    detector_engine: DetectorEngine,
    voice: Optional[Any] = None,
    vlm_interval: float = 5.0,
) -> None:
    """The main VLM analysis loop with dual-trigger scheduler.

    VLM analysis runs on:
    1. Periodic timer (every vlm_interval seconds, default 5s)
    2. VAD speech trigger (via voice session callback)
    """
    logger.info("Main pipeline running. Press Ctrl+C to stop.")
    start_time = asyncio.get_event_loop().time()
    last_vlm_time = 0.0
    analysis_count = 0
    vlm_trigger_requested = asyncio.Event()

    # VAD trigger callback for voice session
    def request_vlm_analysis() -> None:
        """Request immediate VLM analysis (called on VAD speech start)."""
        vlm_trigger_requested.set()

    # Register VAD callback only for realtime voice mode
    if voice and hasattr(voice, "_vlm_trigger_callback"):
        voice._vlm_trigger_callback = request_vlm_analysis  # type: ignore

    while True:  # Loop is broken by run_pipeline's signal handler
        if args.duration > 0:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= args.duration:
                logger.info("Duration limit reached (%.0fs). Stopping.", elapsed)
                break

        frame = capture.get_latest_frame()
        if frame is None:
            await asyncio.sleep(0.5)
            continue

        current_time = asyncio.get_event_loop().time()
        time_since_last_vlm = current_time - last_vlm_time

        # Check if VLM analysis should run
        should_run_vlm = False
        trigger_reason = ""

        # Trigger 1: Periodic timer
        if time_since_last_vlm >= vlm_interval:
            should_run_vlm = True
            trigger_reason = "periodic"

        # Trigger 2: VAD speech start
        if vlm_trigger_requested.is_set():
            should_run_vlm = True
            trigger_reason = "vad-prompt"
            vlm_trigger_requested.clear()

        if should_run_vlm:
            try:
                scene = await asyncio.to_thread(
                    vlm.analyze_screenshot, frame, "image/jpeg"
                )
            except Exception:
                logger.exception("VLM analysis failed — skipping frame")
                await asyncio.sleep(0.5)
                continue

            if not scene or not isinstance(scene, dict) or "error" in scene:
                logger.warning(
                    "VLM returned error/invalid: %s",
                    scene.get("error") if isinstance(scene, dict) else "empty",
                )
                await asyncio.sleep(0.5)
                continue

            analysis_count += 1
            last_vlm_time = current_time

            logger.info(
                "[#%d/%s] Scene: %s | Location: %s | Activity: %s | Health: %s",
                analysis_count,
                trigger_reason,
                (scene.get("description") or "unknown")[:80],
                scene.get("location", "?"),
                scene.get("activity", "?"),
                scene.get("health_status", "?"),
            )

            context_mgr.update_scene(scene)

            # Run frame detectors on the raw frame
            detector_events = await asyncio.to_thread(
                detector_engine.process_frame, frame
            )
            if detector_events and voice:
                for event in detector_events:
                    event_msg = f"[SYSTEM EVENT] {event.type}: {event.data}"
                    logger.info(
                        "Detector event: %s (confidence: %.2f)",
                        event.type,
                        event.confidence,
                    )
                    await voice.inject_context(event_msg)

        await asyncio.sleep(0.5)


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
            window_id=args.window_id, window_name=window_name, interval=args.interval
        )
        if args.window_id:
            logger.info("Mode: LIVE capture of window ID %s", args.window_id)
        else:
            logger.info("Mode: LIVE capture of window name '%s'", window_name)

    cost_tracker = CostTracker()
    logger.info("Cost tracker initialised")

    vlm = SceneAnalyzer(model=args.model, cost_tracker=cost_tracker)
    logger.info("VLM (Gemini) initialised")

    # Initialize RAG orchestrator
    orchestrator = KnowledgeOrchestrator()
    orchestrator.register_retriever(
        LocalGameRetriever(qmd_url=os.environ.get("QMD_URL"))
    )
    orchestrator.register_retriever(ExaRetriever())
    orchestrator.register_memory_retriever(
        MemoryRetriever(qmd_url=os.environ.get("QMD_URL"))
    )
    logger.info("RAG orchestrator initialised with local + Exa + memory retrievers")

    context_mgr = ContextManager(orchestrator=orchestrator)
    logger.info("Context manager loaded")

    synthesizer = ContextSynthesizer(model="gpt-4.1-mini")
    logger.info("Context synthesizer loaded")

    # Initialize detector engine with OpenCV frame detectors
    detector_engine = DetectorEngine(logger)
    # TODO: provide fullscreen screenshots and adjust logic to work with relative spacings, not absolute pixel positions
    # detector_engine.register_detector(TunicDeathDetector())
    # detector_engine.register_detector(TunicHealthDetector())
    logger.info("Detector engine initialised with Tunic detectors")

    voice: Optional[Any] = None
    if not args.no_voice:
        if args.voice_mode == "realtime":
            voice = VoiceSession(
                system_instructions=SYSTEM_INSTRUCTIONS,
                cost_tracker=cost_tracker,
                context_manager=context_mgr,
                synthesizer=synthesizer,
            )
            logger.info("Realtime voice session created")
        else:
            voice = NonRealtimeVoiceSession(
                frame_provider=capture,
                scene_analyzer=vlm,
                context_manager=context_mgr,
                orchestrator=orchestrator,
                cost_tracker=cost_tracker,
                system_instructions=SYSTEM_INSTRUCTIONS,
            )
            logger.info("Non-realtime voice session created")

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
            args=(voice, loop, args.ptt_key),
            daemon=True,
        )
        key_listener_thread.start()

        if args.voice_mode == "realtime":
            try:
                await voice.start()
                logger.info("Voice session connected")
            except Exception:
                logger.exception(
                    "Failed to start voice session — continuing without voice"
                )
                voice = None

    # -- 4. Start main pipeline & synthesis loop ------------------------
    main_task = None
    synthesis_task = None
    try:
        main_task = asyncio.create_task(
            main_pipeline(
                args, capture, vlm, context_mgr, cost_tracker, detector_engine, voice
            )
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
            [main_task, stop_waiter], return_when=asyncio.FIRST_COMPLETED
        )

        if main_task in done:
            try:
                # Check if the task raised an exception
                main_task.result()
                logger.info("Main task completed successfully, initiating shutdown.")
            except Exception:
                logger.exception("Main task failed with an exception.")
        elif stop_waiter in done:
            logger.info("Stop signal received, initiating shutdown.")

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
        if args.voice_mode == "realtime":
            try:
                await voice.stop()
            except Exception:
                logger.exception("Error stopping voice session")
        else:
            try:
                voice.shutdown()
            except Exception:
                logger.exception("Error shutting down voice session")
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
    log_level_name = args.log_level.upper()
    if log_level_name != "DEBUG":
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("google_genai").setLevel(logging.WARNING)
        logging.getLogger("websockets.client").setLevel(logging.INFO)
    elif log_level_name == "DEBUG":
        # Even in DEBUG, websockets.client is extremely chatty with raw frames
        logging.getLogger("websockets.client").setLevel(logging.INFO)

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
