"""Tests for detector engine and frame-based detectors."""

import io
import time

from PIL import Image

from src.detector.engine import DetectorEngine, DetectorEvent


class MockDetector:
    """Mock detector for testing."""

    id = "mock-detector"
    game_id = "test-game"

    def __init__(self, should_emit: bool = True):
        self.should_emit = should_emit
        self.probe_count = 0

    def probe(self, frame_bytes: bytes) -> DetectorEvent | None:
        """Mock probe that optionally emits an event."""
        self.probe_count += 1
        if self.should_emit:
            return DetectorEvent(
                id=f"event-{self.probe_count}",
                game_id=self.game_id,
                timestamp=time.time(),
                type="test_event",
                data={"count": self.probe_count},
                confidence=0.9,
            )
        return None


class FailingDetector:
    """Detector that always raises an exception."""

    id = "failing-detector"
    game_id = "test-game"

    def probe(self, frame_bytes: bytes) -> DetectorEvent | None:
        """Always raises an exception."""
        raise RuntimeError("Detector failure")


def create_test_frame(width: int = 100, height: int = 100) -> bytes:
    """Create a simple test frame as JPEG bytes."""
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_detector_engine_initialization():
    """Test detector engine can be initialized."""
    engine = DetectorEngine()
    assert engine is not None


def test_detector_registration():
    """Test detectors can be registered."""
    engine = DetectorEngine()
    detector = MockDetector()

    engine.register_detector(detector)
    assert len(engine._detectors) == 1


def test_detector_engine_processes_frame():
    """Test detector engine processes frames and returns events."""
    engine = DetectorEngine()
    detector = MockDetector(should_emit=True)
    engine.register_detector(detector)

    frame = create_test_frame()
    events = engine.process_frame(frame)

    assert len(events) == 1
    assert events[0].type == "test_event"
    assert events[0].confidence == 0.9
    assert detector.probe_count == 1


def test_detector_engine_handles_no_events():
    """Test detector engine handles detectors that emit no events."""
    engine = DetectorEngine()
    detector = MockDetector(should_emit=False)
    engine.register_detector(detector)

    frame = create_test_frame()
    events = engine.process_frame(frame)

    assert len(events) == 0
    assert detector.probe_count == 1


def test_detector_engine_isolates_failures():
    """Test detector engine isolates failures from individual detectors."""
    engine = DetectorEngine()

    # Register a failing detector and a working detector
    engine.register_detector(FailingDetector())
    working_detector = MockDetector(should_emit=True)
    engine.register_detector(working_detector)

    frame = create_test_frame()
    events = engine.process_frame(frame)

    # Should get event from working detector despite failing detector
    assert len(events) == 1
    assert events[0].type == "test_event"


def test_detector_engine_multiple_detectors():
    """Test detector engine aggregates events from multiple detectors."""
    engine = DetectorEngine()

    detector1 = MockDetector(should_emit=True)
    detector2 = MockDetector(should_emit=True)
    engine.register_detector(detector1)
    engine.register_detector(detector2)

    frame = create_test_frame()
    events = engine.process_frame(frame)

    assert len(events) == 2
    assert all(e.type == "test_event" for e in events)
