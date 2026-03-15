"""Input components for voice interaction - recording and transcription."""

import io
import logging
import os
import threading
import time
import wave
from typing import Optional, Any

import numpy as np
import openai

try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

from src.voice.config import CHANNELS, SAMPLE_RATE

logger = logging.getLogger(__name__)


class PTTRecorder:
    """Audio recorder with push-to-talk or toggle modes.

    Modes:
        - "push-to-talk": Hold to record, release to stop (default)
        - "toggle": Press once to start, press again to stop
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        mode: str = "push-to-talk",
        enable_sounds: bool = True,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.mode = mode
        self.enable_sounds = enable_sounds
        self._pa: Optional[Any] = None
        self._stream: Optional[Any] = None
        self._frames: list[bytes] = []
        self._lock = threading.Lock()
        self._recording = False
        self._started_at: Optional[float] = None

    def _play_beep(self, frequency: float = 880.0, duration: float = 0.15) -> None:
        """Play a simple beep tone for UX feedback."""
        if not self.enable_sounds or sd is None:
            return
        try:
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)
            tone = np.sin(2 * np.pi * frequency * t) * 0.3
            sd.play(tone.astype(np.float32), samplerate=self.sample_rate, blocking=True)
        except Exception:
            pass  # Ignore sound playback errors

    def toggle_recording(self) -> bytes:
        """Toggle recording state. Returns audio bytes if stopped, empty bytes if started.

        In "toggle" mode: starts if stopped (returns b""), stops if recording (returns audio).
        In "push-to-talk" mode: delegates to start/stop methods.
        """
        if self.mode == "toggle":
            audio = b""
            with self._lock:
                was_recording = self._recording
                if was_recording:
                    # Stop recording
                    audio = self._do_stop_unlocked()
                else:
                    # Start recording
                    self._do_start_unlocked()

            # Play beep AFTER releasing lock to avoid blocking audio callback
            if was_recording:
                self._play_beep(frequency=440.0, duration=0.2)  # Lower tone for stop
                return audio
            else:
                self._play_beep(frequency=880.0, duration=0.15)  # Higher tone for start
                return b""
        else:
            # Push-to-talk mode: this shouldn't be called directly
            # but handle gracefully
            if self._recording:
                self.stop_recording()
                return b""
            else:
                self.start_recording()
                return b""

    def _do_start_unlocked(self) -> None:
        """Internal: start recording (lock must be held by caller)."""
        if pyaudio is None:
            raise RuntimeError("pyaudio unavailable")
        if self._recording:
            return

        self._pa = pyaudio.PyAudio()
        self._frames = []

        def _callback(in_data, frame_count, time_info, status_flags):
            if status_flags:
                logger.warning("Mic status: %s", status_flags)
            with self._lock:
                if self._recording:
                    self._frames.append(in_data)
            return (None, 0 if pyaudio is None else pyaudio.paContinue)

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=int(self.sample_rate / 10),
            stream_callback=_callback,
        )
        self._recording = True
        self._started_at = time.time()
        if self._stream is not None:
            self._stream.start_stream()
        logger.info("Recording started (toggle mode)")

    def _do_stop_unlocked(self) -> bytes:
        """Internal: stop recording and return audio (lock must be held by caller)."""
        if not self._recording:
            return b""

        self._recording = False
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

        raw_audio = b"".join(self._frames)
        self._frames = []

        logger.info("Recording stopped (%d bytes)", len(raw_audio))
        return raw_audio

    def start_recording(self) -> None:
        """Start recording (for push-to-talk mode)."""
        if self.mode == "toggle":
            # In toggle mode, delegate to toggle_recording
            # Discard return value since start_recording() returns None
            _ = self.toggle_recording()
            return

        if pyaudio is None:
            raise RuntimeError("pyaudio unavailable – push-to-talk recording disabled")
        if self._recording:
            return

        self._play_beep(frequency=880.0, duration=0.15)  # High beep for start

        self._pa = pyaudio.PyAudio()
        self._frames = []

        def _callback(in_data, frame_count, time_info, status_flags):
            if status_flags:
                logger.warning("Push-to-talk mic status: %s", status_flags)
            with self._lock:
                if self._recording:
                    self._frames.append(in_data)
            return (None, 0 if pyaudio is None else pyaudio.paContinue)

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=int(self.sample_rate / 10),
            stream_callback=_callback,
        )
        self._recording = True
        self._started_at = time.time()
        if self._stream is not None:
            self._stream.start_stream()
        logger.info("Push-to-talk recording started")

    def stop_recording(self) -> bytes:
        """Stop recording (for push-to-talk mode)."""
        if self.mode == "toggle":
            # In toggle mode, delegate to toggle_recording and return its result
            return self.toggle_recording()

        if not self._recording:
            return b""

        self._recording = False
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

        self._play_beep(frequency=440.0, duration=0.2)  # Low beep for stop

        with self._lock:
            raw_audio = b"".join(self._frames)
            self._frames = []

        logger.info("Push-to-talk recording stopped (%d bytes)", len(raw_audio))
        return raw_audio

    def is_recording(self) -> bool:
        return self._recording

    def discard(self) -> None:
        if self._recording:
            self.stop_recording()


class BatchTranscriber:
    """OpenAI Whisper batch transcription for recorded audio."""

    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-1") -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for batch transcription.")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model

    def transcribe_pcm16(self, pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> str:
        if not pcm_bytes:
            return ""

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)

        wav_buffer.seek(0)
        wav_buffer.name = "push_to_talk.wav"
        transcript = self.client.audio.transcriptions.create(
            model=self.model,
            file=wav_buffer,
        )
        text = transcript.text.strip() if transcript.text else ""
        logger.info("Batch transcription completed: %s", text)
        return text
