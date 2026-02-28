import asyncio
import logging
import threading
from typing import Optional, Callable, Any

import numpy as np

try:
    import sounddevice as sd
except (ImportError, OSError) as _sd_err:
    sd = None

try:
    import pyaudio
except ImportError:
    pyaudio = None

from src.voice.config import SAMPLE_RATE, CHANNELS, CHUNK_SAMPLES, BYTES_PER_SAMPLE

logger = logging.getLogger(__name__)

class AudioManager:
    """Handles microphone input and speaker output streams."""
    
    def __init__(self, on_audio_data: Optional[Callable[[bytes], None]] = None):
        self.on_audio_data = on_audio_data
        
        # Audio output state
        self._playback_buf = bytearray()
        self._playback_lock = threading.Lock()
        
        # Streams
        self._input_stream: Optional[Any] = None
        self._output_stream: Optional[Any] = None
        self._pa: Optional[Any] = None

    def start_output(self) -> None:
        """Start the speaker output stream."""
        if sd is None:
            logger.warning("sounddevice unavailable – audio output disabled")
            return

        def _speaker_callback(
            outdata: np.ndarray, frames: int, time_info: object, status: object
        ) -> None:
            if status:
                logger.warning("Speaker status: %s", status)

            need = frames * BYTES_PER_SAMPLE
            with self._playback_lock:
                available = len(self._playback_buf)
                if available >= need:
                    raw = bytes(self._playback_buf[:need])
                    del self._playback_buf[:need]
                elif available > 0:
                    raw = bytes(self._playback_buf) + b"\x00" * (need - available)
                    self._playback_buf.clear()
                else:
                    outdata.fill(0)
                    return

            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
            outdata[:, 0] = samples[:frames]

        try:
            if sd is not None:
                self._output_stream = sd.OutputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="float32",
                    blocksize=CHUNK_SAMPLES,
                    callback=_speaker_callback,
                )
                self._output_stream.start()
                logger.debug("Audio output started (24 kHz mono)")
        except OSError as exc:
            logger.warning("Cannot open audio output – continuing without: %s", exc)

    async def start_input(self, loop: asyncio.AbstractEventLoop, connected_check: Callable[[], bool]) -> None:
        """Start the microphone input loop."""
        if pyaudio is None:
            logger.warning("pyaudio unavailable – mic input disabled")
            return

        self._pa = pyaudio.PyAudio()
        
        def _mic_callback(in_data, frame_count, time_info, status_flags):
            if status_flags:
                logger.warning("Mic status: %s", status_flags)
            
            # Apply input gain (100x) and clip
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            amplified = np.clip(audio_data.astype(np.float32) * 100.0, -32768, 32767)
            pcm16 = amplified.astype(np.int16).tobytes()
            
            if self.on_audio_data:
                try:
                    loop.call_soon_threadsafe(self.on_audio_data, pcm16)
                except Exception:
                    pass
            
            return (None, 1 if pyaudio is None else pyaudio.paContinue)

        try:
            if self._pa is not None and pyaudio is not None:
                self._input_stream = self._pa.open(
                    format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SAMPLES,
                    stream_callback=_mic_callback
                )
                if self._input_stream is not None:
                    self._input_stream.start_stream()
                    logger.debug("Microphone capture started (24 kHz mono PCM16)")

            while connected_check():
                await asyncio.sleep(0.1)

        except OSError as exc:
            logger.warning("Audio device error – continuing without mic: %s", exc)
        finally:
            self.stop_input()

    def enqueue_playback(self, pcm_bytes: bytes) -> None:
        """Add PCM bytes to the playback buffer."""
        with self._playback_lock:
            self._playback_buf.extend(pcm_bytes)

    def clear_playback(self) -> None:
        """Clear all pending playback audio."""
        with self._playback_lock:
            self._playback_buf.clear()

    def stop_output(self) -> None:
        """Stop the speaker stream."""
        if self._output_stream:
            try:
                self._output_stream.stop()
                self._output_stream.close()
            except Exception:
                pass
            self._output_stream = None

    def stop_input(self) -> None:
        """Stop the microphone stream."""
        if self._input_stream:
            try:
                self._input_stream.stop_stream()
                self._input_stream.close()
            except Exception:
                pass
            self._input_stream = None
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    def stop_all(self) -> None:
        """Stop both input and output."""
        self.stop_input()
        self.stop_output()
        self.clear_playback()
