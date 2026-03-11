import asyncio
import io
import logging
import os
import threading
import time
import wave
from dataclasses import dataclass
from typing import Any, Optional, Protocol

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

from src.context.manager import ContextManager
from src.prompts.tunic_companion import SYSTEM_INSTRUCTIONS
from src.rag.orchestrator import KnowledgeOrchestrator, RetrievalResult
from src.voice.config import CHANNELS, SAMPLE_RATE

logger = logging.getLogger(__name__)

PCM16_MAX = 32767.0


class FrameProvider(Protocol):
    def get_latest_frame(self) -> Optional[bytes]: ...

    def capture_once(self) -> Optional[bytes]: ...


@dataclass
class PromptContext:
    transcript: str
    scene: Optional[dict[str, Any]]
    narrative: str
    retrieval_results: list[RetrievalResult]


class PushToTalkRecorder:
    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS):
        self.sample_rate = sample_rate
        self.channels = channels
        self._pa: Optional[Any] = None
        self._stream: Optional[Any] = None
        self._frames: list[bytes] = []
        self._lock = threading.Lock()
        self._recording = False
        self._started_at: Optional[float] = None

    def start_recording(self) -> None:
        if pyaudio is None:
            raise RuntimeError("pyaudio unavailable – push-to-talk recording disabled")
        if self._recording:
            return

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


class OpenAIBatchTranscriber:
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


class OpenAITTSPlayer:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "tts-1",
        voice: str = "alloy",
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for TTS.")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        self.voice = voice

    def speak(self, text: str) -> None:
        if not text or sd is None:
            return

        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            response_format="wav",
        )
        audio_bytes = response.content
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()

        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / PCM16_MAX
        if channels > 1:
            audio = audio.reshape(-1, channels)
        sd.play(audio, samplerate=sample_rate)
        sd.wait()


class NonRealtimeVoiceSession:
    def __init__(
        self,
        frame_provider: FrameProvider,
        scene_analyzer: Any,
        context_manager: ContextManager,
        orchestrator: KnowledgeOrchestrator,
        cost_tracker: Optional[Any] = None,
        system_instructions: str = SYSTEM_INSTRUCTIONS,
        model: str = "gpt-4.1-mini",
        stt_model: str = "whisper-1",
        tts_model: str = "tts-1",
        tts_voice: str = "alloy",
        game_id: str = "tunic",
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required.")

        self._client = openai.OpenAI(api_key=api_key)
        self._frame_provider = frame_provider
        self._scene_analyzer = scene_analyzer
        self._context_manager = context_manager
        self._orchestrator = orchestrator
        self._cost_tracker = cost_tracker
        self._system_instructions = system_instructions
        self._model = model
        self._game_id = game_id

        self._recorder = PushToTalkRecorder()
        self._transcriber = OpenAIBatchTranscriber(api_key=api_key, model=stt_model)
        self._tts_player = OpenAITTSPlayer(
            api_key=api_key,
            model=tts_model,
            voice=tts_voice,
        )
        self._active_lock = asyncio.Lock()
        self._last_response: Optional[str] = None

    def start_recording(self) -> None:
        self._recorder.start_recording()

    async def stop_recording_and_respond(self) -> Optional[str]:
        pcm_bytes = await asyncio.to_thread(self._recorder.stop_recording)
        if not pcm_bytes:
            return None

        async with self._active_lock:
            started_at = time.perf_counter()
            transcript = await asyncio.to_thread(
                self._transcriber.transcribe_pcm16, pcm_bytes
            )
            if not transcript:
                logger.info("No transcript captured from push-to-talk audio")
                return None

            prompt_context = await self._build_prompt_context(transcript)
            reply = await asyncio.to_thread(self._generate_reply, prompt_context)
            if reply:
                self._last_response = reply
                await asyncio.to_thread(self._tts_player.speak, reply)

            if self._cost_tracker:
                self._cost_tracker.log_call(
                    service="openai",
                    model=self._model,
                    duration_seconds=time.perf_counter() - started_at,
                )
            return reply

    def is_recording(self) -> bool:
        return self._recorder.is_recording()

    def get_last_response(self) -> Optional[str]:
        return self._last_response

    async def inject_context(self, context_text: str) -> None:
        reply = await asyncio.to_thread(
            self._generate_reply_from_event,
            context_text,
        )
        if reply:
            self._last_response = reply
            await asyncio.to_thread(self._tts_player.speak, reply)

    async def _build_prompt_context(self, transcript: str) -> PromptContext:
        frame = self._frame_provider.capture_once()
        if frame is None:
            frame = self._frame_provider.get_latest_frame()

        scene: Optional[dict[str, Any]] = None
        if frame is not None:
            try:
                scene = await asyncio.to_thread(
                    self._scene_analyzer.analyze_screenshot, frame, "image/jpeg"
                )
                if scene and isinstance(scene, dict) and "error" not in scene:
                    self._context_manager.update_scene(scene)
                else:
                    scene = None
            except Exception:
                logger.exception("Prompt-time VLM analysis failed")
                scene = None

        narrative = self._context_manager.get_current_narrative()
        retrieval_results: list[RetrievalResult] = []
        if scene:
            try:
                query = self._build_retrieval_query(transcript, scene)
                retrieval_results = await asyncio.to_thread(
                    self._orchestrator.resolve, query, self._game_id
                )
            except Exception:
                logger.exception("Prompt-time retrieval failed")

        return PromptContext(
            transcript=transcript,
            scene=scene,
            narrative=narrative,
            retrieval_results=retrieval_results[:3],
        )

    def _build_retrieval_query(self, transcript: str, scene: dict[str, Any]) -> str:
        scene_bits = [
            transcript,
            str(scene.get("location", "")),
            str(scene.get("activity", "")),
            str(scene.get("notable_items", "")),
        ]
        return " ".join(bit for bit in scene_bits if bit and bit != "None")

    def _generate_reply(self, prompt_context: PromptContext) -> str:
        scene_text = "No fresh scene analysis available."
        if prompt_context.scene:
            scene = prompt_context.scene
            scene_text = (
                f"Scene description: {scene.get('description', 'unknown')}\n"
                f"Location: {scene.get('location', 'unknown')}\n"
                f"Activity: {scene.get('activity', 'unknown')}\n"
                f"Visible enemies: {scene.get('enemies', 'none')}\n"
                f"Player health: {scene.get('health_status', 'unknown')}\n"
                f"UI elements: {scene.get('ui_elements', 'none')}\n"
                f"Notable items: {scene.get('notable_items', 'none')}"
            )

        retrieval_text = "No additional retrieval context."
        if prompt_context.retrieval_results:
            retrieval_text = "\n\n".join(
                f"[{result.source}]\n{result.content}"
                for result in prompt_context.retrieval_results
            )

        user_prompt = (
            f"Player said: {prompt_context.transcript}\n\n"
            f"Recent narrative: {prompt_context.narrative}\n\n"
            f"{scene_text}\n\n"
            f"Retrieved knowledge:\n{retrieval_text}"
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_instructions},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.8,
            max_tokens=160,
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return ""

    def _generate_reply_from_event(self, context_text: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_instructions},
                {"role": "user", "content": context_text},
            ],
            temperature=0.7,
            max_tokens=60,
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return ""
