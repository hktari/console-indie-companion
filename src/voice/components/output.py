"""Output components for voice interaction - TTS and audio playback."""

import io
import logging
import os
import wave
from typing import Optional

import numpy as np
import openai

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

logger = logging.getLogger(__name__)

PCM16_MAX = 32767.0


class OpenAITTSPlayer:
    """OpenAI Text-to-Speech player."""

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
