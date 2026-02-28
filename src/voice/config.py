from src.prompts.tunic_companion import SYSTEM_INSTRUCTIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 24_000  # 24 kHz required by OpenAI Realtime API
CHANNELS = 1  # Mono
CHUNK_DURATION_MS = 100  # ms per mic capture chunk
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # samples/chunk
BYTES_PER_SAMPLE = 2  # PCM16 = 2 bytes per sample

MODEL = "gpt-realtime-mini"
WS_URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"

# ---------------------------------------------------------------------------
# Session Configuration
# ---------------------------------------------------------------------------

# Official OpenAI Realtime API session configuration object
DEFAULT_SESSION_CONFIG = {
    "modalities": ["audio", "text"],
    "instructions": SYSTEM_INSTRUCTIONS.replace("{preferred_language}", "English"),
    "voice": "alloy",
    "input_audio_format": "pcm16",
    "output_audio_format": "pcm16",
    "input_audio_noise_reduction": {
        "type": "near_field",
    },
    "input_audio_transcription": {
        "model": "whisper-1",
        "language": "en"
    },
    "turn_detection": {
        "type": "semantic_vad",
        "create_response": True,
        "interrupt_response": True
    },
    "tools": [],
    "tool_choice": "auto",
    "tracing": "auto",
    "temperature": 0.8,
    "max_response_output_tokens": "inf",
}
