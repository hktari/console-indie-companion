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
    "modalities": ["audio"],
    "instructions": SYSTEM_INSTRUCTIONS,
    "voice": "ballad",
    "input_audio_noise_reduction": "near_field",
    "input_audio_format": "pcm16",
    "output_audio_format": "pcm16",
    "turn_detection": {
        "type": "server_vad",
        "threshold": 0.2,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 1000,
        "create_response": False,
        "interrupt_response": False,
    },
    "tools": [
            {
                "type": "function",
                "name": "query_knowledge_base",
                "description": "Query the Tunic knowledge base.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {"type": "string"},
                        "metadata_category": {
                            "type": "string",
                            "enum": ["location", "item", "creature", "secret", "mechanic", "general", "speedrun"]
                        }
                    },
                    "required": ["search_query"]
                }
            }
        ],
    "tracing": "auto",
    "tool_choice": "auto",
    "temperature": 0.8,
    "max_response_output_tokens": "inf"
}
