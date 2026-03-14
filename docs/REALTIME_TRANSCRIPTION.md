# Realtime Transcription Mode

## Overview

The companion now supports **realtime transcription with automatic agent submission**. This mode uses OpenAI's Realtime API for continuous speech-to-text with VAD (Voice Activity Detection), while keeping the agent runtime completely local.

## How It Works

1. **Continuous Transcription**: Your speech is transcribed in real-time as you speak
2. **VAD Segmentation**: OpenAI's server-side VAD automatically detects utterance boundaries
3. **Auto-Submit on Silence**: When VAD detects you've stopped speaking, the transcript is automatically submitted to the agent
4. **Local Agent**: Planner, retrieval, memory, and reply generation run locally as before
5. **Natural Conversation**: Just speak naturally and pause - the agent responds when you finish talking

## Key Difference from PTT

| Feature | PTT Mode | Realtime Transcription Mode |
|---------|----------|----------------------------|
| Mic behavior | Only active while key held | Always listening |
| Transcription | Batch after release | Continuous, real-time |
| Agent trigger | Automatic on release | Automatic on VAD silence detection |
| Transcript visibility | None (until complete) | Can see partial/final text |
| OpenAI usage | Whisper batch API | Realtime API (transcription only) |

## Usage

### Basic Command

```bash
# Use realtime transcription with default settings
uv run src/main.py --replay --input-mode realtime --duration 60
```

### Configuration Options

```bash
# Specify the submit key (default: shift_r)
uv run src/main.py --replay --input-mode realtime --ptt-key space

# Combine with live capture
uv run src/main.py --window "PS Remote Play" --input-mode realtime
```

### Key Behavior

In realtime transcription mode:
- **No key required**: Agent automatically responds when VAD detects you've stopped speaking
- **Natural pauses**: The VAD silence detection (default: 700ms) triggers agent submission
- **Hands-free**: Just speak naturally and wait for the response

## Architecture

### Components

1. **RealtimeTranscriber** (`src/voice/realtime_transcriber.py`)
   - Manages WebSocket connection to OpenAI Realtime API
   - Streams mic audio continuously
   - Receives VAD events and transcription
   - Buffers finalized transcripts locally
   - **Does NOT** trigger automatic assistant responses

2. **NonRealtimeVoiceSession** (updated)
   - Now supports both `ptt` and `realtime` input modes
   - Maintains existing agent pipeline (planner, retrieval, memory, TTS)
   - New method: `submit_transcript_and_respond()`

3. **Main Pipeline** (updated)
   - Detects input mode and starts appropriate session
   - Key listener adapts behavior based on mode

### OpenAI Configuration

The realtime transcriber uses these session settings:

```python
{
    "modalities": ["text"],  # Text only, no audio output
    "input_audio_transcription": {"model": "whisper-1"},
    "turn_detection": {
        "type": "server_vad",
        "create_response": False,  # Critical: no auto-response
        "threshold": 0.5,
        "silence_duration_ms": 700,
    },
}
```

**Key setting**: `create_response: False` ensures OpenAI only transcribes, never generates responses.

## Workflow Example

1. Start the companion with realtime transcription:
   ```bash
   uv run src/main.py --replay --input-mode realtime --duration 120
   ```

2. Speak naturally: "Where am I? What should I do next?"

3. Watch the logs for transcription events:
   ```
   [Transcribed] Where am I?
   [Transcribed] What should I do next?
   ```

4. Stop speaking and pause (VAD detects silence after ~700ms)

5. Auto-submit is triggered:
   ```
   VAD detected speech stopped, auto-submitting transcript
   ```

6. Agent processes the combined transcript:
   - Planner routes the query
   - Retrieval gathers evidence
   - Local LLM generates response
   - TTS speaks the answer

7. Continue speaking naturally - each pause triggers a new agent interaction

## Benefits

- **Natural speech flow**: No need to hold a key while speaking
- **Better segmentation**: VAD handles pauses and utterance boundaries
- **Automatic prompting**: VAD decides when to invoke the agent based on natural pauses
- **Same agent quality**: Local planner/retrieval/memory unchanged
- **Visible transcripts**: See what was transcribed before submitting

## Limitations

- **Requires OpenAI API key**: Uses Realtime API for transcription
- **Higher latency**: Transcription is fast, but network adds overhead
- **Automatic submission**: Every pause triggers agent response (may be chatty)
- **No partial editing**: Can't edit transcripts before submission (future enhancement)

## Cost Considerations

Realtime API transcription costs more than batch Whisper:
- Realtime API: ~$0.06/minute of audio input
- Batch Whisper: ~$0.006/minute

However, agent execution (planner, retrieval, LLM) remains local and uses the same models as PTT mode.

## Future Enhancements

Potential improvements:
- [ ] Display partial transcripts in UI
- [ ] Manual override key to force submission or cancel
- [ ] Configurable VAD sensitivity and silence duration
- [ ] Transcript history/recall
- [ ] Multi-turn conversation context
- [ ] Smart filtering to avoid responding to non-query speech

## Troubleshooting

### No transcription appearing

Check:
1. Microphone permissions
2. OpenAI API key is set: `echo $OPENAI_API_KEY`
3. Audio input device is working
4. Logs show "Realtime transcription started"

### Agent responding too frequently

If the agent responds to every pause:
1. Increase VAD silence duration (requires code change currently)
2. Speak in longer sentences without pauses
3. Consider adding a manual submit key as an alternative mode

### Transcription quality issues

Adjust:
- Microphone gain (currently hardcoded 100x in AudioManager)
- Background noise (use quieter environment)
- VAD threshold (requires code change currently)

## Technical Details

### Session Lifecycle

```
1. main.py creates NonRealtimeVoiceSession(input_mode="realtime")
2. Session creates RealtimeTranscriber with on_speech_stopped callback
3. main.py calls voice.start_transcription()
4. RealtimeTranscriber connects to OpenAI WebSocket
5. AudioManager starts mic capture
6. Audio streams to OpenAI continuously
7. VAD events trigger transcript finalization
8. Transcripts buffer locally
9. VAD detects silence → _on_vad_speech_stopped() callback
10. Auto-submit → submit_transcript_and_respond()
11. Agent processes transcript (local pipeline)
12. TTS speaks response
13. Loop continues until shutdown
```

### Event Flow

```
Mic → AudioManager → RealtimeTranscriber → OpenAI Realtime API
                                              ↓
                                         VAD Events
                                              ↓
                                    Transcript Completed
                                              ↓
                                      Local Buffer
                                              ↓
                                    [VAD: speech_stopped]
                                              ↓
                              _on_vad_speech_stopped() callback
                                              ↓
                              submit_transcript_and_respond()
                                              ↓
                                  Local Agent Pipeline
                                   (planner → retrieval
                                    → LLM → memory → TTS)
```
