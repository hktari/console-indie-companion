# Tunic Voice Companion — Desktop POC

A proof-of-concept voice companion for the game **TUNIC** that watches your PlayStation 5 gameplay and provides contextual hints, lore discussion, and assistance through voice chat.

**Architecture**: PS5 (via Remote Play) → Screen Capture → Vision AI (Gemini) → Voice Companion (OpenAI Realtime API)

---

## Quick Start

### 1. Prerequisites

- **Python 3.10+**
- **Linux with X11** (for window capture via `xdotool` + `mss`)
- **PS Remote Play** or **Chiaki-ng** (for streaming PS5 to your PC)
- **API Keys**:
  - Google Gemini API key ([get one here](https://aistudio.google.com/apikey))
  - OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### 2. Installation

```bash
# Clone the repo (if not already)
cd console-indie-companion

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Install system dependency for window capture
sudo apt-get install -y xdotool

# Set up API keys
cp .env.example .env
# Edit .env and add your API keys:
#   GEMINI_API_KEY=your-key-here
#   OPENAI_API_KEY=your-key-here
```

### 3. Run the POC (Replay Mode — No PS5 Needed)

The easiest way to test the POC is **replay mode**, which uses pre-captured Tunic screenshots instead of a live PS5:

```bash
source venv/bin/activate

# Test the VLM pipeline (no voice, no API costs for OpenAI)
python -m src.main --replay --no-voice --duration 60

# Full pipeline with voice (requires microphone + speakers/headphones)
python -m src.main --replay --duration 60
```

**What happens:**
- Cycles through 25 pre-captured Tunic screenshots every 3 seconds
- Gemini Flash analyzes each screenshot (location, enemies, activity, etc.)
- RAG retrieves relevant Tunic wiki knowledge
- Context is injected into the voice session
- You can talk to the AI companion via your PC microphone

---

## Usage

### CLI Options

```
python -m src.main [OPTIONS]

Options:
  --window WINDOW       Window name for live capture (default: 'PS Remote Play')
  --interval INTERVAL   Screenshot interval in seconds (default: 3)
  --replay              Use replay mode (pre-captured screenshots)
  --screenshot-dir DIR  Directory for replay mode (default: data/screenshots/)
  --duration SECONDS    Max runtime in seconds (0 = infinite)
  --no-voice            Skip voice session (test VLM pipeline only)
  --model MODEL         Gemini model: gemini-2.5-flash (default) or gemini-2.5-flash-lite (budget)
```

### Example Commands

```bash
# Replay mode without voice (cheapest — only Gemini API calls)
python -m src.main --replay --no-voice --duration 120

# Replay mode with voice (full experience)
python -m src.main --replay --duration 120

# Replay mode with budget Gemini model (3x cheaper)
python -m src.main --replay --no-voice --model gemini-2.5-flash-lite --duration 60

# Live mode (requires PS Remote Play running)
python -m src.main --window "PS Remote Play" --interval 3
```

---

## Testing Individual Modules

Each module can be tested independently:

### Screen Capture

find window name using `xwininfo | grep "Window id"`

```bash
# Capture any open window (e.g., Terminal, Firefox)
python -m src.capture.capture --window "Terminal" --count 5 --output /tmp/captures/ --interval 1

# Check output
ls -lh /tmp/captures/
```

### VLM Scene Analysis

```bash
# Analyze a single screenshot
python -m src.vlm.analyze --image data/screenshots/tunic_01.png

# Batch analyze all screenshots
python -m src.vlm.analyze --batch data/screenshots/manifest.json --output vlm_results.json
```

### RAG (Tunic Wiki Knowledge)

```bash
# Query the Tunic knowledge base
python -m src.rag.query "how to beat the garden knight boss"
python -m src.rag.query "where is the shield"
python -m src.rag.query "what are the golden coins for"

# Run quality tests
pytest tests/test_rag_quality.py -v
```

### Voice Session (requires microphone)

```bash
# Start a 30-second voice session
python -m src.voice.realtime --duration 30

# Talk to the AI — it will respond with the Tunic companion personality
```

---

## Project Structure

```
console-indie-companion/
├── src/
│   ├── capture/
│   │   ├── capture.py       # Live window capture (X11 + mss)
│   │   └── replay.py        # Replay mode (cycles through screenshots)
│   ├── vlm/
│   │   └── analyze.py       # Gemini Flash vision analysis
│   ├── voice/
│   │   └── realtime.py      # OpenAI Realtime API voice session
│   ├── rag/
│   │   ├── scrape.py        # Tunic wiki scraper
│   │   ├── index.py         # ChromaDB indexer
│   │   └── query.py         # RAG query function
│   ├── context/
│   │   └── manager.py       # Bridges VLM → voice session
│   ├── prompts/
│   │   └── tunic_companion.py  # System prompts + personality
│   ├── utils/
│   │   └── cost_tracker.py  # API cost tracking
│   └── main.py              # Main pipeline orchestrator
├── data/
│   ├── screenshots/         # 25 pre-captured Tunic screenshots
│   ├── wiki/                # 44 scraped Tunic wiki pages (JSON)
│   └── chroma/              # ChromaDB vector store (151 chunks)
├── tests/
│   └── test_rag_quality.py  # RAG quality tests (10 queries)
├── .env                     # API keys (gitignored)
├── .env.example             # Template for API keys
└── pyproject.toml           # Python dependencies
```

---

## How It Works

### Pipeline Flow

```
┌─────────────┐
│   PS5       │  Playing Tunic on TV with controller
│  (via TV)   │
└──────┬──────┘
       │ PS Remote Play
       ▼
┌─────────────┐
│  Your PC    │
│             │
│  ┌────────┐ │  Every 3 seconds
│  │Capture │─┼──────────────────────────────┐
│  └────────┘ │                              │
│             │                              ▼
│  ┌────────┐ │                      ┌──────────────┐
│  │ VLM    │◄├──────────────────────┤ Gemini Flash │
│  │Analysis│ │                      │  (cloud API) │
│  └───┬────┘ │                      └──────────────┘
│      │      │
│      ▼      │
│  ┌────────┐ │  Scene description
│  │Context │ │  + RAG knowledge
│  │Manager │ │
│  └───┬────┘ │
│      │      │
│      ▼      │
│  ┌────────┐ │  Context injection
│  │ Voice  │◄├──────────────────────┐
│  │Session │ │                      │
│  └───┬────┘ │              ┌───────┴────────┐
│      │      │              │ OpenAI Realtime│
│      │      │              │   (cloud API)  │
│      ▼      │              └────────────────┘
│  Headphones │  You talk, AI responds
│             │
└─────────────┘
```

### Key Features

- **Screen Understanding**: Gemini 2.5 Flash analyzes screenshots to understand location, enemies, player health, UI state
- **Game Knowledge**: RAG from 44 Tunic wiki pages provides spoiler-free hints and lore
- **Graduated Hints**: AI gives vague nudges first, only reveals solutions if you explicitly ask
- **Voice Interaction**: OpenAI Realtime API for natural speech-to-speech conversation (~400ms latency)
- **Session Rotation**: Automatically rotates voice sessions at 55 minutes (before 60-min API limit)
- **Cost Tracking**: Logs all API calls and estimates costs

---

## Cost Estimates

### Per Hour of Gameplay

| Component | Model | Cost/Hour |
|-----------|-------|-----------|
| Vision (Gemini 2.5 Flash) | Default | ~$0.54/hr |
| Vision (Gemini 2.5 Flash-Lite) | Budget | ~$0.18/hr |
| Voice (OpenAI Realtime) | gpt-4o-realtime | ~$3.60/hr |
| **Total (Flash)** | | **~$4.14/hr** |
| **Total (Flash-Lite)** | | **~$3.78/hr** |

**Assumptions**: 1 screenshot every 3 seconds (1,200/hr), ~500 input tokens + 100 output tokens per VLM call, continuous voice conversation.

**To reduce costs:**
- Use `--model gemini-2.5-flash-lite` (3x cheaper vision)
- Use `--no-voice` for testing (skips OpenAI Realtime)
- Increase `--interval` to 5-10 seconds (fewer VLM calls)

---

## Development Workflow

### 1. Scrape Fresh Wiki Data (Optional)

The repo includes pre-scraped wiki data. To refresh:

```bash
source venv/bin/activate
python -m src.rag.scrape   # Downloads 44 pages from tunic.fandom.com
python -m src.rag.index    # Indexes into ChromaDB (151 chunks)
```

### 2. Test RAG Quality

```bash
pytest tests/test_rag_quality.py -v
# Expected: 10/10 queries pass (100% relevance)
```

### 3. Test VLM Accuracy

```bash
python -m src.vlm.analyze --batch data/screenshots/manifest.json --output vlm_test.json
# Analyzes all 25 screenshots, saves results
```

### 4. Run End-to-End Pipeline

```bash
# Replay mode (no PS5, no voice) — safest for testing
python -m src.main --replay --no-voice --duration 60

# Check logs for:
# - "VLM (Gemini) initialised"
# - "[#1] Scene: ..." (scene descriptions)
# - "Context injected into voice session" (if voice enabled)
# - "Shutdown complete"
# - Cost summary at the end
```

---

## Live Mode (With PS5)

### Setup

1. **Install PS Remote Play** on your Linux PC:
   - Download from [PlayStation Remote Play](https://www.playstation.com/remote-play/)
   - Or use **Chiaki-ng** (open-source): `sudo apt install chiaki-ng`

2. **Connect to your PS5**:
   - Launch PS Remote Play or Chiaki-ng
   - Sign in with your PlayStation account
   - Connect to your PS5 on the same network

3. **Start Tunic on PS5**:
   - Play the game on your TV with a controller
   - The Remote Play window on your PC shows the same screen

4. **Run the companion**:
   ```bash
   source venv/bin/activate
   python -m src.main --window "PS Remote Play" --interval 3
   ```

5. **Wear PC headphones**:
   - Game audio comes from TV speakers
   - AI companion voice comes from PC headphones
   - Talk to the AI via PC microphone

### Audio Setup

- **Game audio**: Mute the PS Remote Play window audio output (Settings → Audio). Hear game audio from your TV.
- **AI voice**: Comes through PC speakers/headphones.
- **Your voice**: PC microphone picks up your questions.

---

## Troubleshooting

### "GEMINI_API_KEY not found"
- Check `.env` file exists in project root
- Verify the key is set: `cat .env | grep GEMINI`
- Reload: `source venv/bin/activate` and re-run

### "Could not locate capture source"
- For live mode: Ensure PS Remote Play or Chiaki-ng window is open and visible
- Check window name: `xdotool search --name "PS Remote Play"` should return a window ID
- Try a different window name: `--window "Chiaki"` or `--window "chiaki"`

### "Rate limited" or "429 errors"
- Gemini Flash has rate limits. The code retries with exponential backoff.
- If persistent, increase `--interval` to 5-10 seconds
- Or switch to `--model gemini-2.5-flash-lite`

### Voice session doesn't start
- Check `OPENAI_API_KEY` is set in `.env`
- Verify microphone/speakers are working: `python -c "import sounddevice; print(sounddevice.query_devices())"`
- Test without voice first: `--no-voice`

### "No module named 'src'"
- Ensure you're in the project root: `/home/bostjan/source/console-indie-companion`
- Activate venv: `source venv/bin/activate`
- Reinstall: `pip install -e .`

---

## Testing

### Run All Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

### Individual Test Suites

```bash
# RAG quality (10 Tunic questions)
pytest tests/test_rag_quality.py -v

# Expected: 10/10 PASS (100% relevance)
```

---

## API Pricing Reference

### Gemini 2.5 Flash (Stable)
- **Input**: $0.30 per 1M tokens (text/image/video)
- **Output**: $2.50 per 1M tokens
- **Use case**: Best quality, production-ready

### Gemini 2.5 Flash-Lite (Budget)
- **Input**: $0.10 per 1M tokens
- **Output**: $0.40 per 1M tokens
- **Use case**: Cost-sensitive, high-throughput tasks

### OpenAI Realtime API
- **Audio**: ~$0.06 per minute of conversation
- **Use case**: Speech-to-speech voice interaction

**Note**: Gemini batch mode can halve costs (~$0.15 input / $1.25 output for Flash), but not implemented in this POC.

---

## Next Steps (Beyond POC)

- [ ] Mobile app (Flutter/React Native) to replace PC headphones
- [ ] Cloud deployment (user doesn't need a PC)
- [ ] Multi-game support (Hollow Knight, Hades, Stardew Valley)
- [ ] Persistent memory across sessions
- [ ] Proactive AI commentary (optional toggle)
- [ ] Local YOLO/OCR for event detection (reduce VLM calls)
- [ ] Cost optimization (frame selection, caching, batch API)

---

## Architecture Decisions

### Why PS Remote Play instead of HDMI capture card?
- **Software-only**: No $100-250 hardware required
- **Scalable**: Works for any user with a PS5 and PC
- **Latency**: ~60-150ms is acceptable for a companion (not real-time gameplay)

### Why Gemini Flash instead of GPT-4o Vision?
- **Cost**: 3x cheaper than GPT-4o Vision for similar quality
- **Speed**: ~400ms latency for scene understanding
- **Context window**: 1M+ tokens (can handle long conversations)

### Why OpenAI Realtime API instead of custom STT+TTS?
- **Simplicity**: Speech-to-speech in one API call
- **Latency**: ~200-400ms end-to-end (vs 1-2s for separate STT→LLM→TTS)
- **Quality**: Natural conversation flow with built-in VAD

### Why desktop POC instead of mobile?
- **Faster iteration**: No mobile app build/deploy cycle
- **Easier testing**: Direct access to logs, debugger, file system
- **Proves core tech**: Vision + voice pipeline is the hard part

---

## Known Limitations (POC)

- **Linux only**: Window capture uses X11-specific tools (xdotool, mss)
- **Single game**: Tunic-specific prompts and knowledge base
- **No persistent memory**: Each session starts fresh
- **Reactive only**: AI waits for you to speak (no proactive hints)
- **Desktop only**: No mobile app (you wear PC headphones while playing on TV)
- **No session history**: Conversation context is lost after 60 minutes (session rotation preserves only recent context)

---

## License

MIT
