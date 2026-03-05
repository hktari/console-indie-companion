# Detector Engine & RAG Orchestration Implementation

## Summary

Successfully ported the OpenClaw plugin's detector engine and RAG orchestration patterns into the Python console companion project.

## Key Changes

### 1. Detector Engine (`src/detector/`)

**New Files:**
- `src/detector/__init__.py` - Package exports
- `src/detector/engine.py` - Core detector engine with event-based architecture
- `src/detector/tunic_detectors.py` - OpenCV-based Tunic detectors (death, health)

**Architecture:**
- `DetectorEvent` dataclass for structured event data
- `FrameDetector` protocol for pluggable detectors
- `DetectorEngine` class with per-detector error isolation
- Frame-based detection using OpenCV (cv2) for pixel analysis

**Detectors Implemented:**
- `TunicDeathDetector` - Detects death via black screen or red corner vignette
- `TunicHealthDetector` - Detects health state via HUD pixel checks and red corner

### 2. RAG Orchestration (`src/rag/`)

**New Files:**
- `src/rag/orchestrator.py` - Core orchestration layer
- `src/rag/local_retriever.py` - ChromaDB local knowledge retriever
- `src/rag/exa_retriever.py` - Exa AI web search fallback retriever

**Architecture:**
- `RetrievalResult` dataclass for unified result format
- `KnowledgeRetriever` protocol for pluggable retrievers
- `KnowledgeOrchestrator` class that merges and sorts results by confidence
- Automatic fallback to Exa when `EXA_API_KEY` is present

**Features:**
- Multi-source retrieval (local KB + web search)
- Confidence-based result ranking
- Per-retriever error isolation
- Optional LLM query enhancement for Exa searches

### 3. VLM Analysis Scheduler

**Dual-Trigger System:**
1. **Periodic trigger** - Every 5 seconds (configurable via `vlm_interval` parameter)
2. **VAD trigger** - Immediate analysis on user speech start (via voice session callback)

**Implementation:**
- Added `_vlm_trigger_callback` field to `VoiceSession`
- Callback invoked in `input_audio_buffer.speech_started` event handler
- Main pipeline uses `asyncio.Event` for trigger coordination
- Debouncing via `last_vlm_time` tracking

### 4. Context Manager Integration

**Updated `src/context/manager.py`:**
- Added `orchestrator` parameter to constructor
- Replaced direct `query_tunic_knowledge` calls with orchestrator usage
- `get_rag_context()` now queries all registered retrievers and formats top-3 results

### 5. Main Pipeline Integration

**Updated `src/main.py`:**
- Initialize `KnowledgeOrchestrator` with local + Exa retrievers
- Initialize `DetectorEngine` with Tunic detectors
- Pass orchestrator to `ContextManager`
- Updated `main_pipeline()` signature to accept `detector_engine` and `vlm_interval`
- Implemented dual-trigger VLM analysis logic
- Run frame detectors on each VLM analysis and inject events to voice

### 6. Dependencies

**Added to `pyproject.toml`:**
- `opencv-python>=4.8.0` - For frame-based detection

## Testing

**Created Tests:**
- `tests/test_detector_engine.py` - Detector engine unit tests
  - Registration, event emission, error isolation, multi-detector aggregation
- `tests/test_rag_orchestration.py` - RAG orchestration unit tests
  - Retriever registration, result merging, confidence sorting, error isolation

**Run Tests:**
```bash
pytest tests/test_detector_engine.py tests/test_rag_orchestration.py
```

## Usage

### Running with New Features

```bash
# Install dependencies (includes opencv-python)
pip install -e .

# Run with voice (enables VAD-triggered VLM analysis)
python -m src.main --window "PS Remote Play"

# Run in replay mode (periodic VLM only)
python -m src.main --replay --screenshot-dir data/screenshots/

# Run without voice (periodic VLM only, no VAD trigger)
python -m src.main --window "PS Remote Play" --no-voice
```

### Configuration

**Environment Variables:**
- `EXA_API_KEY` - Enable Exa web search fallback (optional)
- `OPENAI_API_KEY` - Required for voice + optional for Exa query enhancement
- `GEMINI_API_KEY` - Required for VLM analysis

**VLM Analysis Cadence:**
- Default: 5 seconds periodic + VAD speech trigger
- Configurable via `vlm_interval` parameter in `main_pipeline()`

## Migration Notes

### Breaking Changes
- Old scene-based `DeathDetector` in `src/context/detectors.py` is no longer used
- `ContextManager` now requires `orchestrator` parameter (optional, but recommended)
- `main_pipeline()` signature changed to accept `detector_engine` instead of `detectors`

### Backward Compatibility
- `query_tunic_knowledge()` still available in `src/rag/query.py` for direct usage
- Context manager works without orchestrator (returns empty RAG context)

## Architecture Decisions

1. **OpenCV for detection** - Enables pixel-level analysis for health/death detection
2. **Event-based detectors** - Structured events with confidence scores vs. string messages
3. **Dual-trigger VLM** - Balance between cost (periodic) and responsiveness (VAD)
4. **Multi-source RAG** - Local KB prioritized, Exa as fallback/augmentation
5. **Error isolation** - Per-detector and per-retriever failure handling

## Known Limitations

1. OpenCV detectors are tuned for 1920x1080 resolution (auto-scales for other resolutions)
2. Exa retriever requires API key (gracefully degrades if missing)
3. VAD trigger may increase VLM API costs if user speaks frequently
4. Detector thresholds may need tuning for different game versions/graphics settings

## Next Steps

1. Add more game-specific detectors (e.g., boss encounters, puzzle states)
2. Implement detector confidence calibration
3. Add retriever priority/weighting system
4. Create detector configuration file for threshold tuning
5. Add telemetry for VLM trigger frequency analysis
