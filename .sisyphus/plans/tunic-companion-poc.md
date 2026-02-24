# Tunic Voice Companion — Desktop POC

## TL;DR

> **Quick Summary**: Build a desktop proof-of-concept that captures PS5 gameplay (Tunic) via PS Remote Play on Linux, sends periodic screenshots to Gemini Flash for scene understanding, and provides a voice companion via OpenAI Realtime API — all running on the user's local PC.
>
> **Deliverables**:
> - Python desktop app that captures PS Remote Play window on Linux
> - Vision pipeline: periodic screenshots → Gemini Flash → structured scene descriptions
> - Voice companion: OpenAI Realtime API with Tunic-specific knowledge (RAG from wiki)
> - Replay mode for development/testing without a live PS5
>
> **Estimated Effort**: Medium (2-3 days of focused work)
> **Parallel Execution**: YES — 4 waves
> **Critical Path**: Scaffolding → Validation Spikes → Integration → Demo

---

## Context

### Original Request
User wants to refine the Console Indie Companion MVP. Specifically, they want to capture the PlayStation screen display, run it through a vision model on a PC, and connect the AI to a voice chat interface. The initial POC runs entirely on the user's Linux PC — no mobile app, no cloud infrastructure.

### Interview Summary
**Key Discussions**:
- PS5 capture methods: HDMI cards, Chiaki-ng, PS Remote Play, RTMP streaming — decided on PS Remote Play (software-only)
- Vision AI models: Gemini Flash ($0.18/hr), GPT-5 Nano ($0.45/hr), Claude Sonnet ($5.40/hr) — decided on Gemini Flash
- Frame sampling: periodic screenshots every 3-5 seconds (no YOLO/OCR/SSIM for POC)
- Voice: OpenAI Realtime API from day one (speech-to-speech, ~400ms latency)
- Test game: Tunic (puzzles + secrets + combat — ideal for graduated hints)
- OS: Linux with X11/xdotool for window capture
- Audio: Game from TV speakers, AI voice from PC headphones

**Research Findings**:
- Chiaki-ng can extract raw AVFrames via FFmpeg decoder, but no official headless mode
- pyremoteplay (Python Remote Play API) is ARCHIVED as of Nov 2025 — unusable
- lorcan2440/View-PS5-Screen-Remote-Play demonstrates PS5→OpenCV pipeline via window capture
- Questie.ai separates Vision Encoder from Dialogue LLM, uses RAG for memory
- OpenAI Realtime API has 60-minute session limit — needs rotation strategy
- Context injection: `conversation.item.create` with `role: "system"` for mid-conversation updates

### Metis Review
**Identified Gaps** (addressed):
- pyremoteplay archived → removed, using PS Remote Play window capture instead
- 60-min Realtime API session limit → session rotation at 55 minutes
- Context injection timing → queue updates, inject during silence gaps only
- Tunic's fictional in-game language → handle explicitly in system prompt
- No .gitignore → add before any code
- No replay mode → build for dev without PS5
- No acceptance criteria → executable test scripts for each module
- Window capture is OS-specific → Linux-specific implementation with X11 + mss

---

## Work Objectives

### Core Objective
Prove that a PS5 game screen can be captured, understood by a vision AI, and used to power a contextually-aware voice companion — running entirely on a Linux desktop.

### Concrete Deliverables
- `src/capture/` — Linux window capture module (X11 + mss)
- `src/vlm/` — Gemini Flash vision integration with Tunic-specific prompting
- `src/voice/` — OpenAI Realtime API voice loop with context injection
- `src/rag/` — Tunic wiki scraper + ChromaDB retrieval
- `src/context/` — Context manager bridging VLM → voice session
- `src/main.py` — Orchestrator wiring all modules
- `tests/` — Validation spike scripts and test data
- `data/screenshots/` — Pre-captured Tunic screenshots for replay mode

### Definition of Done
- [ ] Can capture PS Remote Play window screenshots reliably on Linux
- [ ] VLM correctly describes Tunic game scenes (≥80% accuracy on 20 test screenshots)
- [ ] Voice companion responds contextually to game state ("what boss is this?" → correct answer)
- [ ] RAG retrieves relevant Tunic wiki content for common questions
- [ ] Full pipeline runs for 10+ minutes without crashes
- [ ] Replay mode works with pre-captured screenshots (no PS5 needed)

### Must Have
- Voice companion that can see and understand Tunic gameplay
- Graduated hints (vague → specific → solution, only if asked)
- Spoiler awareness (don't reveal what player hasn't discovered)
- Reactive-only AI (speaks when user speaks, not proactive)
- Session rotation handling for >60 minute play sessions

### Must NOT Have (Guardrails)
- No mobile app — desktop only
- No SSIM change detection, YOLO object detection, or PaddleOCR — simple periodic screenshots only
- No multi-model VLM fallback chains — Gemini Flash only
- No custom TTS/STT — use Realtime API's built-in speech handling
- No proactive AI commentary — reactive only
- No persistent memory across sessions — single session only
- No user progress tracking database — rely on VLM scene understanding
- No cloud infrastructure (AWS/GCP) — local PC only for POC
- No multi-game support — Tunic only
- No pyremoteplay library (archived, dead)
- No over-engineering: simple Python scripts, not a framework

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.
> Acceptance criteria requiring "user manually tests/confirms" are FORBIDDEN.

### Test Decision
- **Infrastructure exists**: NO (greenfield project)
- **Automated tests**: Tests-after (validation spike scripts serve as integration tests)
- **Framework**: pytest for unit tests, standalone scripts for spike validation

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Screen capture**: Bash — run capture script, verify file output
- **VLM**: Bash — send test screenshot, verify response quality
- **Voice**: interactive_bash (tmux) — run voice session, inject test context, verify response
- **RAG**: Bash — run query test, verify retrieval relevance
- **Integration**: interactive_bash (tmux) — run full pipeline, verify 10-minute stability

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation — start immediately):
├── Task 1: Project scaffolding + .gitignore + dependencies [quick]
├── Task 2: Tunic test screenshot collection (20+ screenshots) [quick]
└── Task 3: Tunic wiki scraper + RAG pipeline [unspecified-high]

Wave 2 (Validation Spikes — after Wave 1, MAX PARALLEL):
├── Task 4: Screen capture spike (Linux/X11 window capture) [deep]
├── Task 5: VLM accuracy spike (Gemini Flash + Tunic screenshots) [deep]
├── Task 6: Realtime API voice spike (context injection test) [deep]
└── Task 7: RAG retrieval quality spike (query testing) [quick]

Wave 3 (Integration — after Wave 2):
├── Task 8: Context manager module [deep]
├── Task 9: System prompt engineering (Tunic personality + graduated hints) [artistry]
├── Task 10: Main pipeline orchestrator [deep]
└── Task 11: Replay mode (pre-captured screenshots for dev) [quick]

Wave 4 (Demo + Polish — after Wave 3):
├── Task 12: End-to-end demo run (10-minute Tunic session) [unspecified-high]
└── Task 13: Cost tracking + session rotation [quick]

Wave FINAL (Verification — after ALL tasks):
├── Task F1: Plan compliance audit [oracle]
├── Task F2: Code quality review [unspecified-high]
├── Task F3: End-to-end QA [unspecified-high]
└── Task F4: Scope fidelity check [deep]

Critical Path: Task 1 → Task 4 → Task 8 → Task 10 → Task 12 → F1-F4
Parallel Speedup: ~60% faster than sequential
Max Concurrent: 4 (Waves 2 & 3)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1 | — | 2-7 | 1 |
| 2 | — | 5, 11 | 1 |
| 3 | — | 7 | 1 |
| 4 | 1 | 8, 10 | 2 |
| 5 | 1, 2 | 9, 10 | 2 |
| 6 | 1 | 8, 10 | 2 |
| 7 | 1, 3 | 10 | 2 |
| 8 | 4, 5, 6 | 10 | 3 |
| 9 | 5 | 10 | 3 |
| 10 | 4-9, 11 | 12, 13 | 3 |
| 11 | 2 | 10 | 3 |
| 12 | 10 | F1-F4 | 4 |
| 13 | 10 | F1-F4 | 4 |

### Agent Dispatch Summary

- **Wave 1**: 3 tasks — T1 `quick`, T2 `quick`, T3 `unspecified-high`
- **Wave 2**: 4 tasks — T4 `deep`, T5 `deep`, T6 `deep`, T7 `quick`
- **Wave 3**: 4 tasks — T8 `deep`, T9 `artistry`, T10 `deep`, T11 `quick`
- **Wave 4**: 2 tasks — T12 `unspecified-high`, T13 `quick`
- **FINAL**: 4 tasks — F1 `oracle`, F2 `unspecified-high`, F3 `unspecified-high`, F4 `deep`

---

## TODOs

- [ ] 1. Project Scaffolding + Dependencies

  **What to do**:
  - Create `pyproject.toml` with project metadata and dependencies:
    - `mss` (screen capture), `google-generativeai` (Gemini Flash API), `openai` (Realtime API)
    - `chromadb` (vector store), `beautifulsoup4` + `requests` (wiki scraping)
    - `python-dotenv` (env vars), `pytest` (testing)
  - Create directory structure: `src/capture/`, `src/vlm/`, `src/voice/`, `src/rag/`, `src/context/`, `tests/`, `data/screenshots/`
  - Create `.gitignore` (exclude `.env`, `data/screenshots/*.png`, `__pycache__/`, `*.pyc`, `.sisyphus/evidence/`)
  - Create `.env.example` with placeholder API keys: `GEMINI_API_KEY=`, `OPENAI_API_KEY=`
  - Create `src/__init__.py` and subpackage `__init__.py` files
  - Install dependencies with `pip install -e .` or `pip install -r requirements.txt`

  **Must NOT do**:
  - No Docker setup, no Makefile, no CI/CD config
  - No framework (Flask, FastAPI) — this is a desktop script

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Tasks 4, 5, 6, 7
  - **Blocked By**: None

  **References**:
  - `package.json` — existing project file (only has oh-my-opencode dep; Python project is new)
  - `README.md` — project structure section for context
  - Reference: `lorcan2440/View-PS5-Screen-Remote-Play` on GitHub for typical Python game capture deps

  **Acceptance Criteria**:
  - [ ] `pyproject.toml` or `requirements.txt` exists with all deps listed
  - [ ] All directories created with `__init__.py` files
  - [ ] `.gitignore` blocks `.env` and screenshot data
  - [ ] `pip install -e .` succeeds without errors

  ```
  Scenario: Dependencies install cleanly
    Tool: Bash
    Steps:
      1. Run `pip install -e .` (or `pip install -r requirements.txt`)
      2. Run `python -c "import mss; import google.generativeai; import openai; import chromadb; print('OK')"` 
    Expected Result: Exit code 0, prints 'OK'
    Evidence: .sisyphus/evidence/task-1-deps-install.txt

  Scenario: Project structure exists
    Tool: Bash
    Steps:
      1. Run `find src -name '__init__.py' | wc -l`
      2. Run `test -f .gitignore && echo 'OK'`
      3. Run `test -f .env.example && echo 'OK'`
    Expected Result: At least 6 __init__.py files, .gitignore and .env.example exist
    Evidence: .sisyphus/evidence/task-1-structure.txt
  ```

  **Commit**: YES (group with Wave 1)
  - Message: `feat(scaffold): project setup with dependencies and test data`
  - Files: `pyproject.toml`, `src/**/__init__.py`, `.gitignore`, `.env.example`

- [ ] 2. Tunic Test Screenshot Collection

  **What to do**:
  - Collect 20+ diverse Tunic gameplay screenshots from publicly available sources:
    - Google Image search for "Tunic gameplay screenshot" (various game states)
    - Steam store page screenshots
    - Tunic fandom wiki images
    - YouTube video thumbnails/frames from Tunic gameplay videos
  - Screenshots should cover diverse game states:
    - Overworld exploration (at least 5)
    - Boss fights (at least 3)
    - Inventory/manual page screens (at least 3)
    - Puzzle areas (at least 3)
    - NPCs / dialogue scenes (at least 2)
    - Map screen (at least 2)
    - Death screen / loading screen (at least 2)
  - Save as PNG files in `data/screenshots/` with descriptive names: `tunic_boss_garden_knight.png`, `tunic_overworld_forest.png`, etc.
  - Create `data/screenshots/manifest.json` listing each screenshot with metadata: `{filename, game_state, location, description}`

  **Must NOT do**:
  - No copyrighted content from guides/books — publicly available gameplay screenshots only
  - No need to actually play the game — source from public media

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`playwright`]
    - `playwright`: For browsing and downloading screenshots from web sources

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Tasks 5, 11
  - **Blocked By**: None

  **References**:
  - Tunic Steam store page: https://store.steampowered.com/app/553420/TUNIC/
  - Tunic fandom wiki: https://tunic.fandom.com/wiki/ (image galleries)
  - YouTube: search "Tunic gameplay" for diverse game states

  **Acceptance Criteria**:
  - [ ] At least 20 PNG files in `data/screenshots/`
  - [ ] `manifest.json` exists with metadata for each screenshot
  - [ ] Screenshots cover at least 5 distinct game states

  ```
  Scenario: Screenshots collected and catalogued
    Tool: Bash
    Steps:
      1. Run `ls data/screenshots/*.png | wc -l`
      2. Run `python -c "import json; m=json.load(open('data/screenshots/manifest.json')); print(len(m)); assert len(m) >= 20"` 
      3. Run `python -c "import json; m=json.load(open('data/screenshots/manifest.json')); states=set(s['game_state'] for s in m); print(states); assert len(states) >= 5"` 
    Expected Result: ≥20 files, manifest has ≥20 entries, ≥5 distinct game states
    Evidence: .sisyphus/evidence/task-2-screenshots.txt
  ```

  **Commit**: YES (group with Wave 1)
  - Message: `feat(scaffold): project setup with dependencies and test data`
  - Files: `data/screenshots/*.png`, `data/screenshots/manifest.json`

- [ ] 3. Tunic Wiki Scraper + RAG Pipeline

  **What to do**:
  - Build a scraper for https://tunic.fandom.com/wiki/ that:
    - Discovers all wiki pages via `Special:AllPages` or sitemap
    - Downloads page content (HTML → clean text, strip navigation/ads)
    - Extracts: page title, sections, text content, links
  - Build a RAG indexing pipeline:
    - Chunk wiki pages into ~500-token chunks with overlap
    - Generate embeddings (use Gemini or OpenAI embedding API, or sentence-transformers locally)
    - Store in ChromaDB (in-memory for POC, persistent optional)
  - Build a query function: `query_tunic_knowledge(question: str) -> list[str]` that returns top-5 relevant chunks
  - Save scraped data to `data/wiki/` as JSON for caching (don't re-scrape every run)

  **Must NOT do**:
  - No crawling beyond tunic.fandom.com domain
  - No persistent vector DB server — in-memory ChromaDB is fine
  - No complex chunking strategies — simple fixed-size with overlap

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: Task 7
  - **Blocked By**: None

  **References**:
  - Wiki URL: https://tunic.fandom.com/wiki/
  - Reference tool: `JOHW85/ScrapeFandom` (GitHub, 43 stars) — full wiki dump
  - Reference tool: `GOLEM-lab/fandom-wiki` (GitHub, 27 stars) — structured extraction
  - ChromaDB docs: https://docs.trychroma.com/ for embedding + retrieval patterns

  **Acceptance Criteria**:
  - [ ] Scraper downloads ≥50 wiki pages to `data/wiki/`
  - [ ] ChromaDB collection created with embedded chunks
  - [ ] `query_tunic_knowledge()` returns relevant results for test queries

  ```
  Scenario: Wiki scraper downloads Tunic content
    Tool: Bash
    Steps:
      1. Run `python -m src.rag.scrape`
      2. Run `ls data/wiki/*.json | wc -l`
    Expected Result: ≥50 JSON files in data/wiki/
    Evidence: .sisyphus/evidence/task-3-scraper.txt

  Scenario: RAG retrieves relevant Tunic knowledge
    Tool: Bash
    Steps:
      1. Run `python -m src.rag.query 'how to beat the garden knight boss'`
      2. Verify output contains text about Garden Knight boss fight
      3. Run `python -m src.rag.query 'what are the golden path pages'`
      4. Verify output contains text about manual pages / golden path
    Expected Result: Both queries return Tunic-relevant content
    Failure Indicators: Empty results, generic non-Tunic content, errors
    Evidence: .sisyphus/evidence/task-3-rag-queries.txt
  ```

  **Commit**: YES (group with Wave 1)
  - Message: `feat(scaffold): project setup with dependencies and test data`
  - Files: `src/rag/scrape.py`, `src/rag/query.py`, `src/rag/__init__.py`, `data/wiki/`

- [ ] 4. Screen Capture Spike (Linux/X11 Window Capture)

  **What to do**:
  - Build `src/capture/capture.py` that:
    - Uses `subprocess` + `xdotool` to find the PS Remote Play or Chiaki-ng window by name
    - Uses `mss` to capture that window's screen region
    - Captures at configurable interval (default 3 seconds)
    - Saves as JPEG (lower filesize for API calls) or PNG
    - Handles: window not found, window minimized, window moved
    - Exposes `CaptureService` class with `start()`, `stop()`, `capture_once()`, and `get_latest_frame()` methods

  **Must NOT do**:
  - Use pyremoteplay, implement SSIM/change detection, use dxcam (Windows-only)

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6, 7)
  - **Blocks**: Tasks 8, 10
  - **Blocked By**: Task 1

  **References**:
  - `mss` library docs
  - `xdotool search --name` for window finding
  - `lorcan2440/View-PS5-Screen-Remote-Play` on GitHub

  **Acceptance Criteria**:
  - [ ] `src/capture/capture.py` exists and implements `CaptureService`
  - [ ] Can find and capture a specific window by name
  - [ ] Handles window-not-found errors gracefully

  ```
  Scenario: Capture from a test window
    Tool: Bash
    Steps:
      1. Open a test window (e.g., a browser or terminal)
      2. Run `python -m src.capture.capture --window "Test Window Name" --count 10 --output /tmp/capture_test`
      3. Verify 10 files exist in /tmp/capture_test and each is > 10KB
    Expected Result: 10 images captured successfully
    Evidence: .sisyphus/evidence/task-4-capture-test.txt

  Scenario: Handle window not found
    Tool: Bash
    Steps:
      1. Run `python -m src.capture.capture --window "NonExistentWindowName"`
    Expected Result: Exit with clear error message "Window not found"
    Evidence: .sisyphus/evidence/task-4-window-not-found.txt
  ```

  **Commit**: YES (group with Wave 2)
  - Message: `feat(spikes): validation spikes for capture, VLM, voice, and RAG`
  - Files: `src/capture/capture.py`

- [ ] 5. VLM Accuracy Spike (Gemini Flash + Tunic Screenshots)

  **What to do**:
  - Build `src/vlm/analyze.py` that:
    - Loads a screenshot from file path
    - Sends to Google Gemini Flash API with a Tunic-specific prompt
    - Prompt should ask for structured JSON output: `{location, enemies, health_status, ui_state, activity, description}`
    - The prompt should explain Tunic's visual style (isometric, pixel-art-ish, uses a fictional in-game language)
    - Returns structured scene description
    - Handles: API errors, rate limiting, empty/corrupt images
  - Run against ALL screenshots in `data/screenshots/manifest.json`
  - Score accuracy: does the VLM correctly identify game state, location, enemies? Target: ≥80%

  **Must NOT do**:
  - Use multiple VLM models, implement caching, add conversation history

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 6, 7)
  - **Blocks**: Tasks 9, 10
  - **Blocked By**: Tasks 1, 2

  **References**:
  - Google Generative AI Python SDK docs
  - `data/screenshots/manifest.json` for test data

  **Acceptance Criteria**:
  - [ ] `src/vlm/analyze.py` exists and returns structured JSON
  - [ ] VLM correctly identifies game state in ≥80% of test screenshots
  - [ ] Handles API errors and rate limiting gracefully

  ```
  Scenario: VLM analysis of test screenshots
    Tool: Bash
    Steps:
      1. Run `python -m src.vlm.analyze --manifest data/screenshots/manifest.json --limit 5`
      2. Verify output JSON structure for each screenshot
      3. Verify at least 4/5 have correct game state description
    Expected Result: Structured JSON output with ≥80% accuracy
    Evidence: .sisyphus/evidence/task-5-vlm-accuracy.txt
  ```

  **Commit**: YES (group with Wave 2)
  - Message: `feat(spikes): validation spikes for capture, VLM, voice, and RAG`
  - Files: `src/vlm/analyze.py`

- [ ] 6. Realtime API Voice Spike (Context Injection Test)

  **What to do**:
  - Build `src/voice/realtime.py` that:
    - Connects to OpenAI Realtime API via WebSocket
    - Sets up audio input from PC microphone (using `pyaudio` or `sounddevice`)
    - Sets up audio output to PC speakers/headphones
    - Implements the voice conversation loop (VAD, turn detection handled by API)
    - Supports injecting game context via `conversation.item.create` with `role: "system"` messages
    - Initial system instructions set personality to "knowledgeable friend who's beaten Tunic"
    - Exposes `VoiceSession` class with `start()`, `stop()`, `inject_context(text: str)`, and `on_response(callback)` methods
    - Handles: WebSocket disconnection, API errors, graceful shutdown

  **Must NOT do**:
  - Implement session rotation yet (Task 13), build custom VAD/STT/TTS, implement wake words
,
  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5, 7)
  - **Blocks**: Tasks 8, 10
  - **Blocked By**: Task 1

  **References**:
  - OpenAI Realtime API docs (https://platform.openai.com/docs/guides/realtime)
  - `conversation.item.create` system message format
  - `session.update` for initial instructions

  **Acceptance Criteria**:
  - [ ] `src/voice/realtime.py` exists and implements `VoiceSession`
  - [ ] Can start a voice session and run for 30 seconds without crashing
  - [ ] AI references injected context in its response

  ```
  Scenario: Voice session with context injection
    Tool: interactive_bash (tmux)
    Steps:
      1. Start a voice session
      2. Inject test context: "The player is fighting the Garden Knight boss in the garden area. Their health is at 50%."
      3. Ask a question via voice or verify AI references the Garden Knight in its next response
    Expected Result: AI acknowledges the Garden Knight and the player's situation
    Evidence: .sisyphus/evidence/task-6-voice-context.txt
  ```

  **Commit**: YES (group with Wave 2)
  - Message: `feat(spikes): validation spikes for capture, VLM, voice, and RAG`
  - Files: `src/voice/realtime.py`

- [ ] 7. RAG Retrieval Quality Spike

  **What to do**:
  - Create `tests/test_rag_quality.py` that:
    - Defines 10 common Tunic questions players ask
    - Runs each through `query_tunic_knowledge()` from Task 3
    - Evaluates: does the returned content actually answer the question?
    - Scoring: relevant (2), partially relevant (1), irrelevant (0). Target: average ≥1.4 (70% quality)
    - Test questions should cover: boss strategies, puzzle solutions, secret locations, item uses, navigation help

  **Must NOT do**:
  - Modify the RAG pipeline itself (that's Task 3). Just test it.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5, 6)
  - **Blocks**: Task 10
  - **Blocked By**: Tasks 1, 3

  **References**:
  - `src/rag/query.py` from Task 3

  **Acceptance Criteria**:
  - [ ] `tests/test_rag_quality.py` exists
  - [ ] Average relevance score ≥ 1.4 across 10 test questions

  ```
  Scenario: RAG quality evaluation
    Tool: Bash
    Steps:
      1. Run `pytest tests/test_rag_quality.py -v`
    Expected Result: ≥7/10 queries return relevant results, average score ≥ 1.4
    Evidence: .sisyphus/evidence/task-7-rag-quality.txt
  ```

  **Commit**: YES (group with Wave 2)
  - Message: `feat(spikes): validation spikes for capture, VLM, voice, and RAG`
  - Files: `tests/test_rag_quality.py`

- [ ] 8. Context Manager Module

  **What to do**:
  - Build `src/context/manager.py` that:
    - Receives scene descriptions from VLM module (Task 5)
    - Receives RAG results relevant to current scene
    - Maintains a context buffer with the last 5 scene descriptions (rolling window)
    - Formats context as a system message: combines VLM scene description + RAG knowledge into a single coherent context update
    - Queues context updates and only injects them into the voice session (Task 6) during silence gaps (between `response.done` and next user input)
    - Exposes `ContextManager` class with `update_scene(description)`, `update_rag(results)`, `get_pending_context()`, and `flush_to_voice(voice_session)` methods

  **Must NOT do**:
  - Build persistent memory, implement complex state machines, track user progress

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 9, 10, 11)
  - **Blocks**: Task 10
  - **Blocked By**: Tasks 4, 5, 6

  **References**:
  - `src/vlm/analyze.py` (output format)
  - `src/voice/realtime.py` (inject_context API)
  - `src/rag/query.py`

  **Acceptance Criteria**:
  - [ ] `src/context/manager.py` exists and implements `ContextManager`
  - [ ] Context buffer maintains only the last 5 descriptions
  - [ ] `flush_to_voice` produces a well-structured system message

  ```
  Scenario: Context buffer and formatting
    Tool: Bash
    Steps:
      1. Run a unit test that feeds 5 scene descriptions
      2. Verify the context buffer maintains only the last 5, formatted correctly
      3. Verify `flush_to_voice` output combines VLM + RAG context
    Expected Result: Correct buffer management and message formatting
    Evidence: .sisyphus/evidence/task-8-context-manager.txt
  ```

  **Commit**: YES (group with Wave 3)
  - Message: `feat(integration): context manager, prompts, and main orchestrator`
  - Files: `src/context/manager.py`

- [ ] 9. System Prompt Engineering (Tunic Personality + Graduated Hints)

  **What to do**:
  - Create `src/prompts/tunic_companion.py` containing:
    - `SYSTEM_INSTRUCTIONS` — the initial system prompt for the Realtime API session, defining:
      - Personality: Knowledgeable friend who's beaten Tunic, casual/friendly tone
      - Graduated hints: When user asks for help, provide hints in 3 levels:
        1. Vague nudge ("Have you explored all the paths in this area?")
        2. Specific hint ("There's a hidden passage behind the waterfall")
        3. Full solution (only if user explicitly asks "just tell me")
      - Spoiler awareness: Don't reveal things the player hasn't discovered yet. Use VLM context to understand what they've seen.
      - Tunic's fictional language: Explain that Tunic uses a made-up alphabet, and the VLM may see text it can't read — that's normal.
      - Reactive behavior: Only speak when spoken to. Don't interrupt gameplay.
    - `CONTEXT_UPDATE_TEMPLATE` — a template for injecting scene context: "The player is currently {description}. Based on what I can see: {vlm_output}. Relevant game knowledge: {rag_results}"
    - `VLM_ANALYSIS_PROMPT` — the prompt sent to Gemini Flash with each screenshot (may refine the one from Task 5)

  **Must NOT do**:
  - Implement character customization, multiple personalities, mood systems

  **Recommended Agent Profile**:
  - **Category**: `artistry`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 8, 10, 11)
  - **Blocks**: Task 10
  - **Blocked By**: Task 5

  **References**:
  - Task 5 VLM output format
  - Task 6 Realtime API system instructions format
  - Tunic wiki for game-accurate terminology

  **Acceptance Criteria**:
  - [ ] `src/prompts/tunic_companion.py` exists with all required templates
  - [ ] Prompts include graduated hints, spoiler awareness, and personality definitions

  ```
  Scenario: Prompt content review
    Tool: Bash
    Steps:
      1. Grep for "graduated hints", "spoiler awareness", "knowledgeable friend", and "fictional language" in `src/prompts/tunic_companion.py`
    Expected Result: All key elements are present in the prompt file
    Evidence: .sisyphus/evidence/task-9-prompt-review.txt
  ```

  **Commit**: YES (group with Wave 3)
  - Message: `feat(integration): context manager, prompts, and main orchestrator`
  - Files: `src/prompts/tunic_companion.py`

- [ ] 10. Main Pipeline Orchestrator

  **What to do**:
  - Build `src/main.py` that:
    - Parses CLI arguments: `--window` (window name), `--interval` (screenshot interval, default 3), `--replay` (use pre-captured screenshots), `--screenshot-dir` (for replay mode), `--duration` (max runtime)
    - Loads `.env` for API keys using python-dotenv
    - Initializes all modules: CaptureService, VLM analyzer, VoiceSession, RAG query, ContextManager
    - Main loop:
      1. CaptureService captures screenshot at interval
      2. VLM analyzes the screenshot → scene description
      3. RAG queries for relevant knowledge based on scene description
      4. ContextManager buffers and formats the context
      5. ContextManager injects into VoiceSession when appropriate
      6. VoiceSession handles user voice input/output independently
    - Graceful shutdown on Ctrl+C (cleanup all modules)
    - Logging: log each VLM call, context injection, and errors
    - In replay mode: reads screenshots from a directory instead of capturing live

  **Must NOT do**:
  - Implement cloud upload, web server, GUI, monitoring dashboard

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 8, 9, 11)
  - **Blocks**: Tasks 12, 13
  - **Blocked By**: Tasks 4-9, 11

  **References**:
  - All src/ modules from Tasks 4-9
  - `src/prompts/tunic_companion.py` for prompt templates

  **Acceptance Criteria**:
  - [ ] `src/main.py` exists and wires all modules correctly
  - [ ] Replay mode works as expected
  - [ ] Graceful shutdown on Ctrl+C

  ```
  Scenario: Main pipeline replay mode test
    Tool: Bash
    Steps:
      1. Run `python -m src.main --replay --screenshot-dir data/screenshots/ --duration 30`
      2. Verify logs show VLM analysis, context injections, and clean shutdown
    Expected Result: Process runs for 30s, processes screenshots, and exits cleanly
    Evidence: .sisyphus/evidence/task-10-main-replay.txt
  ```

  **Commit**: YES (group with Wave 3)
  - Message: `feat(integration): context manager, prompts, and main orchestrator`
  - Files: `src/main.py`

- [ ] 11. Replay Mode

  **What to do**:
  - Build `src/capture/replay.py` that:
    - Implements the same interface as `CaptureService` from Task 4
    - Instead of capturing a window, reads screenshots from a directory in sequence
    - Cycles through screenshots at the configured interval (simulates live gameplay)
    - When all screenshots are exhausted, loops back to the beginning
    - This enables development and testing without a live PS5

  **Must NOT do**:
  - Implement video file playback, add new dependencies

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 8, 9, 10)
  - **Blocks**: Task 10
  - **Blocked By**: Task 2

  **References**:
  - `src/capture/capture.py` (same interface)
  - `data/screenshots/` (test data from Task 2)

  **Acceptance Criteria**:
  - [ ] `src/capture/replay.py` exists and implements `CaptureService` interface
  - [ ] Cycles through screenshots at configured interval

  ```
  Scenario: Replay mode functionality
    Tool: Bash
    Steps:
      1. Run replay mode for 30 seconds
      2. Verify it cycles through screenshots and produces the same output format as live capture
    Expected Result: Correct screenshot cycling and output format
    Evidence: .sisyphus/evidence/task-11-replay-test.txt
  ```

  **Commit**: YES (group with Wave 3)
  - Message: `feat(integration): context manager, prompts, and main orchestrator`
  - Files: `src/capture/replay.py`

- [ ] 12. End-to-End Demo Run

  **What to do**:
  - Run the full pipeline in replay mode for 10 minutes:
    - `python -m src.main --replay --screenshot-dir data/screenshots/ --duration 600`
    - Verify: no crashes, VLM responses are Tunic-relevant, context is injected into voice session, voice session stays alive
    - Test specific Tunic scenarios: feed a boss fight screenshot, then ask "what should I do?" via voice — verify AI gives Tunic-specific advice
    - Log all API calls with timestamps
    - Calculate and report: total VLM cost, total Realtime API cost, average latency per VLM call
    - Save comprehensive evidence: logs, VLM outputs, cost report

  **Must NOT do**:
  - Require a live PS5 (replay mode only). Must NOT fix bugs in other modules — just report them.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Task 13)
  - **Blocks**: F1-F4
  - **Blocked By**: Task 10

  **References**:
  - All src/ modules

  **Acceptance Criteria**:
  - [ ] Pipeline runs for 10 full minutes without unhandled exceptions
  - [ ] VLM produces ≥80% relevant descriptions
  - [ ] Cost report is generated

  ```
  Scenario: 10-minute demo run
    Tool: interactive_bash (tmux)
    Steps:
      1. Run `python -m src.main --replay --screenshot-dir data/screenshots/ --duration 600`
      2. Monitor logs for stability and relevance
      3. Generate cost report from logs
    Expected Result: Stable 10-minute run with relevant AI responses and cost report
    Evidence: .sisyphus/evidence/task-12-demo-run.txt
  ```

  **Commit**: YES (group with Wave 4)
  - Message: `feat(demo): end-to-end demo run and cost tracking`
  - Files: None

- [ ] 13. Cost Tracking + Session Rotation

  **What to do**:
  - Add two features to the existing pipeline:
    1. **Cost tracking** (`src/utils/cost_tracker.py`):
       - Log every API call with: timestamp, model, input tokens, output tokens, estimated cost
       - Provide `get_session_cost()` returning total cost breakdown
       - Print cost summary on shutdown
    2. **Session rotation** (update `src/voice/realtime.py`):
       - Track session duration
       - At 55 minutes, create a new Realtime API session
       - Inject a condensed summary of the conversation + current scene context as the initial system message
       - Seamlessly switch to the new session (user should barely notice)
       - Handle rotation failures gracefully (retry once, then log and continue)

  **Must NOT do**:
  - Build a billing system, persist cost data to database, implement complex session state serialization

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Task 12)
  - **Blocks**: F1-F4
  - **Blocked By**: Task 10

  **References**:
  - `src/voice/realtime.py` from Task 6
  - OpenAI Realtime API session lifecycle docs

  **Acceptance Criteria**:
  - [ ] `src/utils/cost_tracker.py` exists and logs API calls correctly
  - [ ] Session rotation logic implemented in `src/voice/realtime.py`
  - [ ] Cost summary printed on shutdown

  ```
  Scenario: Cost tracking and session rotation
    Tool: Bash
    Steps:
      1. Run pipeline for 60 seconds and check cost report output
      2. Simulate time advancement to 55 minutes (or mock timer) and verify new session creation
    Expected Result: Correct cost logging and session rotation behavior
    Evidence: .sisyphus/evidence/task-13-cost-rotation.txt
  ```

  **Commit**: YES (group with Wave 4)
  - Message: `feat(demo): end-to-end demo run and cost tracking`
  - Files: `src/utils/cost_tracker.py`, `src/voice/realtime.py`
---

## Final Verification Wave

> 4 review agents run in PARALLEL. ALL must APPROVE. Rejection → fix → re-run.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run linter + type check if applicable. Review all Python files for: `# type: ignore`, bare `except:`, `print()` in prod code, unused imports, hardcoded API keys. Check for AI slop: excessive comments, over-abstraction, generic variable names.
  Output: `Lint [PASS/FAIL] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **End-to-End QA** — `unspecified-high`
  Start from clean state. Run the replay mode pipeline end-to-end. Verify: screenshots are processed, VLM returns scene descriptions, RAG retrieves content, voice session accepts context injection. Run for 5+ minutes. Capture evidence.
  Output: `Scenarios [N/N pass] | Stability [N min] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual code. Verify 1:1 — everything in spec was built, nothing beyond spec was built. Check "Must NOT do" compliance (no YOLO, no mobile, no pyremoteplay). Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Scope [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

- **Wave 1 complete**: `feat(scaffold): project setup with dependencies and test data`
- **Wave 2 complete**: `feat(spikes): validation spikes for capture, VLM, voice, and RAG`
- **Wave 3 complete**: `feat(integration): context manager, system prompt, main pipeline, replay mode`
- **Wave 4 complete**: `feat(demo): end-to-end demo with cost tracking and session rotation`

---

## Success Criteria

### Verification Commands
```bash
# Capture test
python -m src.capture.capture --window 'PS Remote Play' --count 10 --output /tmp/captures/
# Expected: 10 PNG files, each > 50KB

# VLM test
python -m src.vlm.analyze --image data/screenshots/tunic_boss.png
# Expected: JSON with scene description mentioning combat/boss/enemy

# RAG test
python -m src.rag.query 'how do I open the sealed door in the eastern vault'
# Expected: Relevant Tunic wiki content about Eastern Vault

# Voice test (manual verification via replay mode)
python -m src.main --replay --screenshot-dir data/screenshots/ --duration 60
# Expected: Voice session runs for 60 seconds, processes screenshots, responds to audio

# Full pipeline stability
python -m src.main --replay --screenshot-dir data/screenshots/ --duration 600
# Expected: Runs 10 minutes, no crashes, logs show VLM + context injection activity
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] Replay mode works without PS5
- [ ] VLM accuracy ≥80% on Tunic screenshots
- [ ] Voice responds with game context awareness
- [ ] Pipeline stable for 10+ minutes