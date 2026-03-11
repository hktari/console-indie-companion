# Agent Framework Implementation Summary

## Overview

Successfully implemented planner-controlled research routing for the non-realtime voice path, removing Exa from always-on retrieval and introducing intelligent routing decisions.

## Implementation Details

### Phase 1: Planner Foundation ✅

#### New Modules Created

**`src/agent/models.py`**
- `RouteType`: Enum for routing decisions (DIRECT_ANSWER, KNOWLEDGE_BASE, MEMORY_SEARCH, WEB_RESEARCH, COMBINED)
- `AgentDecision`: Structured decision with route, reasoning, confidence, and tools to call
- `EvidenceBundle`: Collected evidence with KB results, memory results, research memo, and source provenance

**`src/agent/tools.py`**
- `knowledge_base_search`: LangChain tool wrapper for local game KB (QMD)
- `memory_search`: LangChain tool wrapper for conversation memory
- `web_search`: LangChain tool wrapper for Exa web search

**`src/agent/planner.py`**
- `RequestPlanner`: Lightweight planner using heuristic-based routing
- Routes queries based on keywords and context:
  - Memory queries: "earlier", "before", "remember", "what did i"
  - Web research: "look up", "search for", "find out", "what is"
  - Knowledge base: game-specific terms with scene context
  - Direct answer: simple conversational responses

#### Integration Changes

**`src/main.py`**
- Removed `ExaRetriever()` from default orchestrator for non-realtime mode
- Exa only registered for realtime mode (preserves existing behavior)
- Added logging to distinguish between modes

**`src/voice/non_realtime.py`**
- Added `RequestPlanner` initialization with QMD URL
- Updated `_build_prompt_context()` to use planner decisions
- Replaced unconditional retrieval with planner-controlled evidence gathering
- Updated `_generate_reply()` to consume `EvidenceBundle` instead of raw retrieval results
- Added logging for planner decisions and evidence collection

### Phase 2: Research Subagent ✅

**`src/agent/research.py`**
- `ResearchSubagent`: Isolated research agent for web search
- Searches both KB and web sources
- Synthesizes findings into compact memo (<200 words)
- Uses LangChain `ChatOpenAI` for cleaner message handling
- Returns structured research memo with citations instead of raw search dumps

**Integration with Planner**
- Planner delegates to research subagent when `web_search` tool is called
- Research subagent output stored in `EvidenceBundle.research_memo`
- Memo includes source citations ([KB], [Web]) for provenance

### Dependencies Added

**`pyproject.toml`**
- `langchain-core>=0.3.0`: Core LangChain abstractions
- `langchain-openai>=0.2.0`: OpenAI integration for LangChain
- `langsmith>=0.2.0`: LangSmith tracing support

## Architecture

```
User Request
    ↓
NonRealtimeVoiceSession
    ↓
RequestPlanner.plan()
    ↓
AgentDecision (route + tools)
    ↓
RequestPlanner.gather_evidence()
    ↓
    ├─ knowledge_base_search (if needed)
    ├─ memory_search (if needed)
    └─ ResearchSubagent.research() (if web needed)
           ↓
           ├─ KB search
           ├─ Web search (Exa)
           └─ LLM synthesis → compact memo
    ↓
EvidenceBundle
    ↓
_generate_reply() → TTS
```

## Key Benefits

1. **Exa is no longer always-on**: Only invoked when planner decides web research is needed
2. **Cleaner prompts**: Research subagent synthesizes findings instead of dumping raw search results
3. **Source provenance**: Evidence bundle tracks sources (KB, memory, web)
4. **Debuggable decisions**: Planner logs routing decisions with reasoning and confidence
5. **Memory-aware**: Planner can route to conversation memory for past context
6. **Extensible**: Easy to add new routing heuristics or upgrade to LLM-based classification

## Validation Checklist

### Manual Testing

Run the application in non-realtime mode with various queries:

```bash
# Start in replay mode for testing
uv run src/main.py --replay --screenshot-dir data/screenshots/ --duration 60
```

**Test Cases:**

1. **Simple conversational query** (should NOT trigger Exa)
   - Say: "hello" or "thanks"
   - Expected: Direct answer, no retrieval logged

2. **Memory query** (should trigger memory search only)
   - Say: "what did I do earlier?"
   - Expected: Planner decision = MEMORY_SEARCH, memory_search tool called

3. **Game-specific query** (should trigger KB search only)
   - Say: "what is this item?" (with scene context)
   - Expected: Planner decision = KNOWLEDGE_BASE, knowledge_base_search tool called

4. **General knowledge query** (should trigger web research)
   - Say: "look up the history of indie games"
   - Expected: Planner decision = WEB_RESEARCH, research subagent invoked, compact memo returned

5. **Game-specific with "how to"** (should prefer KB over web)
   - Say: "how do I beat the tunic boss?"
   - Expected: Planner decision = COMBINED, knowledge_base_search called (not web)

### Log Verification

Check logs for:
- `Planner decision: <route> (confidence: X.XX) - <reasoning>`
- `Evidence gathered: X KB results, X memory results, sources: <list>`
- `Delegating to research subagent for web research` (only for web queries)
- `Research memo generated (X chars)` (when web research runs)

### Type Checking

```bash
uv run pyright src/agent/
uv run pyright src/voice/non_realtime.py
uv run pyright src/main.py
```

All checks passed ✅

## Future Enhancements (Phase 3)

1. **LLM-based classification**: Replace heuristic routing with model-based classification
2. **Multi-turn reasoning**: Add checkpointer for complex queries requiring multiple steps
3. **Memory quality filtering**: Tag memory summaries with source provenance, exclude low-confidence web results
4. **Realtime integration**: Reuse planner/research core for realtime tool-calling path
5. **LangSmith tracing**: Enable tracing for debugging planner decisions
6. **Tuning**: Adjust routing heuristics based on logged decision quality

## Notes

- Realtime mode still uses Exa in orchestrator (unchanged behavior)
- Research subagent uses same model as planner (default: gpt-4.1-mini)
- Memory write is unchanged - existing `ConversationMemoryManager` handles turn tracking
- QMD URL is passed through from environment to all components
