# Memory Integration

This document describes the conversation memory system that captures and retrieves gameplay context across sessions.

## Overview

The memory integration adds a persistent layer on top of the existing conversation flow. It periodically summarizes conversation exchanges into compact markdown documents, stores them in QMD for retrieval, and surfaces relevant memories during future interactions.

## Architecture

### Components

1. **MemoryDocument** (`src/memory/document.py`)
   - Data model for memory summaries
   - Serializes to/from markdown format
   - Captures: summary, goals, discoveries, locations, items, open questions

2. **ConversationMemoryManager** (`src/memory/manager.py`)
   - Tracks conversation turns in a rolling buffer
   - Triggers summarization after N turns (default: 10)
   - Uses GPT-4o-mini to extract structured information
   - Saves memory documents to disk as markdown

3. **MemoryRetriever** (`src/memory/retriever.py`)
   - Queries memory documents from QMD
   - Returns results with lower confidence than static knowledge
   - Collection naming: `{game_id}-memory`

4. **KnowledgeOrchestrator** (updated)
   - Registers memory retriever separately
   - Queries memory after static knowledge
   - Merges and ranks all results by confidence

### Data Flow

```
User Input → Voice Session → Add Turn → Memory Manager
                                              ↓
                                    Check if N turns reached
                                              ↓
                                    Summarize with LLM
                                              ↓
                                    Save markdown to disk
                                              ↓
                                    Index in QMD (manual step)
                                              ↓
Next Query → Orchestrator → Memory Retriever → QMD
                                              ↓
                                    Merge with KB results
                                              ↓
                                    Return to Voice Session
```

## Memory Document Format

```markdown
# Memory: tunic - Session abc123

**Timestamp:** 2024-01-15 14:30:00
**Turn Count:** 10

## Summary

Player explored the Overworld forest area and discovered a hidden path behind the waterfall. They defeated several enemies and collected health potions.

## Current Goals

- Find the legendary sword
- Defeat the forest boss
- Unlock the eastern gate

## Discoveries

- Secret path behind waterfall leads to cave
- Red gems restore health
- Shield blocks projectile attacks

## Locations Visited

- Overworld Forest
- Hidden Cave
- Waterfall Area

## Items Found

- Health Potion x3
- Shield
- Red Gem

## Open Questions

- What is the glowing door in the cave?
- How to cross the lava pit?

**Source Turns:** 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
```

## Configuration

### Memory Manager Settings

- `turns_per_summary`: Number of turns before creating a summary (default: 10)
- `max_history`: Maximum turns to keep in memory (default: 50)
- `memory_dir`: Directory for markdown files (default: `var/memory`)

### Voice Session Integration

The `NonRealtimeVoiceSession` automatically:
- Tracks user prompts and assistant responses
- Tracks event-triggered messages separately
- Creates summaries when threshold is reached
- Flushes pending summaries on shutdown

## QMD Storage

### Collection Naming

Memory documents are stored in separate collections:
- Static knowledge: `{game_id}` (e.g., `tunic`)
- Memory: `{game_id}-memory` (e.g., `tunic-memory`)

### Indexing Memory Documents

After memory documents are created, they must be indexed in QMD:

```bash
# Index all memory documents for a game
qmd --index game-companion add -c tunic-memory var/memory/memory_tunic_*.md

# Or use the HTTP API
curl -X POST http://localhost:18788/index \
  -H "Content-Type: application/json" \
  -d '{
    "collection": "tunic-memory",
    "files": ["var/memory/memory_tunic_20240115_143000.md"]
  }'
```

## Retrieval Strategy

### Confidence Scoring

- Static knowledge: Original QMD score (0.0-1.0)
- Memory: QMD score × 0.8 (slightly lower priority)

This ensures that authoritative game knowledge ranks higher than remembered context, while still allowing relevant memories to surface.

### Result Limits

- Memory retriever returns top 2 results
- Combined with other retrievers in orchestrator
- Final results sorted by confidence descending

## Usage Example

```python
from pathlib import Path
from src.memory.manager import ConversationMemoryManager

# Create manager
manager = ConversationMemoryManager(
    game_id="tunic",
    memory_dir=Path("var/memory"),
    turns_per_summary=10,
)

# Add conversation turns
manager.add_turn(
    user_input="Where am I?",
    assistant_response="You're in the Overworld forest.",
    scene_context={"location": "Forest"},
)

# Check if summary needed
if manager.should_summarize():
    memory_doc = manager.create_summary()
    print(f"Created memory: {memory_doc.summary}")

# Flush on shutdown
manager.flush()
```

## Testing

Run basic memory tests:

```bash
uv run python test_memory_basic.py
```

## Future Enhancements

1. **Automatic QMD Indexing**: Index memory documents immediately after creation
2. **Memory Pruning**: Remove old or low-value memories to prevent bloat
3. **Cross-Session Continuity**: Link memories across sessions for the same player
4. **Memory Importance Scoring**: Weight memories by relevance and recency
5. **Event-Triggered Summaries**: Create summaries for significant gameplay events
6. **Memory Consolidation**: Merge related memories to reduce duplication

## Validation Criteria (from Plan)

- ✅ Memory documents are human-readable and compact
- ✅ Summaries preserve useful progress/state without storing unnecessary transcript detail
- ✅ QMD retrieval can surface relevant memory when the user asks follow-up questions
- ✅ Memory and static knowledge remain clearly separated in storage and retrieval behavior
- ✅ The design supports expansion to other game verticals without cross-game contamination

## Files Modified

- `src/memory/__init__.py` - Memory module exports
- `src/memory/document.py` - Memory document model
- `src/memory/manager.py` - Conversation memory manager
- `src/memory/retriever.py` - Memory retriever for QMD
- `src/voice/non_realtime.py` - Voice session integration
- `src/rag/orchestrator.py` - Memory retriever registration
- `src/main.py` - Orchestrator setup and shutdown handling

## Notes

- Memory summarization requires OpenAI API key
- Summaries are created using `gpt-4o-mini` for cost efficiency
- Memory documents are stored as markdown for inspectability
- The system gracefully handles missing API keys (no summarization)
- Memory retrieval failures are logged but don't block other retrievers
