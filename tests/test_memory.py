"""Tests for memory subsystem."""

import tempfile
from pathlib import Path

import pytest

from src.memory.document import MemoryDocument
from src.memory.manager import ConversationMemoryManager


def test_memory_document_to_markdown():
    """Test memory document serialization to markdown."""
    doc = MemoryDocument(
        game_id="tunic",
        session_id="test123",
        summary="Player explored the forest area.",
        goals=["Find the sword", "Defeat the boss"],
        discoveries=["Secret path behind waterfall"],
        locations_visited=["Forest", "Cave"],
        items_found=["Health potion", "Key"],
        open_questions=["What is the glowing door?"],
        turn_count=5,
        source_turns=[1, 2, 3, 4, 5],
    )
    
    markdown = doc.to_markdown()
    
    assert "# Memory: tunic - Session test123" in markdown
    assert "Player explored the forest area." in markdown
    assert "Find the sword" in markdown
    assert "Secret path behind waterfall" in markdown
    assert "Forest" in markdown
    assert "Health potion" in markdown
    assert "What is the glowing door?" in markdown


def test_memory_document_from_markdown():
    """Test memory document deserialization from markdown."""
    markdown = """# Memory: tunic - Session test123

**Timestamp:** 2024-01-01 12:00:00
**Turn Count:** 5

## Summary

Player explored the forest area.

## Current Goals

- Find the sword
- Defeat the boss

## Discoveries

- Secret path behind waterfall

## Locations Visited

- Forest
- Cave

## Items Found

- Health potion
- Key

## Open Questions

- What is the glowing door?

**Source Turns:** 1, 2, 3, 4, 5
"""
    
    doc = MemoryDocument.from_markdown(markdown, "tunic", "test123")
    
    assert doc is not None
    assert doc.game_id == "tunic"
    assert doc.session_id == "test123"
    assert "Player explored the forest area." in doc.summary
    assert "Find the sword" in doc.goals
    assert "Secret path behind waterfall" in doc.discoveries
    assert "Forest" in doc.locations_visited
    assert "Health potion" in doc.items_found
    assert "What is the glowing door?" in doc.open_questions
    assert doc.turn_count == 5
    assert doc.source_turns == [1, 2, 3, 4, 5]


def test_conversation_memory_manager_add_turn():
    """Test adding conversation turns to memory manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConversationMemoryManager(
            game_id="tunic",
            memory_dir=Path(tmpdir),
            turns_per_summary=5,
            api_key=None,
        )
        
        turn_id = manager.add_turn(
            user_input="Where am I?",
            assistant_response="You're in the forest.",
            scene_context={"location": "Forest"},
            is_event_triggered=False,
        )
        
        assert turn_id == 1
        assert manager.turn_count == 1
        assert manager.pending_turns == 1
        assert not manager.should_summarize()
        
        for i in range(4):
            manager.add_turn(
                user_input=f"Question {i}",
                assistant_response=f"Answer {i}",
            )
        
        assert manager.turn_count == 5
        assert manager.pending_turns == 5
        assert manager.should_summarize()


def test_conversation_memory_manager_flush():
    """Test flushing memory on shutdown."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConversationMemoryManager(
            game_id="tunic",
            memory_dir=Path(tmpdir),
            turns_per_summary=10,
            api_key=None,
        )
        
        manager.add_turn(
            user_input="Test question",
            assistant_response="Test answer",
        )
        
        assert manager.pending_turns == 1
        
        result = manager.flush()
        
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
