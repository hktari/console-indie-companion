"""Basic test for memory subsystem without pytest."""

import tempfile
from pathlib import Path

from src.memory.document import MemoryDocument
from src.memory.manager import ConversationMemoryManager


def test_memory_document():
    """Test memory document serialization."""
    print("Testing MemoryDocument...")
    
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
    
    parsed = MemoryDocument.from_markdown(markdown, "tunic", "test123")
    assert parsed is not None
    assert parsed.game_id == "tunic"
    assert "Find the sword" in parsed.goals
    
    print("✓ MemoryDocument serialization works")


def test_memory_manager():
    """Test conversation memory manager."""
    print("Testing ConversationMemoryManager...")
    
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
        assert manager.should_summarize()
        
        print("✓ ConversationMemoryManager turn tracking works")


if __name__ == "__main__":
    try:
        test_memory_document()
        test_memory_manager()
        print("\n✅ All memory tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise
