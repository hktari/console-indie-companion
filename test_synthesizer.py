"""Standalone test script for the ContextSynthesizer."""

import logging
from dotenv import load_dotenv
from src.context.synthesizer import ContextSynthesizer

# --- Setup ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
load_dotenv()


def test_synthesis_in_isolation():
    """Tests the ContextSynthesizer with mock data."""
    logger = logging.getLogger(__name__)
    logger.info("--- Running ContextSynthesizer Test ---")

    # 1. Mock Data
    mock_scenes = [
        {
            "description": "The player, a small fox, is standing in a grassy area with a large, sealed stone door.",
            "location": "Overworld",
            "activity": "exploring",
            "health_status": "full",
            "enemies": "none",
            "notable_items": ["sealed stone door", "sword"],
        },
        {
            "description": "The player is now fighting two small slime enemies near some trees.",
            "location": "Overworld",
            "activity": "combat",
            "health_status": "medium",
            "enemies": ["slime (x2)"],
            "notable_items": ["trees"],
        },
        {
            "description": "The player has defeated the slimes and is now standing in front of a treasure chest that has appeared.",
            "location": "Overworld",
            "activity": "discovery",
            "health_status": "medium",
            "enemies": "none",
            "notable_items": ["treasure chest"],
        },
    ]

    mock_rag_context = (
        "Sealed stone doors often require finding hidden levers or keys in the nearby area. "
        "Slimes are weak but can be dangerous in groups. They sometimes guard secrets."
    )

    logger.info("Synthesizing narrative from mock data...")

    # 2. Execution
    try:
        synthesizer = ContextSynthesizer(model="gpt-4.1-mini")
        narrative = synthesizer.synthesize(mock_scenes, mock_rag_context)

        # 3. Verification
        print("\n" + "-"*20)
        logger.info("SYNTHESIZED NARRATIVE:")
        if narrative:
            print(narrative)
        else:
            logger.warning("Synthesis returned an empty narrative.")
        print("-"*20 + "\n")

    except Exception:
        logger.error("An error occurred during the synthesizer test.", exc_info=True)


if __name__ == "__main__":
    test_synthesis_in_isolation()
