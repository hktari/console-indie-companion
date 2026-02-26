"""Context Synthesizer -- uses a fast LLM to convert raw frame data into a cohesive narrative."""

import logging
import os
import json
from typing import Any, Optional

import openai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


SYNTHESIS_PROMPT_TEMPLATE = """
As the internal monologue of a helpful AI game companion, your task is to synthesize a brief, cohesive narrative of the player's recent actions and current situation. 

Use the following data:
1. A list of recent scene descriptions, in chronological order.
2. Relevant information from the game's knowledge base.

Combine these sources into a 2-3 sentence story that flows naturally. Focus on the player's activity, their immediate environment, and their current status (like health or items). Weave in the knowledge base context where it's relevant.

**Recent Scenes:**
{scenes_json}

**Knowledge Base Context:**
{rag_context}

**Narrative:**
"""

class ContextSynthesizer:
    """Calls a fast LLM to synthesize a narrative from recent game context."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None) -> None:
        load_dotenv()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required for ContextSynthesizer.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        logger.info(f"ContextSynthesizer initialised with model: {model}")

    def synthesize(self, scenes: list[dict[str, Any]], rag_context: str) -> str:
        """Generates a narrative summary of the provided scenes and RAG context."""
        if not scenes:
            return ""

        try:
            # Sanitize scenes for JSON serialization
            scenes_json = json.dumps(scenes, indent=2)

            prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
                scenes_json=scenes_json,
                rag_context=rag_context or "No additional information."
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150,
            )
            
            narrative = ""
            if response.choices and response.choices[0].message.content:
                narrative = response.choices[0].message.content.strip()
            logger.debug(f"Synthesized narrative: {narrative}")
            return narrative

        except Exception:
            logger.exception("Error during context synthesis.")
            return ""
