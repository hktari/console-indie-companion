"""Context Synthesizer -- uses a fast LLM to convert raw frame data into a cohesive narrative."""

import logging
import os
import json
from typing import Any, Optional

import openai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


SYNTHESIS_PROMPT_TEMPLATE = """
Your task is to create a concise summary of the player's recent activity based on the provided data.

- Use the list of recent scene descriptions and knowledge base information.
- Create a 2-3 sentence summary of events.
- Focus on facts: player actions, location, and status.
- Do not use narrative, poetic, or interpretive language.

**Recent Scenes:**
{scenes_json}

**Knowledge Base Context:**
{rag_context}

**Summary:**
"""


class ContextSynthesizer:
    """Calls a fast LLM to synthesize a narrative from recent game context."""

    # TODO: test reasoning and non-reasoning models. Weigh cost -performance tradeoff.
    def __init__(
        self, model: str = "gpt-4.1-mini", api_key: Optional[str] = None
    ) -> None:
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
                rag_context=rag_context or "No additional information.",
            )

            logger.debug(f"Synthesis prompt: {prompt}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
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
