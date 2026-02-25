"""VLM Scene Analysis module for Tunic game screenshots using Google Gemini."""

import json
import os
import re
import time
import argparse
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types


ANALYSIS_PROMPT = """\
You are analyzing a screenshot from the video game TUNIC.

TUNIC is an isometric action-adventure game where the player controls a small fox protagonist \
exploring a mysterious world filled with ruins, forests, caverns, and dangerous enemies. \
The game features souls-like combat, environmental puzzles, and a unique in-game manual \
written in a fictional language. Text visible in screenshots may be in this fictional \
language and is intentionally unreadable — that is normal.

Analyze this screenshot and return a JSON object with exactly these fields:

- "location": Where in the game world this appears to be (e.g., "Overworld", "East Forest", "West Garden", "Ruins", "Cavern", "Boss Arena", "Menu", "Shop")
- "activity": What the player is doing (e.g., "exploring", "fighting", "solving puzzle", "in menu", "viewing map", "in dialogue", "shopping", "died")
- "enemies": A list of visible enemies, or "none" if no enemies are visible
- "health_status": Player health if visible (e.g., "full", "low", "critical", "not visible")
- "ui_elements": List of visible UI elements (e.g., "health bar", "stamina bar", "inventory", "map", "manual page", "dialogue box")
- "notable_items": Any notable items, NPCs, or interactive elements visible
- "description": A 1-2 sentence natural language description of the scene

Respond ONLY with valid JSON. No markdown fences, no explanation, no extra text.\
"""


class SceneAnalyzer:
    """Analyzes Tunic game screenshots using Google Gemini VLM.
    
    Supported models:
    - gemini-2.5-flash (default) — best quality, $0.30/$2.50 per 1M tokens
    - gemini-2.5-flash-lite — budget option, $0.10/$0.40 per 1M tokens
    """

    SUPPORTED_MODELS = ("gemini-2.5-flash", "gemini-2.5-flash-lite")

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash-lite"):
        """Initialize with Gemini API key (from env if not provided).
        
        Args:
            api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
            model: Gemini model name. Supports 'gemini-2.5-flash' (default) 
                   and 'gemini-2.5-flash-lite' (budget).
        """
        load_dotenv()
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it in .env or pass api_key parameter."
            )
        self._client = genai.Client(api_key=self._api_key)
        self._model = model

    def analyze_screenshot(self, image_data: bytes, mime_type: str = "image/png") -> dict:
        """Analyze a game screenshot, return structured scene description.

        Args:
            image_data: Raw image bytes.
            mime_type: MIME type of the image (default: image/png).

        Returns:
            Dict with structured scene analysis fields.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_bytes(data=image_data, mime_type=mime_type),
                                types.Part.from_text(text=ANALYSIS_PROMPT),
                            ],
                        )
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )
                return self._parse_response(response.text)

            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "429" in error_str or "quota" in error_str:
                    if attempt < max_retries - 1:
                        wait = 2 ** (attempt + 1)
                        print(f"  Rate limited, retrying in {wait}s...")
                        time.sleep(wait)
                        continue
                if "invalid" in error_str and "image" in error_str:
                    return {"error": f"Invalid image: {e}"}
                raise

        return {"error": "Max retries exceeded due to rate limiting"}

    def analyze_file(self, image_path: str) -> dict:
        """Analyze a screenshot from file path.

        Args:
            image_path: Path to the image file.

        Returns:
            Dict with structured scene analysis fields.
        """
        path = Path(image_path)
        if not path.exists():
            return {"error": f"File not found: {image_path}"}

        suffix = path.suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
        mime_type = mime_map.get(suffix, "image/png")

        image_data = path.read_bytes()
        return self.analyze_screenshot(image_data, mime_type=mime_type)

    def _parse_response(self, text: str) -> dict:
        """Parse JSON from model response, handling markdown fences."""
        cleaned = text.strip()
        # Strip markdown code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Last resort: find first { ... } block
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {
                "error": "Failed to parse JSON response",
                "raw_response": text[:500],
            }


def run_single(image_path: str) -> None:
    """Analyze a single image and print results."""
    analyzer = SceneAnalyzer()
    result = analyzer.analyze_file(image_path)
    print(json.dumps(result, indent=2))


def run_batch(manifest_path: str, output_path: str) -> None:
    """Analyze all images in a manifest file and save results."""
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        print(f"Error: Manifest not found: {manifest_path}")
        return

    manifest = json.loads(manifest_file.read_text())
    screenshots_dir = manifest_file.parent
    analyzer = SceneAnalyzer()

    results = []
    total = len(manifest)

    for i, entry in enumerate(manifest, 1):
        filename = entry["filename"]
        image_path = screenshots_dir / filename
        print(f"[{i}/{total}] Analyzing {filename}...")

        if not image_path.exists():
            print(f"  WARNING: File not found, skipping: {image_path}")
            results.append({
                "filename": filename,
                "manifest": entry,
                "analysis": {"error": f"File not found: {image_path}"},
            })
            continue

        analysis = analyzer.analyze_file(str(image_path))
        results.append({
            "filename": filename,
            "manifest": entry,
            "analysis": analysis,
        })

        if "error" not in analysis:
            print(f"  -> location={analysis.get('location')}, activity={analysis.get('activity')}")
        else:
            print(f"  -> ERROR: {analysis.get('error')}")

        # Small delay to avoid rate limiting
        if i < total:
            time.sleep(0.5)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")
    print(f"Analyzed {len(results)} screenshots")


def score_results(results: list[dict]) -> dict:
    """Score VLM results against manifest ground truth.

    Returns accuracy report dict.
    """
    # Mapping from manifest game_state to acceptable VLM activity/location keywords
    state_keywords = {
        "overworld_exploration": {
            "activity": ["exploring", "exploration", "traversing", "walking", "running", "moving"],
            "location": ["forest", "ruins", "cavern", "biome", "overworld", "garden", "beach",
                         "swamp", "mountain", "cliff", "bridge", "path", "field", "shore",
                         "graveyard", "cemetery", "town", "village", "waterfall"],
        },
        "boss_fight": {
            "activity": ["fighting", "combat", "battling", "attacking", "dodging", "boss"],
            "location": ["boss", "arena", "chamber", "lair"],
        },
        "puzzle_area": {
            "activity": ["puzzle", "solving", "interacting", "activating", "exploring"],
            "location": ["puzzle", "chamber", "temple", "shrine"],
        },
        "npc_dialogue": {
            "activity": ["dialogue", "talking", "conversation", "interacting", "shopping", "trading"],
            "location": ["village", "shop", "npc", "town", "merchant"],
        },
        "inventory_manual": {
            "activity": ["menu", "inventory", "manual", "reading", "viewing", "browsing"],
            "location": ["menu", "manual", "inventory", "page"],
        },
        "map_screen": {
            "activity": ["map", "viewing", "navigating", "menu"],
            "location": ["map", "world map", "navigation"],
        },
        "death_loading": {
            "activity": ["died", "death", "loading", "game over", "respawn"],
            "location": ["game over", "death", "loading", "respawn"],
        },
    }

    correct = 0
    total = 0
    details = []

    for result in results:
        analysis = result.get("analysis", {})
        manifest = result.get("manifest", {})
        game_state = manifest.get("game_state", "")

        if "error" in analysis:
            details.append({
                "filename": result["filename"],
                "expected_state": game_state,
                "match": False,
                "reason": f"Analysis error: {analysis['error']}",
            })
            total += 1
            continue

        vlm_activity = (analysis.get("activity") or "").lower()
        vlm_location = (analysis.get("location") or "").lower()
        vlm_description = (analysis.get("description") or "").lower()

        keywords = state_keywords.get(game_state, {})
        activity_kws = keywords.get("activity", [])
        location_kws = keywords.get("location", [])

        # Check if any keyword matches in activity, location, or description
        activity_match = any(kw in vlm_activity for kw in activity_kws)
        location_match = any(kw in vlm_location for kw in location_kws)
        description_match = any(
            kw in vlm_description for kw in activity_kws + location_kws
        )

        matched = activity_match or location_match or description_match
        if matched:
            correct += 1

        total += 1
        details.append({
            "filename": result["filename"],
            "expected_state": game_state,
            "vlm_activity": vlm_activity,
            "vlm_location": vlm_location,
            "match": matched,
            "reason": "activity" if activity_match else ("location" if location_match else ("description" if description_match else "no match")),
        })

    accuracy = (correct / total * 100) if total > 0 else 0.0

    return {
        "total_screenshots": total,
        "correct_matches": correct,
        "accuracy_percent": round(accuracy, 1),
        "details": details,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tunic screenshot scene analysis via Gemini VLM")
    parser.add_argument("--image", type=str, help="Path to a single image to analyze")
    parser.add_argument("--batch", type=str, help="Path to manifest.json for batch analysis")
    parser.add_argument("--output", type=str, default="results.json", help="Output path for batch results")
    args = parser.parse_args()

    if args.image:
        run_single(args.image)
    elif args.batch:
        run_batch(args.batch, args.output)
    else:
        parser.print_help()
