import os
import sys
import json
import urllib.request
import urllib.error
import logging

logger = logging.getLogger(__name__)


async def check_api_quota() -> None:
    """Perform a pre-flight check to ensure the OpenAI API key is valid and has quota."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    logger.info("Performing pre-flight API check...")
    try:
        # We use a simple models list request to verify auth
        # It's lightweight and fails if the key is invalid or quota is exceeded
        req = urllib.request.Request(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                logger.info(
                    "API key is valid. Checking quota by initializing a minimal text completion..."
                )

        # To truly check quota for a model, we try a minimal 1-token completion
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "gpt-4o-mini",  # Use a cheap model for the check
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 1,
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                logger.info("Pre-flight check passed! API quota is available.")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            error_json = json.loads(error_body)
            error_msg = error_json.get("error", {}).get("message", e.reason)
            error_code = error_json.get("error", {}).get("code", "unknown")

            logger.error(f"API Error [{error_code}]: {error_msg}")

            if error_code == "insufficient_quota":
                logger.error("\n❌ ERROR: Insufficient OpenAI API Quota")
                logger.error(
                    "Your API key is valid, but you have run out of credits or hit your billing limit."
                )
                logger.error(
                    "Please check your billing details at: https://platform.openai.com/account/billing"
                )
            elif e.code == 401:
                logger.error("\n❌ ERROR: Invalid OpenAI API Key")
                logger.error("Please ensure your OPENAI_API_KEY is correct.")
            else:
                logger.error(f"\n❌ ERROR: API Check Failed ({e.code})")
                logger.error(f"Details: {error_msg}")
        except Exception:
            logger.error(f"HTTP Error {e.code}: {e.reason}")
            logger.error(f"\n❌ ERROR: Pre-flight check failed (HTTP {e.code})")

        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to perform pre-flight check: {e}")
        sys.exit(1)
