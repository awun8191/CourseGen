"""Gemini API key management - env-only configuration."""
import os
import logging

logger = logging.getLogger(__name__)


def get_gemini_api_keys() -> list[str]:
    """Load Gemini API keys from GEMINI_API_KEYS env var.
    
    Keys should be comma-separated in the environment variable.
    Returns a list of key strings.
    """
    keys_str = os.environ.get("GEMINI_API_KEYS", "")
    if not keys_str:
        logger.warning("GEMINI_API_KEYS environment variable is not set")
        return []
    
    keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    logger.info(f"Loaded {len(keys)} Gemini API key(s) from environment")
    return keys
