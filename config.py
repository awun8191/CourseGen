"""Application configuration - env-only."""
import os
import json
import logging

logger = logging.getLogger(__name__)


def get_config(key: str, default=None):
    """Get a configuration value from environment variables.
    
    Supports simple string values and JSON-encoded complex values.
    """
    value = os.environ.get(key)
    if value is None:
        return default
    # Try parsing as JSON for complex types (lists, dicts, etc.)
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


# Common configuration accessors
def get_data_dir() -> str:
    """Get the data directory path from environment."""
    return os.environ.get("DATA_DIR", "data")


def get_chromadb_storage() -> str:
    """Get ChromaDB storage path from environment."""
    return os.environ.get("CHROMADB_STORAGE", "chromadb_storage")


def get_ocr_cache_dir() -> str:
    """Get OCR cache directory from environment."""
    return os.environ.get("OCR_CACHE_DIR", "data/ocr_cache")


def get_export_dir() -> str:
    """Get export directory from environment."""
    return os.environ.get("EXPORT_DIR", "data/exported_data")


def is_billing_enabled() -> bool:
    """Check if billing is enabled."""
    return os.environ.get("BILLING_ENABLED", "false").lower() == "true"
