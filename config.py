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
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def get_data_dir() -> str:
    """Get the data directory path from environment."""
    return os.environ.get("DATA_DIR", "data")


def get_vector_database() -> str:
    """Get ChromaDB vector database path from environment."""
    return os.environ.get("CHROMA_PERSIST_DIR", "output_data/vector_database")


def get_ocr_cache_dir() -> str:
    """Get OCR cache directory from environment."""
    return os.environ.get("OCR_CACHE_DIR", "data/ocr_cache")


def get_export_dir() -> str:
    """Get export directory from environment."""
    return os.environ.get("EXPORT_DIR", "data/exported_data")


def is_billing_enabled() -> bool:
    """Check if billing is enabled."""
    return os.environ.get("BILLING_ENABLED", "false").lower() == "true"
