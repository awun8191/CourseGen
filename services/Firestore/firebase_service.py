"""Firebase/Firestore service - env-only configuration."""
import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_firebase_credentials() -> dict:
    """Load Firebase service account credentials from environment.
    
    Expects FIREBASE_CREDENTIALS_JSON env var containing the full
    service account JSON string, or FIREBASE_CREDENTIALS_PATH pointing
    to a JSON file.
    """
    creds_json = os.environ.get("FIREBASE_CREDENTIALS_JSON")
    creds_path = os.environ.get("FIREBASE_CREDENTIALS_PATH")
    
    if creds_json:
        try:
            return json.loads(creds_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid FIREBASE_CREDENTIALS_JSON: {e}")
    
    if creds_path:
        if not os.path.exists(creds_path):
            raise FileNotFoundError(f"Firebase credentials file not found: {creds_path}")
        with open(creds_path) as f:
            return json.load(f)
    
    raise EnvironmentError(
        "Either FIREBASE_CREDENTIALS_JSON or FIREBASE_CREDENTIALS_PATH must be set in environment"
    )


def get_firestore_database() -> Optional[str]:
    """Get the Firestore database ID from environment."""
    return os.environ.get("FIRESTORE_DATABASE")


def get_project_id() -> Optional[str]:
    """Get the GCP project ID from environment."""
    return os.environ.get("GCP_PROJECT_ID")
