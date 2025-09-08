import os
import json
from dataclasses import dataclass, field

CONFIG_FILE = os.getenv('CONFIG_FILE', 'config.json')

@dataclass
class Config:
    """Holds configurable paths and credentials for the project."""

    firestore_service_account: str = os.getenv('FIRESTORE_SERVICE_ACCOUNT', r'/home/raregazetto/Downloads/engineering_hub_service_account.json')
    pdf_directory: str = os.getenv('PDF_DIRECTORY', '')
    processed_courses_cache: str = os.getenv('PROCESSED_COURSES_CACHE', '')
    courses_json_path: str = os.getenv('COURSES_JSON_PATH', '')
    transfer_service_account_path: str = os.getenv('TRANSFER_SERVICE_ACCOUNT', '')
    reciving_service_account_path: str = os.getenv('RECEIVING_SERVICE_ACCOUNT', '')
    sample_pdf_path: str = os.getenv('SAMPLE_PDF_PATH', '')
    document_source: str = os.getenv('DOCUMENT_SOURCE', '')
    test_directory: str = os.getenv('TEST_DIRECTORY', '')
    gemini_api_keys: list[str] = field(default_factory=lambda: os.getenv('GEMINI_API_KEYS', '').split(','))
    chroma_persist_dir: str = os.getenv('CHROMA_PERSIST_DIR', 'chromadb_storage')
    collections_json_path: str = os.getenv('COLLECTIONS_JSON_PATH', 'collections.json')


def load_config() -> Config:
    """Load configuration from environment variables and ``config.json``."""

    config = Config()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                data = json.load(f)
            for field in config.__dataclass_fields__:
                if getattr(config, field) == '' and field in data:
                    setattr(config, field, data[field])
        except Exception:
            pass
    return config
