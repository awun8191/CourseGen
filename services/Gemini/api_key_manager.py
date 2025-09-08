import datetime
import time
from typing import List
from pathlib import Path

from COURSEGEN.utils.Caching.cache import Cache
from .rate_limit_data import RATE_LIMITS
from .gemini_api_keys import GeminiApiKeys


class ApiKeyManager:
    """Manages a pool of API keys, rotating them as needed."""

    def __init__(
        self, api_keys: List[str] = None, cache_file: str = None
    ):
        # Use gemini_api_keys.py if no keys provided
        if api_keys is None:
            gemini_keys = GeminiApiKeys()
            api_keys = gemini_keys.get_keys()
        
        # Set cache file to data/gemini_cache directory
        if cache_file is None:
            cache_dir = Path(__file__).parent.parent.parent.parent / "data" / "gemini_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = str(cache_dir / "api_key_cache.json")
            
        self.api_keys = api_keys
        self.cache = Cache(cache_file)
        self.cache_data = self._load_cache()
        self.current_key_index = self.cache_data.get("current_key_index", 0)
        # Track RPM timestamps PER KEY PER MODEL to avoid cross-contamination of limits
        self.rpm_timestamps: list[dict[str, list[float]]] = [
            {m: [] for m in RATE_LIMITS.keys()} for _ in api_keys
        ]

    def _load_cache(self) -> dict:
        """Load cache data and reset if it's a new day."""
        cache_data = self.cache.read_cache()
        today = datetime.date.today().isoformat()

        if cache_data.get("date") != today:
            cache_data = {
                "date": today,
                "current_key_index": 0,
                "keys": {
                    key: {
                        "rpd": 0,
                        "total_tokens": 0,
                        "models": {
                            "flash": {"rpd": 0, "total_tokens": 0},
                            "lite": {"rpd": 0, "total_tokens": 0},
                            "pro": {"rpd": 0, "total_tokens": 0},
                            "embedding": {"rpd": 0, "total_tokens": 0}
                        },
                    }
                    for key in self.api_keys
                },
            }
            self.cache.write_cache(cache_data)

        return cache_data

    def get_key(self, model: str = "flash") -> str:
        """Get the current API key."""
        if not self.api_keys:
            raise ValueError("No API keys configured.")

        if self.current_key_index >= len(self.api_keys):
            raise ValueError("All API keys have been used.")

        key = self.api_keys[self.current_key_index]
        if not self.is_key_available(key, model):
            return self.rotate_key(model)

        return key

    def is_key_available(self, key: str, model: str = "flash") -> bool:
        """Check if a key is within its usage limits (per model)."""
        key_data = self.cache_data["keys"][key]
        model_data = key_data["models"][model]
        rate_limit = RATE_LIMITS[model]

        # Check per-day limit for the specific model only
        if model_data["rpd"] >= rate_limit.per_day:
            return False

        # Check RPM per model (clean up old timestamps)
        now = time.time()
        current_model_ts = self.rpm_timestamps[self.current_key_index][model]
        self.rpm_timestamps[self.current_key_index][model] = [
            t for t in current_model_ts if now - t < 60
        ]
        if len(self.rpm_timestamps[self.current_key_index][model]) >= rate_limit.per_minute:
            return False

        # Check token usage per model (approximation)
        if model_data["total_tokens"] >= 2_000_000:
            return False

        return True

    def update_usage(self, key: str, model: str, tokens: int):
        """Update usage data for a key and model."""
        key_data = self.cache_data["keys"][key]
        model_data = key_data["models"][model]

        # Keep aggregate counters for observability (not used for gating)
        key_data["rpd"] += 1
        key_data["total_tokens"] += tokens
        # Per-model counters used for gating
        model_data["rpd"] += 1
        model_data["total_tokens"] += tokens

        # Track RPM per model
        self.rpm_timestamps[self.current_key_index][model].append(time.time())
        self.cache.write_cache(self.cache_data)

    def rotate_key(self, model: str = "flash") -> str:
        """Rotate to the next available API key."""
        start_index = self.current_key_index
        while True:
            self.current_key_index = (
                (self.current_key_index + 1) % len(self.api_keys)
            )
            if self.current_key_index == start_index:
                raise ValueError("All API keys are over their limits.")

            key = self.api_keys[self.current_key_index]
            if self.is_key_available(key, model):
                self.cache_data["current_key_index"] = self.current_key_index
                self.cache.write_cache(self.cache_data)
                return key
