import datetime
import json
import time
from pathlib import Path
from typing import List
import os
from zoneinfo import ZoneInfo

try:  # pragma: no cover - optional dependency
    from COURSEGEN.utils.Caching.cache import Cache  # type: ignore
except Exception:  # pragma: no cover - fallback for tests
    try:
        from utils.Caching.cache import Cache  # type: ignore
    except Exception:  # pragma: no cover - lightweight fallback
        class Cache:  # type: ignore[override]
            def __init__(self, cache_file: str) -> None:
                self.path = Path(cache_file)

            def read_cache(self) -> dict:
                try:
                    return json.loads(self.path.read_text(encoding="utf-8"))
                except FileNotFoundError:
                    return {}
                except Exception:
                    return {}

            def write_cache(self, data: dict) -> None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
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
        
        # Set cache file to OUTPUT_DATA2/data/gemini_cache directory (persistent volume)
        if cache_file is None:
            default_root = Path(__file__).resolve().parents[2] / "OUTPUT_DATA2"
            cache_root = Path(os.environ.get("COURSEGEN_CACHE_ROOT", str(default_root)))
            cache_dir = cache_root / "data" / "gemini_cache"
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
        pacific_today = datetime.datetime.now(ZoneInfo("America/Los_Angeles")).date()
        today = pacific_today.isoformat()

        # Check for environment variable to disable daily reset
        disable_daily_reset = os.environ.get("COURSEGEN_DISABLE_CACHE_DAILY_RESET", "false").lower() == "true"

        if cache_data.get("date") != today and not disable_daily_reset:
            cache_data = {
                "date": today,
                "current_key_index": 0,
                "keys": {
                    key: {
                        "rpd": 0,
                        "total_tokens": 0,
                        "exhausted": False,
                        "exhausted_reason": "",
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

        # Ensure all keys have exhaustion markers when loading an existing cache file
        for key in self.api_keys:
            key_data = cache_data.setdefault("keys", {}).setdefault(key, {})
            key_data.setdefault("rpd", 0)
            key_data.setdefault("total_tokens", 0)
            key_data.setdefault("exhausted", False)
            key_data.setdefault("exhausted_reason", "")
            key_data.setdefault(
                "models",
                {
                    "flash": {"rpd": 0, "total_tokens": 0},
                    "lite": {"rpd": 0, "total_tokens": 0},
                    "pro": {"rpd": 0, "total_tokens": 0},
                    "embedding": {"rpd": 0, "total_tokens": 0},
                },
            )
        return cache_data

    def get_key(self, model: str = "flash") -> str:
        """Get the current API key."""
        if not self.api_keys:
            raise ValueError("No API keys configured.")

        if self.current_key_index >= len(self.api_keys):
            raise ValueError("All API keys have been used.")

        key = self.api_keys[self.current_key_index]
        if not self.is_key_available(key, model):
            # Try rotating to next key for the same model
            try:
                return self.rotate_key(model)
            except ValueError:
                # If all keys are exhausted for this model, try other models
                for alt_model in ["lite", "flash", "pro"]:
                    if alt_model != model:
                        try:
                            return self.rotate_key(alt_model)
                        except ValueError:
                            continue
                raise ValueError(f"All API keys exhausted for all models including {model}")

        return key

    def is_key_available(self, key: str, model: str = "flash") -> bool:
        """Check if a key is within its usage limits (per model)."""
        key_data = self.cache_data["keys"][key]
        if key_data.get("exhausted"):
            return False
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
        """Rotate to the next available API key for the specified model."""
        start_index = self.current_key_index
        max_attempts = len(self.api_keys) * 2  # Allow trying each key twice
        attempts = 0

        while attempts < max_attempts:
            self.current_key_index = (
                (self.current_key_index + 1) % len(self.api_keys)
            )
            attempts += 1

            if self.current_key_index == start_index and attempts >= len(self.api_keys):
                raise ValueError(f"All API keys are over their limits for model {model}.")

            key = self.api_keys[self.current_key_index]
            if self.is_key_available(key, model):
                self.cache_data["current_key_index"] = self.current_key_index
                self.cache.write_cache(self.cache_data)
                return key

        raise ValueError(f"All API keys are over their limits for model {model}.")

    def mark_key_exhausted(self, key: str, model: str, reason: str = "") -> None:
        if key not in self.cache_data.get("keys", {}):
            return
        key_data = self.cache_data["keys"][key]
        key_data["exhausted"] = True
        key_data["exhausted_reason"] = reason
        # Clear per-model rpms to avoid stale entries when reset occurs next day
        if model in key_data.get("models", {}):
            key_data["models"][model]["rpd"] = RATE_LIMITS[model].per_day
        self.cache.write_cache(self.cache_data)

    def all_keys_exhausted(self, model: str = "flash") -> bool:
        """Check if all keys are exhausted for the given model."""
        for idx, key in enumerate(self.api_keys):
            key_data = self.cache_data.get("keys", {}).get(key)
            if not key_data or key_data.get("exhausted"):
                continue
            # Temporarily reuse availability check without altering state
            previous_index = self.current_key_index
            self.current_key_index = idx
            available = self.is_key_available(key, model)
            self.current_key_index = previous_index
            if available:
                return False
        return True

    def force_terminate_if_all_exhausted(self, model: str = "flash") -> None:
        """Force termination if all keys are exhausted. Use when API errors indicate exhaustion."""
        if self.all_keys_exhausted(model):
            # Mark all remaining keys as exhausted to prevent further attempts
            for key in self.api_keys:
                if key in self.cache_data.get("keys", {}):
                    key_data = self.cache_data["keys"][key]
                    key_data["exhausted"] = True
                    key_data["exhausted_reason"] = "All keys exhausted - forced termination"
            self.cache.write_cache(self.cache_data)

            # Send email notification if email service is configured
            try:
                from services.Email.email_service import send_termination_notification
                send_termination_notification(len(exhausted_keys), len(self.api_keys), model, questions_generated=0)
            except Exception as notification_error:
                # Don't fail termination if notification fails
                pass
    
            # Raise a clear termination error
            exhausted_keys = [k for k in self.api_keys if self.cache_data.get("keys", {}).get(k, {}).get("exhausted")]
            raise RuntimeError(
                f"🚨 ALL API KEYS EXHAUSTED - TERMINATING OPERATIONS 🚨\n"
                f"Exhausted keys: {len(exhausted_keys)}/{len(self.api_keys)}\n"
                f"Model: {model}\n"
                f"Check your API quotas and add more keys to services/Gemini/gemini_api_keys.py"
            )
