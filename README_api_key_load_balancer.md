# Gemini API Key Load Balancer

High-volume Gemini workloads in CourseGen rely on `services/Gemini/api_key_manager.py` to rotate across multiple API keys, enforce per-model quotas, and surface exhaustion state to upstream pipelines. This document explains how the manager works, how to configure it, and how to monitor its state.

## Core Responsibilities
- **Key discovery** – loads keys from `GeminiApiKeys` (list in `services/Gemini/gemini_api_keys.py`) or from explicit arguments when instantiated manually.
- **Persistent usage tracking** – stores daily counters, tokens, and exhaustion flags in `OUTPUT_DATA2/data/gemini_cache/api_key_cache.json` (path adjustable via `COURSEGEN_CACHE_ROOT`).
- **Per-model quotas** – enforces rate limits defined in `services/Gemini/rate_limit_data.py` for `flash`, `lite`, `pro`, and `embedding` model families.
- **RPM throttling** – keeps rolling timestamps per key per model to avoid exceeding per-minute limits.
- **Failure handling** – marks keys exhausted on fatal errors, escalates to email notifications (via `services.Email.email_service`) when every key is exhausted, and raises a terminating `RuntimeError`.

## Cache Layout
```json
{
  "date": "2024-05-17",
  "current_key_index": 2,
  "keys": {
    "AIza...123": {
      "rpd": 87,
      "total_tokens": 312000,
      "exhausted": false,
      "exhausted_reason": "",
      "models": {
        "flash": {"rpd": 40, "total_tokens": 210000},
        "lite": {"rpd": 47, "total_tokens": 102000},
        "pro": {"rpd": 0, "total_tokens": 0},
        "embedding": {"rpd": 0, "total_tokens": 0}
      }
    }
  }
}
```
- Counters reset at midnight **America/Los_Angeles** unless `COURSEGEN_DISABLE_CACHE_DAILY_RESET=true`.
- `exhausted_reason` provides context (quota exceeded, manual override, forced termination, etc.).
- `current_key_index` is used for round-robin rotation across the configured key list.

## Rate Limits (defaults from `rate_limit_data.py`)
| Model family | Requests per minute | Requests per day |
| --- | --- | --- |
| `lite` | 15 | 1,000 |
| `flash` | 10 | 250 |
| `pro` | 5 | 25 |
| `embedding` | 5 | 1,000 |

These gates are conservative starting points—adjust them in `rate_limit_data.py` if your project-specific quotas differ.

## Configuration Steps
1. **List your keys** – edit `services/Gemini/gemini_api_keys.py`:
   ```python
   class GeminiApiKeys:
       def __init__(self, api_keys: list[str] = None):
           self.api_keys = api_keys or [
               "REDACTED_API_KEY...first",
               "REDACTED_API_KEY...second",
               "..."
           ]
   ```
   > Keep production keys out of source control. Consider loading them from environment variables or a secrets manager when running in deployed environments.
2. **Persist cache directory** – ensure `OUTPUT_DATA2/data/gemini_cache` (or your override) is writable and mounted persistently in Docker/EC2 runs.
3. **(Optional) Disable daily reset** – set `COURSEGEN_DISABLE_CACHE_DAILY_RESET=true` if you want counters to span multiple days (generally not recommended).

## Using the Manager
Most pipelines instantiate `GeminiService` without worrying about the manager:
```python
from services.Gemini.gemini_service import GeminiService
service = GeminiService()  # Auto-wires ApiKeyManager + GeminiApiKeys
```

To customise behaviour:
```python
from services.Gemini.api_key_manager import ApiKeyManager
from services.Gemini.gemini_service import GeminiService

manager = ApiKeyManager(["REDACTED_API_KEY...1", "REDACTED_API_KEY...2"])
service = GeminiService(api_key_manager=manager, model="gemini-2.5-flash")

prompt = "Summarise Fourier series."
response = service.generate(prompt)
```

## Inspecting Usage
- **Read the cache file directly**:
  ```bash
  jq '.' OUTPUT_DATA2/data/gemini_cache/api_key_cache.json
  ```
- **Quick Python probe**:
  ```python
  from services.Gemini.api_key_manager import ApiKeyManager
  mgr = ApiKeyManager()
  print(mgr.cache_data["keys"])
  ```
- **Rotate manually**:
  ```python
  mgr.rotate_key(model="flash")
  ```

## Exhaustion Handling
- When a generation call fails with quota errors, the manager:
  1. Marks the active key exhausted for the relevant model.
  2. Attempts to rotate to the next available key.
  3. If every key is exhausted, sets `exhausted_reason` for all keys, raises a `RuntimeError` (`🚨 ALL API KEYS EXHAUSTED - TERMINATING OPERATIONS 🚨`), and optionally triggers an email alert via `services.Email.email_service`.
- Pipelines such as question generation capture this exception, flush any results already written, send email notifications, and halt further processing. Resume after new keys or quota resets becomes trivial—rerun the same command; the manager starts fresh the next day.

## Environment Hooks
- `COURSEGEN_CACHE_ROOT` – base directory for cache files (defaults to `<repo>/OUTPUT_DATA2`).
- `COURSEGEN_DISABLE_CACHE_DAILY_RESET` – skip the midnight reset (useful for testing, not production).
- `COURSEGEN_DEBUG_DUMP_DIR` – controls where failed Gemini payloads land; indirectly useful when debugging key exhaustion because it co-locates with cache data.

## Integration Points
- **Question generator** – checks `ApiKeyManager.all_keys_exhausted()` before each subtopic and aborts gracefully when true.
- **OCR / Embedding fallbacks** – other pipelines can reuse the same manager to share quota knowledge across services.
- **Email notifications** – `ApiKeyManager.force_terminate_if_all_exhausted()` invokes `EmailService.send_termination_notification` if available, providing the number of exhausted keys and the model family.

## Operational Tips
- Keep at least **twice** the number of keys as you expect concurrent workers (e.g., 10 keys for 3–4 workers) to absorb bursts.
- Rotate or invalidate compromised keys by editing `api_key_cache.json` (set `exhausted=true`) or removing them from `GeminiApiKeys`.
- Back up `api_key_cache.json` before large runs if you need an audit trail of usage.
- The manager does not mask keys in-memory; treat the cache directory as sensitive.

With a healthy key pool and the persistent cache mounted, CourseGen can sustain high-throughput Gemini usage without hitting per-key limits or silently degrading throughput.
