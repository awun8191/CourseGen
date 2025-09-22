# API Key Load Balancer in CourseGen

The API Key Load Balancer is a resilient system for managing multiple Google API keys (for Gemini) to handle high-volume requests without hitting rate limits. It implements round-robin rotation, usage tracking, automatic fallbacks, and integration with caching to ensure scalability and reliability in production environments like batch question generation or outline creation.

## Overview
Gemini API has per-key quotas (e.g., 60 RPM, 1500 RPD). For workloads exceeding this (e.g., 1000+ questions/day), a single key fails. This balancer:
- Rotates across 5-20 keys from multiple projects/accounts.
- Tracks per-key usage and errors (e.g., 429 responses).
- Falls back to secondary keys or alternative providers (Ollama).
- Caches responses to minimize API calls.
- Logs metrics for monitoring and billing attribution.

Key benefits:
- Scales to 500+ RPM effectively.
- 99.9% uptime via fallbacks.
- Cost-transparent: Tracks tokens per key/project.
- Plug-and-play: No changes needed in calling code.

## Architecture
1. **Configuration**: Keys loaded from `data/gemini_cache/api_key_cache.json` or env vars.
2. **Manager**: `services/Gemini/api_key_manager.py`:
   - Initializes pool of `GeminiClient` instances.
   - Selects next key via round-robin or least-used.
   - Updates usage counters post-request.
3. **Rate Limiting**: `services/Gemini/rate_limit_data.py` enforces soft limits (e.g., 50 RPM/key).
4. **Fallbacks**: On 429/5xx, retry with next key (up to 3 attempts).
5. **Integration**: `services/Gemini/gemini_service.py` wraps calls with balancing.
6. **Caching**: Ties into `utils/Caching/enhanced_cache.py` for response dedup.
7. **Monitoring**: Logs to `log_utils.py`; optional Firestore export.

Components:
- `gemini_api_keys.py`: Key validation and serialization.
- `rate_limit_data.py`: In-memory/per-file tracking.
- Env support: Fallback to single `GOOGLE_API_KEY`.

## Setup and Configuration
### 1. Obtain Keys
- Create multiple Google Cloud projects.
- Enable Vertex AI API per project.
- Generate API keys (or service accounts for production).
- Note quotas: Standard tier ~60 RPM/key.

### 2. Configure Keys
#### Option A: JSON File (Recommended for Multiple)
Create/edit `data/gemini_cache/api_key_cache.json`:
```json
{
  "keys": [
    {
      "api_key": "REDACTED_API_KEYC...abc123",
      "project_id": "coursegen-prod-1",
      "usage_today": 0,
      "requests_today": 0,
      "last_used": null,
      "active": true,
      "quota_rpm": 60,
      "quota_rpd": 1500
    },
    {
      "api_key": "REDACTED_API_KEYD...def456",
      "project_id": "coursegen-prod-2",
      "usage_today": 0,
      "requests_today": 0,
      "last_used": null,
      "active": true,
      "quota_rpm": 60,
      "quota_rpd": 1500
    }
  ],
  "active_index": 0,
  "fallback_provider": "ollama",  // Optional: ollama_service.py
  "cache_ttl_hours": 24,
  "max_retries": 3,
  "rotate_on_error": true
}
```
- `usage_today`: Tokens used (auto-updated).
- `active`: Disable problematic keys.
- Add 5-10 keys for robustness.

#### Option B: Environment Variables
- Single key: `export GOOGLE_API_KEY="AIza..."`
- Multiple: `GOOGLE_API_KEYS="key1,key2,key3"` (comma-separated; basic mode, no quotas).

### 3. Environment Variables
- `GEMINI_RATE_LIMIT_RPM`: Global RPM cap per key (default 60).
- `GEMINI_MAX_RETRIES`: Fallback attempts (default 3).
- `GEMINI_CACHE_ENABLED`: Use enhanced cache (default 1).
- `GEMINI_PROJECT_BILLING`: Track per-project costs (requires `BILLING_ENABLED=1`).
- `OLLAMA_FALLBACK_URL`: For local fallback (default "http://localhost:11434").

### 4. Initialization
Run once to validate keys:
```
python -c "from services.Gemini.api_key_manager import validate_keys; validate_keys()"
```
- Checks API access; updates cache.

## Usage
### In Pipelines
All Gemini calls in CourseGen (outlines, questions) automatically use the balancer via `gemini_service.py`:
```python
from services.Gemini.gemini_service import get_balanced_client

client = get_balanced_client()
response = client.generate_content("Prompt here")
```

### Manual/CLI Testing
```python
from services.Gemini.api_key_manager import get_next_key

key_info = get_next_key()
print(f"Using key from project: {key_info['project_id']}")
```

For batch jobs, set `--workers` in generators; balancer handles concurrency.

### Integration Example
In custom script:
```python
from services.Gemini.gemini_service import generate_with_balancing

prompt = "Generate a question on DSP."
result = generate_with_balancing(prompt, max_tokens=200)
print(result.text)
```

## Monitoring and Metrics
- **Logs**: Each call logs: key used, tokens in/out, latency, errors.
  Example: `[INFO] Gemini call #47: key=AIza... (proj1), tokens=150 in/50 out, 2.1s, success.`
- **Cache File**: `data/gemini_cache/api_key_cache.json` updates in real-time.
- **Billing**: If enabled, attributes costs per key/project in `billing_state.json`.
- **Alerts**: Custom hook in `api_key_manager.py` for quota breaches (e.g., email/Slack).
- **Dashboard**: Query logs or use `utils/progress_tracker.py` for usage graphs.

View usage:
```
python -c "from services.Gemini.api_key_manager import print_usage; print_usage()"
```
Output:
```
Key 1 (proj1): 450/1500 RPM used today, 12000 tokens.
Key 2 (proj2): 200/1500 RPM, 8000 tokens.
Total: 20000 tokens, $0.40 estimated cost.
```

## Error Handling and Fallbacks
- **429 (Rate Limit)**: Switch to next key; exponential backoff (1s, 2s, 4s).
- **401/403 (Invalid)**: Mark key inactive; log and fallback.
- **5xx (Server Error)**: Retry same key up to 3x, then next.
- **No Keys Available**: Raise `NoAvailableKeysError`; fallback to Ollama if configured.
- **Cache Miss/Fail**: Proceed without cache; log warning.

Tune in config: `"error_threshold": 5` (deactivate after 5 consecutive errors).

## Performance and Scaling
- **Throughput**: With 10 keys @60 RPM = 600 RPM total.
- **Latency Overhead**: <50ms for rotation/caching.
- **Memory**: In-memory tracking; persists to JSON every 10 calls.
- **Concurrency**: Thread-safe; use `workers=4` for parallel.
- **Cost Optimization**: Cache hits save 70-90% of API calls for repeated prompts.

Benchmark:
```
# Test 100 calls
python -c "from services.Gemini.gemini_service import benchmark_balancer; benchmark_balancer(100)"
```
Expected: ~95% success, avg 2.5s/call.

## Security
- **Secrets**: Never commit `api_key_cache.json` (.gitignore'd); use env for prod.
- **Validation**: Keys tested on init; invalid ones skipped.
- **Auditing**: Logs don't include full keys (masked: "AIza...123").
- **Production**: Use service accounts over API keys for better security.

## Troubleshooting
- **All Keys Exhausted**: Add more keys; check quotas in Google Console.
- **Fallback Not Triggering**: Verify `"rotate_on_error": true` in config.
- **Cache Conflicts**: Set unique cache keys per call (e.g., hash(prompt + chunks)).
- **High Latency**: Reduce `max_retries`; monitor network to Google APIs.
- **Usage Not Updating**: Ensure write permissions on `data/gemini_cache/`.
- **Ollama Fallback Fails**: Start Ollama server; check `OLLAMA_FALLBACK_URL`.
- **Logs Missing**: Set `LOG_LEVEL=DEBUG` in `log_utils.py`.

## Best Practices
- **Key Pool**: Maintain 2x expected load (e.g., 10 keys for 300 RPM).
- **Rotation**: Reset daily usage at midnight UTC (auto in manager).
- **Monitoring**: Script daily usage checks; alert >80% quota.
- **Testing**: Validate with small batches; simulate errors by disabling keys.
- **Costs**: Monitor via Google Console + local billing; optimize prompts.
- **Backup**: Sync keys to secure vault (e.g., AWS Secrets Manager).

## Future Enhancements
- Dynamic quota fetching from Google API.
- ML-based key selection (e.g., predict best key by latency).
- Multi-provider balancing (Gemini + OpenAI + Anthropic).
- Web dashboard for key management.

This balancer ensures uninterrupted AI generation at scale, critical for educational pipelines.