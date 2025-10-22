# Engineering Hub Question Generation Pipeline

This pipeline produces **20 fully grounded questions per subtopic** (10 theory + two 5-question calculation batches) across every course that already has an outline in `courses.json`. It retrieves context from ChromaDB, calls Gemini through the in-repo API key balancer, and persists progress so long-running jobs survive restarts.

## Overview
- **Deterministic batches**: `theory-1`, `calculation-1`, `calculation-2` ensure exactly 20 questions per subtopic.
- **RAG-driven prompts**: `ChromaQuery` supplies filtered chunks (metadata + similarity gating) before each Gemini request.
- **High-throughput execution**: Topic-parallel workers (ThreadPool) process multiple topics at once while honouring request delays.
- **Resilience**: Disk-backed caches, `CourseProgressCache`, and Firestore snapshots allow interruption-free resumes.
- **Operational safeguards**: Gemini API keys rotate via `ApiKeyManager`; exhaustion triggers a controlled shutdown and optional email alerts.
- **Calculation quality**: Step-by-step LaTeX solutions are normalized, and solution steps can be wrapped/unwrapped with `--no-latex-wrap`.

### Key Components
| Concern | Module |
| --- | --- |
| RAG retrieval | `services/QuestionRag/utils/chromadb_query.py` |
| Prompt assembly | `services/QuestionRag/pipelines/prompt_utils.py` |
| Gemini orchestration | `services/Gemini/gemini_service.py` |
| Batch parsing | `services/QuestionRag/pipelines/json_utils.py` |
| Progress + cache | `services/QuestionRag/utils/course_progress.py`, `services/QuestionRag/utils/cache.py` |
| Topic parallelism | `services/QuestionRag/pipelines/worker_pool.py` |
| Firestore integration (optional) | `services/Firestore/firebase_service.py` |

## Architecture
1. **Outline discovery** – loads `courses.json` and filters courses that already supply `outline` blocks.
2. **Per-topic fan out** – `TopicWorkerPool` assigns topics to workers (default 3 threads) unless `--disable-parallel` is provided.
3. **RAG retrieval** – per subtopic, `ChromaQuery` fetches `rag_topk` candidates, prunes with similarity thresholds, and slices into context windows (`rag_context_limit` per request). Optional metadata filtering via `--rag-where`.
4. **Prompt construction** – `build_question_generation_prompt` assembles request metadata (difficulty, Bloom level, request kind) and RAG snippets.
5. **Gemini call** – `GeminiService` selects a key using `ApiKeyManager`, applies structured output or thinking mode if requested, and retries failed calls (`request_attempts`).
6. **Validation + caching** – responses are parsed into `GeminiQuestionBatch`, coerced to `Question` Pydantic models, stored in `QuestionCache`, and progress counters are updated.
7. **Persistence** – Firestore updates run after both calculation batches succeed; JSONL export happens when `--output-jsonl` is provided.
8. **Notifications** – optional `services.Email.email_service` emits start/completion/emergency emails (Docker helper script wires this up automatically).

## Prerequisites
- Populate `services/Gemini/gemini_api_keys.py` with valid Gemini API keys **or** expose them via environment variables before starting the container.
- Ensure embeddings exist in ChromaDB (see [README_convert_to_embeddings.md](README_convert_to_embeddings.md)).
- `courses.json` must contain `outline` arrays per course. Default path resolves through `config.py` or `COURSEGEN_COURSES_JSON`.
- Optional: set up Firestore credentials and `EMAIL_NOTIFICATIONS_ENABLED=true` if you want completion emails.

## CLI Usage
```bash
python -m services.QuestionRag.pipelines.question_generator [OPTIONS]
```
Important flags (see `build_arg_parser()` for the full list):

| Flag | Purpose / Default |
| --- | --- |
| `--course-code STR` | Course code (`"all"` by default processes every outlined course). |
| `--courses-json PATH` | Overrides the course catalog location. |
| `--cache-dir PATH` | Root for caches; defaults to `OUTPUT_DATA2/cache` via central config. |
| `--rag-topk`, `--rag-final-k`, `--rag-tau`, `--rag-min-sim` | Retrieval tuning knobs. |
| `--rag-where JSON` | Extra metadata filter, e.g. `'{"LEVEL": {"$eq": "400"}}'`. |
| `--theory-per-request`, `--calc-per-request` | Targets per Gemini request (calc resets to 5 internally to keep two batches). |
| `--temperature`, `--top-p`, `--model`, `--max-output-tokens` | Gemini generation controls. |
| `--structured-output` / `--no-structured-output` | Toggle Gemini schema-based parsing. |
| `--thinking`, `--thinking-budget` | Enable Gemini thinking mode for compatible models. |
| `--request-delay`, `--delay-jitter` | Back pressure between API calls. |
| `--request-attempts`, `--rag-attempts` | Retry counts for Gemini and retrieval respectively. |
| `--topics`, `--subtopics` | Case-insensitive filters. |
| `--output-jsonl PATH` | Dump all questions produced in this run to a single JSONL file. |
| `--disable-parallel` | Force sequential topic processing. |
| `--skip-firestore`, `--no-resume`, `--no-latex-wrap` | Opt out of persistence, resume cache, or LaTeX formatting. |

### Command Examples
- **Default full run (all courses):**
  ```bash
  python -m services.QuestionRag.pipelines.question_generator
  ```
- **Single course with structured output and slower pacing:**
  ```bash
  python -m services.QuestionRag.pipelines.question_generator \
    --course-code "EEE 471" \
    --structured-output \
    --request-delay 2.0 \
    --output-jsonl OUTPUT_DATA2/questions_EEE471.jsonl
  ```
- **Target a subset of subtopics with metadata filter:**
  ```bash
  python -m services.QuestionRag.pipelines.question_generator \
    --course-code "MTH 313" \
    --topics "Complex Analysis" \
    --subtopics "Residue Calculus" \
    --rag-where '{"CATEGORY": {"$in": ["TEXTBOOK", "PAST_QUESTIONS"]}}'
  ```
- **Disable parallel workers (useful on small machines or when debugging):**
  ```bash
  python -m services.QuestionRag.pipelines.question_generator --disable-parallel
  ```

### EC2 Helper Script
`./ec2_execution.sh` wraps an ECR-hosted container, syncs textbook data + embeddings onto the host, mounts persistent volumes (`~/OUTPUT_DATA2/...`), and mirrors the Gemini cache so key rotation survives container churn. Use flags like `--course-code`, `--structured-output`, `--background`, `--skip-sync`, or `--env-file` as needed; email notifications propagate automatically when SMTP credentials are present.

## Batch Semantics
Every subtopic always attempts these batches:

1. `theory-1` → 10 MCQs (default difficulty rank from `question_gen_config`).
2. `calculation-1` → 5 calculation MCQs with LaTeX-rendered solutions.
3. `calculation-2` → another 5 calculation MCQs to reach 10 total calculations.

`QuestionBatchConfig.request_plan()` hard-codes this structure so you get consistent counts even when overrides are provided.

## Programmatic Usage
```python
from pathlib import Path

from services.QuestionRag.pipelines.config import QuestionBatchConfig
from services.QuestionRag.pipelines.question_generator import QuestionBatchRunner, QuestionGenerator
from services.Gemini.gemini_service import GeminiService
from services.Gemini.api_key_manager import ApiKeyManager
from services.Gemini.gemini_api_keys import GeminiApiKeys

api_keys = GeminiApiKeys().get_keys()
generator = QuestionGenerator(
    gemini_service=GeminiService(api_key_manager=ApiKeyManager(api_keys)),
    use_structured=True,
)

config = QuestionBatchConfig(
    course_code="EEE 471",
    courses_json_path=Path("data/textbooks/courses.json"),
    cache_dir=Path("OUTPUT_DATA2/cache"),
    resume=True,
    store_firestore=False,
)

runner = QuestionBatchRunner(generator)
questions = runner.run_parallel(config)
print(f"Generated {len(questions)} questions")
```

## Caching & Progress Tracking
- **Question cache** – `OUTPUT_DATA2/cache/question_gen/cache.json` indexes per-request payloads and metadata; each successful batch stores JSON to disk for reuse.
- **Course progress** – `OUTPUT_DATA2/cache/course_progress/{course}.json` records theory/calculation counters, request state, completion status, and timestamps.
- **Error dumps** – raw Gemini payloads that fail schema validation are written to `OUTPUT_DATA2/cache/failed_responses/` (override with `COURSEGEN_DEBUG_DUMP_DIR`).
- **Firestore** – the `GenerationProgress` collection tracks per-course status once both calculation batches complete.
- **Resume workflow** – interrupted batches are marked `in_progress`; next run upgrades them to `interrupted`, clears stale cache entries, and retries automatically. `--no-resume` forces regeneration from scratch.

### Sample `course_progress` entry
```json
{
  "course": "EEE 471",
  "topics": {
    "Z-Transforms": {
      "subtopics": {
        "Properties": {
          "theory_progress": 10,
          "calculation_progress": 5,
          "calc_progress2": 5,
          "state": "completed",
          "persisted": true,
          "completed_at": 1716503251.291
        }
      }
    }
  }
}
```

## Operational Notes
- **API key exhaustion** – `ApiKeyManager` tracks RPM/RPD per model family, stores state in `OUTPUT_DATA2/data/gemini_cache/api_key_cache.json`, and aborts runs with a clear 🚨 message when every key is exhausted. Email alerts fire when configured.
- **Parallelism defaults** – `qg_enable_topic_parallelism` (central config) toggles thread usage; CLI `--disable-parallel` overrides it at runtime.
- **Delays & jitter** – `request_delay_s` and `delay_jitter` limit concurrency pressure on Gemini. Jitter is applied multiplicatively (`±25%` by default).
- **LaTeX wrapping** – disabled via `--no-latex-wrap` for consumers that do not support math delimiters.
- **Output export** – `--output-jsonl` writes the exact questions returned in this session; Firestore / local caches remain authoritative for resume runs.

## Troubleshooting
- **No RAG context** → verify embeddings exist for the course code; optionally allow broader metadata via `--rag-where` or lower `--rag-min-sim`.
- **Duplicate questions** → delete the relevant cache entry under `OUTPUT_DATA2/cache/question_gen` and rerun with `--no-resume`.
- **Key rotation stalls** → confirm `services/Gemini/gemini_api_keys.py` is populated and that the cache directory is writable; resetting quotas may require deleting `api_key_cache.json`.
- **Firestore errors** → check credentials and network; use `--skip-firestore` to continue locally if Firestore is unavailable.
- **Validation failures** → inspect dumped payload in `failed_responses/`; structured output (`--structured-output`) often eliminates parsing issues.
- **Slowdowns** → reduce `--max-topic-workers`, bump `--request-delay`, or filter via `--topics`/`--subtopics` while debugging.

## Best Practices
- Run `python -m services.QuestionRag.pipelines.question_generator --course-code "all"` nightly so caches stay warm and Firestore progress stays fresh.
- Keep Gemini keys in sync across environments and mount `OUTPUT_DATA2/data/gemini_cache` when running in containers so daily usage survives restarts.
- Review a sample from each course regularly; calculation questions rely on LaTeX rendering, so validate consumer support.
- Monitor `OUTPUT_DATA2/cache/course_progress/*.json` and Firestore dashboards to catch stalled subtopics early.

## Future Enhancements
- Adaptive difficulty selection based on historic performance.
- Backpressure from Firestore to pause problematic courses automatically.
- Multi-model generation (e.g., fallback to Gemini Pro for hard topics).

The question generator is the backbone of CourseGen’s assessment tooling—treat the cache directories and API key pool as critical infrastructure to keep question production reliable.
