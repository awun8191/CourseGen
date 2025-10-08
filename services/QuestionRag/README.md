# QuestionRag: RAG-Powered Question Generation Pipeline

Generates 30 MCQs per subtopic (20 theory, 10 calc) using Gemini + Chroma RAG, with caching, Firestore persistence, and LaTeX for calculations. All questions grounded in retrieved docs; skips if no context.

## Usage
- Single course: `python -m services.QuestionRag.gemini_question_gen --generate-questions --course-code EEE301`
- All courses (with outlines): `python -m services.QuestionRag.gemini_question_gen --generate-questions`
  (Skips courses without outlines; processes all depts.)

- Requires pre-generated outlines in courses.json (from course_outline_generator).
- Embeddings in OUTPUT_DATA2/chroma (via services.RAG.convert_to_embeddings).
- Metadata: course code/name, topic/subtopic, level/semester from courses.json; question/options/answer/explanation/steps/rag_sources generated.

### Parallel Processing
The question generation now supports **topic-level parallelism** for significantly faster processing:

- **Multi-worker mode** (default): Processes multiple topics simultaneously using 2-4 worker threads
- **Sequential mode**: Use `--disable-parallel` for single-threaded processing (original behavior)

**Performance**: ~2-3x faster for courses with multiple topics while respecting API rate limits.

### CLI Examples

```bash
# Use default parallel processing (3 workers)
python -m services.QuestionRag.gemini_question_gen --generate-questions --course-code EEE301

# Custom worker configuration
python -m services.QuestionRag.gemini_question_gen --generate-questions \
  --max-topic-workers 4 \
  --worker-timeout 600 \
  --worker-retry-attempts 3

# Disable parallel processing (sequential mode)
python -m services.QuestionRag.gemini_question_gen --generate-questions \
  --disable-parallel

# All courses with custom worker settings
python -m services.QuestionRag.gemini_question_gen --generate-questions \
  --max-topic-workers 2 \
  --worker-timeout 300
```

## Docker
Use root docker-compose.yml: `docker-compose run --rm coursegen python -m services.QuestionRag.gemini_question_gen ...`

## Cache & Resume
- Cache: OUTPUT_DATA2/cache (JSONL per request key: course-topic-subtopic-request).
- Resume: Loads completed batches; skips failed (marked with reason).

## Validation
- 4 unique options (A-D).
- Valid correct answer.
- Non-empty question/explanation.
- Calc: Non-empty LaTeX-wrapped steps.

## Configuration

### Worker Configuration (New)
Configure parallel processing via environment variables or CLI arguments:

| Setting | Environment Variable | CLI Argument | Default | Description |
|---------|---------------------|--------------|---------|-------------|
| Max Workers | `QG_MAX_TOPIC_WORKERS` | `--max-topic-workers` | 3 | Number of concurrent topic workers (2-4 recommended) |
| Worker Timeout | `QG_WORKER_TIMEOUT` | `--worker-timeout` | 300 | Timeout in seconds for individual workers |
| Retry Attempts | `QG_WORKER_RETRY_ATTEMPTS` | `--worker-retry-attempts` | 2 | Number of retries for failed workers |
| Enable Parallel | `QG_ENABLE_TOPIC_PARALLELISM` | `--disable-parallel` | true | Disable to use sequential processing |

### Existing Configuration
See question_generator.py for additional config options (e.g., rag_topk=30, temperature=0.25, request delays, etc.).

## Technical Details

### Multi-Worker Architecture
- **Thread Pool**: Uses `ThreadPoolExecutor` for concurrent topic processing
- **Resource Safety**: Thread-safe access to API keys, caches, and Firestore
- **Error Isolation**: Individual worker failures don't affect other workers
- **Progress Tracking**: Real-time progress updates across all workers
- **Automatic Retry**: Failed workers are automatically retried up to configured limit

### Worker Pool Features
- Configurable number of workers (2-4 recommended)
- Individual worker timeouts and retry attempts
- Graceful shutdown and resource cleanup
- Comprehensive error logging and reporting
