# Engineering Hub RAG Question Generator

This module generates exactly **20 questions per subtopic** (10 theory + 10 calculation) for engineering courses, leveraging RAG to ensure relevance to specific topics and learning objectives. It uses Gemini AI to create high-quality MCQ questions with detailed explanations, solution steps for calculations, and source references. The system includes sophisticated caching and progress tracking to handle interruptions and resume functionality.

## Overview
The Engineering Hub RAG Question Generator automates the creation of assessment questions for all courses while maintaining pedagogical quality:
- **Exactly 20 questions per subtopic**: 10 theory questions + 10 calculation questions
- **All courses processed**: Automatically discovers and processes all courses from courses.json
- **Theory questions**: Multiple choice with 4 options, detailed explanations, and source citations
- **Calculation questions**: Multiple choice with 4 options, step-by-step LaTeX-formatted solutions
- **RAG-powered**: Retrieves context-specific chunks from ChromaDB for accurate, grounded questions
- **Robust caching**: Fine-grained progress tracking with cache.json structure and in-memory warm-cache reuse
- **Resume functionality**: Survives container restarts and network interruptions with automatic recovery of interrupted batches
- **Progress tracking**: Both cache.json (fine-grained) and Firestore (coarse-grained) progress tracking

Key benefits:
- Generates exactly 20 questions per subtopic with consistent quality
- Reduces bias/hallucinations via RAG grounding with multiple context chunks
- Sophisticated caching prevents regeneration of completed batches
- Resume functionality handles interruptions gracefully
- Progress tracking enables monitoring of generation status
- LaTeX formatting for mathematical content in calculation questions

## Architecture
1. **Input**: Course outline MD/JSON (`--input-outline`), or direct query/topic (e.g., "Explain Fourier transforms in DSP").
2. **Retrieval**: `services/QuestionRag/utils/chromadb_query.py` fetches top-k chunks, filtered by metadata (e.g., `LEVEL=400`, `CATEGORY=PQ` for past questions).
3. **Generation**: `services/QuestionRag/question_generator.py` (or `gemini_question_gen.py`):
   - Loads prompts from resources (e.g., JSONL templates for MCQ vs. essay).
   - Calls balanced Gemini client (`gemini_service.py`).
   - Generates per-topic: Questions + Answer + Explanation + Difficulty + Bloom Level + Sources.
4. **Post-Processing**: Deduplicates (`utils/Remove Duplicates/`), formats (JSONL/MD), caches responses.
5. **Output**: JSONL files (e.g., `data/questions_EEE471.jsonl`) or integrated with Firestore.
6. **Batching**: `services/QuestionRag/utils/batch_utils.py` for parallel processing across topics/courses.

Dependencies:
- Embeddings/ChromaDB.
- Gemini keys (load balanced).
- Optional: `data/courses_fallback_with_topics.json` for topic extraction.

## Usage
### Prerequisites
- Embeddings processed (see [Convert to Embeddings README](README_convert_to_embeddings.md)).
- Course outlines available in `data/textbooks/courses.json`.
- Firestore configured with service account credentials.
- Gemini API keys configured in `services/Gemini/gemini_api_keys.py` (load balancing automatic).

### CLI Command
```bash
python -m services.QuestionRag.pipelines.question_generator [OPTIONS]
# By default, processes all courses in courses.json
```

### EC2 Helper Script (`ec2_execution.sh`)
- Syncs the latest `courses.json` and embeddings from the published Docker image onto an EC2 host before executing the container.
- Respects your `.env` (defaulting to `~/.env`) and reuses the same question-generation flags as `run.sh` (course code, theory/calculation counts, resume controls, etc.).
- Mounts host paths for cache (`~/OUTPUT_DATA2/cache`), textbooks (`~/data/textbooks/courses.json`), and embeddings (`~/OUTPUT_DATA2/emdeddings`) so progress persists between runs while still picking up fresh assets on every pull.
- Mirrors the Gemini API key cache (`~/OUTPUT_DATA2/data/gemini_cache`) so key rotation state persists across runs.
- Sends email notifications via `services/Email/email_service.py` whenever the container finishes or crashes (enable with `EMAIL_NOTIFICATIONS_ENABLED=true` and provide SMTP credentials in the env file).

#### Usage
```bash
# 1. Ensure ~/.env is populated with Gemini, Cloudflare, Firestore credentials
# 2. Pull the latest image & start generation (sync happens automatically)
./ec2_execution.sh --course-code "EEE 471" --request-delay 2.0

# Optional flags
./ec2_execution.sh \
  --env-file ~/.env.production \
  --structured-output \
  --background \
  --port 8000:8000
```

Key behaviour:
- Pass `--skip-sync` to skip copying data from the image, or `--no-pull` to use the local image.
- Use `DATA_DIR`, `CACHE_ROOT`, or `EMBEDDINGS_DIR` env vars to customise host mount locations.
- Override `GEMINI_CACHE_DIR` if you store API key cache elsewhere.
- Background mode keeps the container alive (`docker logs -f coursegen-rag` to follow output); rerun the script to replace an existing container.

#### Key Options
- `--course-code STR`: Course code (e.g., "EEE 471") or "all" for all courses (default: "all")
- `--topics LIST`: Optional list of topics to include (case insensitive)
- `--subtopics LIST`: Optional list of subtopics to include (case insensitive)
- `--cache-dir PATH`: Directory for generation cache (default: "data/gemini_cache")
- `--theory-per-request INT`: Theory questions per request (default: 10)
- `--calc-per-request INT`: Calculation questions per request (default: 5; executed twice per subtopic)
- `--no-resume`: Disable resume functionality (regenerate all questions)
- `--skip-firestore`: Disable Firestore persistence
- `--rag-topk INT`: RAG retrieval pool size (default: 30)
- `--rag-final-k INT`: Context chunks passed to LLM (default: 12)
- `--request-delay FLOAT`: Delay between requests in seconds (default: 1.5)
- `--model STR`: Gemini model name (default: "gemini-2.5-flash")
- `--temperature FLOAT`: Generation temperature (default: 0.25)
- `--max-output-tokens INT`: Maximum tokens per request (default: 6000)

#### Example: Generate Questions for All Courses (Default)
```bash
python -m services.QuestionRag.pipelines.question_generator \
  --cache-dir data/gemini_cache \
  --theory-per-request 10 \
  --calc-per-request 5 \
  --request-delay 2.0 \
  --model gemini-2.5-flash
```

#### Example: Generate Questions for Specific Course
```bash
python -m services.QuestionRag.pipelines.question_generator \
  --course-code "EEE 471" \
  --cache-dir data/gemini_cache \
  --request-delay 1.5
```

#### Example: Resume Interrupted Generation
```bash
python -m services.QuestionRag.pipelines.question_generator \
  --resume \
  --request-delay 1.5
```

#### Example: Generate for Specific Topics Only
```bash
python -m services.QuestionRag.pipelines.question_generator \
  --topics "Z-Transforms" "Fourier Transforms" \
  --subtopics "Properties" "Applications" \
  --cache-dir data/gemini_cache
```

## Caching and Progress Tracking

The system implements sophisticated caching and progress tracking to handle interruptions and enable resume functionality. Recent updates include:
- **Warm payload reuse** so repeated resume runs avoid re-reading completed batch files from disk.
- **Interrupted batch detection** that flags and automatically retries any requests left `in_progress` after an unexpected shutdown.
- **Improved error artifacts** – unparseable Gemini payloads are written to `OUTPUT_DATA2/cache/failed_responses/` and their paths are surfaced directly in the logs for easier debugging.
- **Course progress manifests** – each course maintains a JSON file under `OUTPUT_DATA2/cache/course_progress/` capturing `theory_progress`, `calculation_progress`, `calc_progress2`, and the `state` (`in_progress`, `completed`, or `error`) for every subtopic.
- **Automatic cache cleanup** – once a subtopic is marked completed, its per-request cache files in `OUTPUT_DATA2/cache/question_gen/` are removed, keeping disk usage under control.
- **Firestore writes on completion** – questions are pushed to Firestore only after a subtopic’s theory and both calculation batches are successfully generated.

### Cache Structure
The system uses a hierarchical cache.json structure:
```json
{
  "EEE 471": {
    "Z-Transforms": {
      "Properties": {
        "theory_batches": ["completed"],
        "calc_batches": ["completed"],
        "updated_at": 1640995200.0
      }
    }
  }
}
```

### Progress Tracking
- **Fine-grained tracking**: cache.json tracks individual batch completion status
- **Coarse-grained tracking**: Firestore "GenerationProgress" collection tracks overall course progress
- **Resume functionality**: System loads cache.json on startup and resumes only incomplete batches
- **Interrupted recovery**: Entries left `in_progress` from a prior run are converted to `interrupted` and retried automatically on the next invocation
- **Progress updates**: Updates progress after each batch completion (10 theory questions or 5-question calculation batches)

### Batch Processing
Each subtopic generates exactly 20 questions across 3 batches:
1. **theory-1**: 10 theory questions (difficulty rank 4)
2. **calculation-1**: 5 calculation questions (difficulty rank 6)
3. **calculation-2**: 5 calculation questions (difficulty rank 6)

### Programmatic Usage
```python
from services.QuestionRag.pipelines.config import QuestionBatchConfig
from services.QuestionRag.pipelines.question_generator import QuestionGenerator
from services.Gemini.gemini_service import GeminiService
from services.Firestore.firebase_service import FireStore

# Configure generation for all courses (default)
config = QuestionBatchConfig(
    course_code="all",  # Process all courses
    cache_dir="data/gemini_cache",
    theory_questions_per_request_override=10,
    calc_questions_per_request_override=10,
    resume=True,
    store_firestore=True
)

# Or configure for a specific course
config = QuestionBatchConfig(
    course_code="EEE 471",  # Specific course
    cache_dir="data/gemini_cache",
    resume=True,
    store_firestore=True
)

# Initialize generator
generator = QuestionGenerator(
    gemini_service=GeminiService(),
    firestore=FireStore()
)

# Generate questions
questions = generator.generate_course_questions(config)

# Questions are automatically cached and persisted to Firestore
print(f"Generated {len(questions)} questions")
```

## Prompt Engineering
Templates in `services/QuestionRag/resources/` (JSONL):
```jsonl
{"role": "system", "content": "Generate {num} {type} questions on {topic} from {chunks}. Include 4 options for MCQ with one correct. Difficulty: {difficulty} (Bloom: {level}). Provide explanation and source."}
{"role": "user", "content": "Topic: {topic}\nChunks: {retrieved_text}\nOutput JSON: [{question, type, options[], answer, explanation, difficulty, bloom_level, sources[]}]"}
```
- Customize: Add distractor strategies for MCQs, word limits for essays.
- Test: Run with small `--num-questions` and inspect for accuracy.

## Output Format
### JSONL (Default)
Each line:
```json
{
  "id": "q_EEE471_ztransform_1",
  "question": "What is the z-transform of the unit step sequence u[n]?",
  "type": "mcq",
  "difficulty": "medium",
  "bloom_level": "understand",
  "options": [
    "A) 1/(1 - z^-1)",
    "B) 1/z",
    "C) z/(z-1)",
    "D) 1"
  ],
  "correct_answer_indexes": [0],
  "answer": "A",
  "explanation": "The z-transform of u[n] is the sum from n=0 to inf of z^-n = 1/(1 - z^-1) for |z| > 1.",
  "sources": [
    {"path": "EEE471_textbook.pdf", "chunk_index": 45, "page": 23}
  ],
  "topic": "Z-Transforms",
  "generated_at": "2025-09-22T10:00:00Z"
}
```

### Markdown
```
## Questions for Module 1: Z-Transforms

### Q1 (MCQ, Medium)
What is the z-transform of the unit step sequence u[n]?

A) 1/(1 - z^-1)  
B) 1/z  
C) z/(z-1)  
D) 1  

**Answer**: A  
**Explanation**: The z-transform of u[n] is ...  
**Source**: EEE471_textbook.pdf (page 23)
```

## Integration
- **With Outlines**: Auto-generate questions per module.
- **With Platforms**: Export to JSONL for Quizlet/Moodle import.
- **With Firestore**: Store for real-time quiz apps.
- **Batch from Catalog**: Use `data/courses.json` + `batch_utils.py`.
- **Customization**: Extend `question_model.py` for new types (e.g., fill-in-blank).

## Testing and Validation
- **Unit Tests**: `pytest tests/test_gemini_question_gen_cache.py`, `test_filter.py`.
- **Retrieval Tests**: `test_chromadb_query.py` for relevance.
- **Quality Checks**: Manual review; compute diversity (e.g., unique topics covered).
- **Edge Cases**: Test low-retrieval scenarios, duplicate questions.
- **Performance**: Time 100 questions; ensure <5s/question with caching.
- **Live probe**: Use `python scripts/calc_probe.py ...` to inspect prompts, raw Gemini responses, and parsed JSON for a single calculation batch when debugging.

## Troubleshooting
- **Irrelevant Questions**: Refine retrieval filters or increase `--rag-topk`.
- **Rate Limits**: Load balancer uses `services/Gemini/gemini_api_keys.py`. Use `--request-delay` to add delays between requests.
- **API Key Issues**: Check `services/Gemini/gemini_api_keys.py` for valid keys. Verify keys in Google AI Studio.
- **API Key Exhaustion**: The generator halts once all Gemini keys report quota exhaustion and logs the failure; no additional requests are attempted until quotas reset or new keys are added.
- **Hallucinations**: Lower `--temperature`; verify sources match chunks. Check RAG context quality.
- **Format Errors**: Validate with `question_model.py`; debug prompts. Ensure calculation questions have proper LaTeX formatting.
- **Cache Issues**: Clear `data/gemini_cache/` if stale. Check cache.json structure for corruption.
- **Resume Problems**: Verify cache.json exists and has correct structure. Use `--no-resume` to regenerate everything.
- **Firestore Errors**: Check service account credentials and network connectivity.
- **Progress Tracking**: Monitor both cache.json and Firestore "GenerationProgress" collection for status.
- **Logs**: Use `services/RAG/log_utils.py` for traces. Enable debug logging with `COURSEGEN_QG_LOGLEVEL=DEBUG`.
- **Failed payload dumps**: Unparseable Gemini responses are persisted under `OUTPUT_DATA2/cache/failed_responses/`; logs include the exact file path for inspection.

## Best Practices
- **Batch Processing**: The system generates exactly 20 questions per subtopic (10 theory + 10 calculation) across one theory batch and two 5-question calculation batches.
- **Resume Strategy**: Always use `--resume` flag to avoid regenerating completed batches. The system handles interruptions gracefully.
- **Progress Monitoring**: Monitor both cache.json (fine-grained) and Firestore "GenerationProgress" (coarse-grained) for status.
- **Resource Management**: Use appropriate `--request-delay` to avoid rate limits. Default 1.5s between requests is usually sufficient.
- **Quality Control**: Review generated questions for accuracy, especially calculation questions with LaTeX formatting.
- **Cache Management**: Keep cache.json clean and backed up. Clear cache directory only when necessary.
- **Error Handling**: The system includes retry logic for failed requests (up to 3 attempts per batch).
- **Testing**: Test with small courses first to verify the workflow before large-scale generation.

## Future Enhancements
- **Adaptive Generation**: Adjust difficulty based on user performance and learning analytics.
- **Multi-modal Questions**: Include diagram-based and interactive questions.
- **Auto-grading**: Implement automated grading of student responses using Gemini.
- **Question Variants**: Generate multiple variants of questions for different assessments.
- **Performance Analytics**: Track question effectiveness and student performance metrics.
- **Collaborative Features**: Enable multiple educators to contribute to question banks.
- **Localization**: Support for multiple languages and educational contexts.

## Summary

The Engineering Hub RAG Question Generator implements a robust, production-ready system for generating exactly 20 questions per subtopic across all courses with sophisticated caching, progress tracking, and resume functionality. The system ensures:

- **Consistent Quality**: Exactly 20 questions per subtopic (10 theory + 10 calculation)
- **Automatic Processing**: Discovers and processes all courses from courses.json by default
- **Reliability**: Robust error handling with retry logic and resume functionality
- **Efficiency**: Smart caching prevents regeneration of completed work
- **Monitoring**: Comprehensive progress tracking at both fine-grained and coarse-grained levels
- **Scalability**: Handles multiple courses with many topics and subtopics

This system empowers educators to build comprehensive, high-quality question banks efficiently while maintaining control over the generation process and ensuring pedagogical standards are met.
