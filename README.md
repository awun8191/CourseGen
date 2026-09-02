# CourseGen: Recursive PDF Extraction, RAG, and Automated Course Content Generation

CourseGen is a modular pipeline for processing educational materials (PDFs, images, textbooks) through OCR, embedding, retrieval-augmented generation (RAG), and automated content creation. It supports generating course outlines, interactive questions, and more, with resumable processing, multi-provider API integration (Gemini, Cloudflare, Ollama), robust caching, cost tracking, and scalable architecture.

## Key Features

- **End-to-End Pipeline**: From raw PDFs to searchable embeddings and AI-generated educational content.
- **Robust OCR**: Handles scanned documents with preprocessing, rotation detection, and tunable Tesseract parameters.
- **Intelligent RAG**: Semantic search over ChromaDB with metadata filtering (e.g., by course code, department).
- **Automated Generation**: Course outlines and questions using structured Gemini prompts.
- **Load Balancing**: Rotates API keys to handle high-volume requests without rate limits.
- **Observability**: Detailed logging, progress tracking, billing ledgers, and resumability.
- **Modularity**: Separate services for RAG, Question Generation, providers (Gemini/Cloudflare/Ollama/Firestore), and utilities.
- **Scalability**: Parallel processing, batching, and caching for large datasets (e.g., university course libraries).
- **Docker Integration**: Full containerization with persistent volumes and AWS ECR deployment.
- **Production Ready**: Optimized for deployment to AWS ECS, EKS, or other container platforms.

The project is organized into `services/` (core pipelines), `data_models/` (Pydantic schemas), `utils/` (helpers), `data/` (inputs/outputs, gitignored), and `tests/` (PyTest suite).

## Project Structure

```
services/
├── RAG/                     # PDF ingestion, OCR, chunking, embedding, ChromaDB storage
├── QuestionRag/             # Course outline and question generation
│   ├── pipelines/           # course_outline_generator.py, question_generator.py
│   └── utils/               # chromadb_query.py, batch_utils.py, cache.py, courses.py
├── Gemini/                  # API client with key load balancing
├── Cloudflare/              # BGE-M3 embedding client
├── Ollama/                  # Local embedding fallback
└── Firestore/               # Cloud storage for outlines/questions
utils/                       # Caching, data cleaning, PDF/image tools, progress tracking
data_models/                 # Pydantic schemas for courses, questions, documents
data/                        # Sample textbooks, courses.json, caches (gitignored)
  └── textbooks/             # Course PDFs organized by department
specs/                       # Design documents
steering/                    # High-level architecture
tests/                       # Unit/integration tests (PyTest)
```

---

## Setup

### Prerequisites

- **Python 3.10+** (virtualenv recommended)
- **Docker** (for containerized runs) with at least 4 GB RAM and 10 GB free disk space
- **API Keys** for Google Gemini and Cloudflare Workers AI
- **Tesseract OCR** (only for local runs; install via your package manager)
- **AWS CLI** (optional, for ECR deployment)

### Environment Configuration

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Gemini API key(s); comma-separated for multiple keys |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare account ID for embeddings |
| `CLOUDFLARE_API_TOKEN` | Cloudflare API token |
| `TESSDATA_PREFIX` | Path to Tesseract data (local runs) |

### Docker Setup (Recommended)

Use `docker-compose.yml` for containerized runs with **persistent volumes** that survive container rebuilds.

#### Quick Start with Persistent Volumes

```bash
# Build the optimized Docker image
./build.sh

# Start with persistent volumes (recommended)
docker-compose up

# Or run specific service
docker-compose up coursegen-questions

# Generate questions for all courses (20 per subtopic: 10 theory + 10 calculation)
docker-compose run --rm coursegen --theory-per-request 10 --calc-per-request 5

# Generate questions for specific course
docker-compose run --rm coursegen --course-code "AAE 101" --theory-per-request 10 --calc-per-request 5
```

#### Persistent Data Directories

Only cache and course metadata are mounted from the host at runtime:

- `./output_data/cache/` — question generation caches that survive between runs
- `./data/` — course inputs, outlines, and configuration files

> Embeddings live inside the Docker image at `/app/output_data/vector_database`. When you refresh them locally, rebuild (and optionally redeploy) the image so every environment picks up the new bundle.

#### Updating Embeddings

1. **Regenerate embeddings on the host** so `output_data/vector_database` contains the new Chroma database:

   ```bash
   python -m services.RAG.convert_to_embeddings \
     -i data/textbooks/EEE \
     --with-chroma \
     -c pdfs_bge_m3_cloudflare \
     --workers 4 \
     --resume
   ```

2. **Rebuild (and optionally deploy) the Docker image** to bake embeddings into the container:

   ```bash
   ./build.sh --cleanup    # rebuild locally
   ./build.sh --deploy     # push to ECR when ready
   ```

#### Advanced Docker Usage

```bash
# Interactive shell with persistent volumes
docker-compose run --rm coursegen bash

# Custom environment file with volumes
docker-compose --env-file .env.production up

# Run with specific settings
docker-compose run --rm coursegen \
  --course-code "AAE 101" \
  --theory-per-request 10 \
  --calc-per-request 5 \
  --request-delay 2 \
  --temperature 0.7

# Debug mode with verbose output
docker-compose run --rm coursegen \
  --course-code "AAE 101" \
  --theory-per-request 5 \
  --calc-per-request 5 \
  --no-resume \
  --request-delay 2
```

### Local Development Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

Install Tesseract OCR and set `TESSDATA_PREFIX` env var. For OCR-heavy archives, install optional dependencies:

```bash
pip install easyocr      # local OCR fallback
pip install opencv-python  # image preprocessing
```

---

## Pipeline

### Convert to Embeddings

`services/RAG/convert_to_embeddings.py` converts collections of PDFs into searchable vectors with rich metadata, billing information, and resume state. It uses Cloudflare's BGE-M3 embeddings and the CourseGen directory layout (`output_data`), with modular code paths for local experimentation.

#### Processing Flow

1. **Discovery** — recursively walks `--input-dir`, skipping dot directories and non-PDF files.
2. **Text extraction** — attempts direct text via PyMuPDF; delegates to OCR (Gemini, EasyOCR, or hybrid) if insufficient text or `--force-ocr` is set.
3. **Chunking & dedupe** — breaks text into 2-paragraph windows with sentence overlap, then applies SHA1 + fuzzy dedup.
4. **Embedding** — streams batches through Cloudflare's BGE-M3 endpoint with adaptive batch sizes and token accounting; vectors cached per chunk hash.
5. **Export** — writes per-PDF JSONL and optionally upserts batches into Chroma (default).
6. **Progress update** — `progress_state.json` and `seen_files.json` updated after each file for resumability.

#### CLI Usage

```bash
python -m services.RAG.convert_to_embeddings -i <PDF_ROOT> [options]
```

**Frequently Used Flags:**

| Flag | Description |
|------|-------------|
| `-i, --input-dir` | **Required.** Root directory containing PDFs (traversed recursively). |
| `--export-dir` | Where JSONL + progress files live (default `output_data/progress_report`). |
| `--cache-dir` | OCR + text cache root (default `output_data/cache`). |
| `--with-chroma` / `--no-chroma` | Toggle Chroma upserts (default on). |
| `-c, --collection` | Chroma collection name (default `course_embeddings`). |
| `-p, --persist-dir` | Chroma persistence directory (default `output_data/vector_database`). |
| `--workers` | ProcessPool workers for PDF processing (default 2). |
| `--ocr-dpi` | Render DPI when OCR is needed (default 200). |
| `--engine` | OCR engine: `gemini`, `hybrid`, `easyocr` (default `gemini`). |
| `--force-ocr` | Skip native text extraction even if the PDF has a text layer. |
| `--resume` | Always on; delete `progress_state.json` to restart from scratch. |

**Examples:**

```bash
# Standard run with Chroma upserts
python -m services.RAG.convert_to_embeddings \
  -i data/textbooks/EEE \
  --collection pdfs_bge_m3_cloudflare \
  --persist-dir output_data/vector_database \
  --workers 4

# OCR-heavy archive (high DPI, Gemini + EasyOCR hybrid)
python -m services.RAG.convert_to_embeddings \
  -i data/textbooks/scanned \
  --force-ocr \
  --ocr-dpi 450 \
  --engine hybrid \
  --workers 2

# Dry run on a limited subset
python -m services.RAG.convert_to_embeddings \
  -i data/textbooks/sample \
  --max-pdfs 5 \
  --no-chroma
```

#### Output Artifacts

- **Per-PDF JSONL** — `output_data/progress_report/<stem>.jsonl`
- **Chroma** — vectors upserted to target collection
- **Progress** — `progress_state.json` (file status, timing, chunk counts)
- **Billing** — `billing_state.json` (token counts and cost estimates)
- **Dedup index** — `seen_files.json` (SHA-256 prefixes)
- **Cache** — OCR intermediates, text snapshots, per-chunk embedding caches

#### Performance Tuning

- Keep `--workers` low (1–2) for high-DPI OCR to avoid thrashing.
- Start embedding batch size at 16–32; the script adapts based on Cloudflare responses.
- Use `--timeout` (default 30 min) to prevent pathological files from hanging the pool.

#### Verify Embeddings

```bash
python services/RAG/inspect_chroma.py \
  -c pdfs_bge_m3_cloudflare \
  -p output_data/vector_database \
  --query "z-transform"
```

---

### Course Outline Generation

`services/QuestionRag/pipelines/course_outline_generator.py` produces rich course outlines (description + 8–12 modules with 5 learning objectives each) from ChromaDB embeddings. It is the authoritative source for refreshing `courses.json` and exporting per-course outline JSON files.

#### What It Does

- **Chroma-first retrieval** — scans metadata to determine available courses; filters chunks by department, code, and level.
- **Structured prompting** — uses Gemini with deterministic prompts to generate markdown-ready outlines with schema validation.
- **Subtopic refinement** — optional RAG pass refines each module to exactly five comprehensive learning objectives.
- **Resume-friendly orchestration** — caches which courses have outlines, missing embeddings, or errors; TTL support for forced reprocessing.
- **courses.json integration** — updates the central catalog in place (with `.bak` backup).
- **Bulk modes** — scan every unique course folder in Chroma or restrict to a specific department.

#### Components

| Role | Module / Path |
|------|---------------|
| Outline generation core | `services/QuestionRag/pipelines/course_outline_generator.py` |
| RAG retrieval | `services/QuestionRag/utils/chromadb_query.py` |
| Course store | `CourseStore` (reads/writes `courses.json`) |
| Outline cache | `output_data/cache/outline_cache_<DEPT>.json` |
| Progress log | `output_data/cache/outline_progress_<DEPT>.json` |
| Per-course exports | `output_data/cache/outlines_by_chroma/course_outline_<CODE>_<timestamp>.json` |

#### CLI Usage

```bash
python -m services.QuestionRag.pipelines.course_outline_generator [OPTIONS]
```

| Flag | Description |
|------|-------------|
| `--scan_chroma_all` | Enumerate every course present in Chroma (default true). |
| `--department_only` | Restrict to department inferred from `--department_from`. |
| `--department_from` | Seed course code for department prefix (e.g., `"EEE 315"` → `"EEE"`). |
| `--thinking` | Enable Gemini thinking model for richer outlines. |
| `--variation` | Allow retrieval temperature / prompt variation for diverse coverage. |
| `--skip_existing` / `--no_skip_existing` | Skip courses with existing description + outline (default skip). |
| `--allow_dept_fallback` | Fall back to department-level chunks when course embeddings are absent. |
| `--force_regenerate` | Rebuild outlines even if signatures match prior exports. |
| `--dry_run` | Do retrieval, log hit counts, but skip Gemini calls and file writes. |

**Examples:**

```bash
# Refresh every outlined course in Chroma
python -m services.QuestionRag.pipelines.course_outline_generator --scan_chroma_all

# Reprocess a single department with fallback
python -m services.QuestionRag.pipelines.course_outline_generator \
  --department_only \
  --department_from "EEE 315" \
  --allow_dept_fallback

# Force regeneration with thinking mode
python -m services.QuestionRag.pipelines.course_outline_generator \
  --scan_chroma_all \
  --force_regenerate \
  --thinking

# Dry run for retrieval coverage
python -m services.QuestionRag.pipelines.course_outline_generator \
  --department_only \
  --department_from "CVE 201" \
  --dry_run \
  --no_skip_existing
```

#### Programmatic Usage

```python
from services.QuestionRag.pipelines.course_outline_generator import GeminiQuestionGen

gen = GeminiQuestionGen(is_thinking=False)
outline = gen.generate_outline_for_course(
    course_title="Digital Signal Processing",
    course_code="EEE471",
    department_code="EEE",
    level="400",
    department_str_for_prompt="Electrical Engineering",
    variation=True,
    allow_dept_fallback=True,
)
```

#### Operational Notes

- **Subtopic RAG** — controlled by `GEN_QG_SUBTOPIC_RAG` env var (on by default). When enabled, each topic triggers an extra retrieval pass to refine learning objectives.
- **Pacing knobs** — `GEN_QG_COURSE_DELAY_S`, `GEN_QG_TOPIC_DELAY_S`, `GEN_QG_QUERY_DELAY_S` manage throughput.
- **Fallback strategy** — `allow_dept_fallback` prevents gaps when course-specific PDFs are missing.

---

### Question Generation

`services/QuestionRag/pipelines/question_generator.py` produces **20 fully grounded questions per subtopic** (10 theory + two 5-question calculation batches) across every course that has an outline in `courses.json`. It retrieves context from ChromaDB, calls Gemini through the API key balancer, and persists progress for resume safety.

#### Key Components

| Concern | Module |
|---------|--------|
| RAG retrieval | `services/QuestionRag/utils/chromadb_query.py` |
| Prompt assembly | `services/QuestionRag/pipelines/prompt_utils.py` |
| Gemini orchestration | `services/Gemini/gemini_service.py` |
| Batch parsing | `services/QuestionRag/pipelines/json_utils.py` |
| Progress + cache | `services/QuestionRag/utils/course_progress.py`, `cache.py` |
| Topic parallelism | `services/QuestionRag/pipelines/worker_pool.py` |
| Firestore (optional) | `services/Firestore/firebase_service.py` |

#### Architecture

1. **Outline discovery** — loads `courses.json` and filters courses with `outline` blocks.
2. **Per-topic fan out** — `TopicWorkerPool` assigns topics to workers (default 3 threads).
3. **RAG retrieval** — `ChromaQuery` fetches candidates, prunes with similarity thresholds, slices into context windows.
4. **Prompt construction** — assembles request metadata (difficulty, Bloom level, request kind) and RAG snippets.
5. **Gemini call** — `GeminiService` selects a key via `ApiKeyManager`, applies structured output or thinking mode, retries on failure.
6. **Validation + caching** — responses parsed into `Question` Pydantic models, stored in `QuestionCache`, progress counters updated.
7. **Persistence** — Firestore updates after both calculation batches succeed; optional JSONL export.

#### Batch Semantics

Every subtopic always attempts three batches:

1. `theory-1` → 10 MCQs
2. `calculation-1` → 5 calculation MCQs with LaTeX solutions
3. `calculation-2` → another 5 calculation MCQs (total 10 calculations)

#### CLI Usage

```bash
python -m services.QuestionRag.pipelines.question_generator [OPTIONS]
```

| Flag | Purpose / Default |
|------|-------------------|
| `--course-code STR` | Course code (`"all"` processes every outlined course). |
| `--courses-json PATH` | Override course catalog location. |
| `--rag-topk`, `--rag-final-k`, `--rag-tau`, `--rag-min-sim` | Retrieval tuning knobs. |
| `--rag-where JSON` | Extra metadata filter, e.g. `'{"LEVEL": {"$eq": "400"}}'`. |
| `--theory-per-request`, `--calc-per-request` | Targets per Gemini request. |
| `--structured-output` / `--no-structured-output` | Toggle Gemini schema-based parsing. |
| `--thinking`, `--thinking-budget` | Enable Gemini thinking mode. |
| `--request-delay`, `--delay-jitter` | Back pressure between API calls. |
| `--request-attempts`, `--rag-attempts` | Retry counts for Gemini and retrieval. |
| `--topics`, `--subtopics` | Case-insensitive filters. |
| `--output-jsonl PATH` | Dump all questions to a single JSONL file. |
| `--disable-parallel` | Force sequential topic processing. |
| `--skip-firestore`, `--no-resume`, `--no-latex-wrap` | Opt-out flags. |

**Examples:**

```bash
# Default full run (all courses)
python -m services.QuestionRag.pipelines.question_generator

# Single course with structured output
python -m services.QuestionRag.pipelines.question_generator \
  --course-code "EEE 471" \
  --structured-output \
  --request-delay 2.0 \
  --output-jsonl output_data/questions_EEE471.jsonl

# Target specific subtopics with metadata filter
python -m services.QuestionRag.pipelines.question_generator \
  --course-code "MTH 313" \
  --topics "Complex Analysis" \
  --subtopics "Residue Calculus" \
  --rag-where '{"CATEGORY": {"$in": ["TEXTBOOK", "PAST_QUESTIONS"]}}'

# Disable parallel workers (debugging)
python -m services.QuestionRag.pipelines.question_generator --disable-parallel
```

#### Caching & Progress

- **Question cache** — `output_data/cache/question_gen/cache.json` indexes per-request payloads and metadata.
- **Course progress** — `output_data/cache/course_progress/{course}.json` records theory/calculation counters, state, and timestamps.
- **Error dumps** — failed Gemini payloads at `output_data/cache/failed_responses/`.
- **Resume workflow** — interrupted batches marked `in_progress`; next run upgrades them to `interrupted`, clears stale cache entries, and retries.

#### Programmatic Usage

```python
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
    cache_dir=Path("output_data/cache"),
    resume=True,
    store_firestore=False,
)

runner = QuestionBatchRunner(generator)
questions = runner.run_parallel(config)
```

---

## Deployment

### AWS ECR Deployment

Deploy CourseGen to AWS Elastic Container Registry for production use:

```bash
# Fix Docker credential issues (if needed)
./build.sh --fix-credentials

# Build and deploy to AWS ECR
./build.sh --deploy

# Run from ECR with persistent volumes
./run.sh --course-code "EEE 315"

# Manual ECR authentication
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${ECR_REGISTRY}
```

**ECR Repository:** `${ECR_REGISTRY:-coursegen}:latest`

#### Embeddings Management Workflow

```bash
# ONE COMMAND: Update embeddings, rebuild image, and deploy to AWS
./build.sh --update-embeddings

# OR manually (3-step process):
# Step 1: Update local persistent volume embeddings
docker run --rm \
  -v $(pwd)/output_data:/app/output_data \
  -v $(pwd)/data:/app/data \
  ${ECR_REGISTRY:-coursegen}:latest \
  python -m services.RAG.convert_to_embeddings \
  -i data/textbooks/EEE \
  --with-chroma \
  -c pdfs_bge_m3_cloudflare \
  --workers 4 \
  --resume

# Step 2: Rebuild image with updated embeddings
./build.sh --cleanup

# Step 3: Deploy updated image to AWS ECR
./build.sh --deploy
```

#### Host vs Image Embeddings

- **Host directory** (`output_data/vector_database/`): Where regeneration writes during development; rebuild after updating it.
- **Image embeddings**: Copied into the Docker image at `/app/output_data/vector_database` during `./build.sh`.
- **AWS ECR image**: The pushed artifact — rebuild & deploy whenever you refresh embeddings locally.

### EC2 Helper Script

`./ec2_execution.sh` wraps an ECR-hosted container, syncs textbook data + embeddings onto the host, mounts persistent volumes (`~/output_data/...`), and mirrors the Gemini cache so key rotation survives container churn. Use flags like `--course-code`, `--structured-output`, `--background`, `--skip-sync`, or `--env-file` as needed.

### Build Script Features

`./build.sh` includes:

- **System Resource Checks** — validates disk space and Docker daemon status
- **Retry Logic** — automatically retries failed builds with exponential backoff
- **Debug Mode** — detailed system information for troubleshooting
- **Cleanup Options** — removes old images and containers
- **AWS ECR Deployment** — automated push to AWS Elastic Container Registry
- **Credential Helper Fix** — resolves Docker credential helper issues
- **Multiple Build Targets** — support for full and minimal Dockerfiles
- **Health Verification** — validates built images can run successfully

| Flag | Action |
|------|--------|
| `--update-embeddings` | Regenerate embeddings, rebuild image, deploy to ECR (one command) |
| `--cleanup` | Rebuild locally with cleanup |
| `--deploy` | Build and deploy to AWS ECR |
| `--fix-credentials` | Resolve Docker credential helper issues |
| `--debug` | Debug build issues |
| `--minimal` | Build minimal version |
| `--help` | Show all options |

### Run Script Features

`./run.sh` provides:

- **AWS ECR Integration** — automatic authentication and image pulling from ECR
- **Cache Volume Management** — binds cache/data directories needed at runtime
- **Flexible Configuration** — custom environment files and parameters
- **Interactive/Background Modes** — `-i` for interactive, `-b` for detached
- **Smart Prerequisites** — validates Docker image availability

```bash
./run.sh                                          # Show help
./run.sh --course-code "EEE 315"                  # Specific course
./run.sh --theory-per-request 5 --calc-per-request 3  # Custom counts
./run.sh -i --course-code "AAE 101"               # Interactive mode
./run.sh -b --course-code "EEE 315"               # Background mode
./run.sh --debug --course-code "AAE 101"          # Debug mode
./run.sh --env-file .env.production               # Custom env
```

### Docker Compose Commands

```bash
# Start all services
docker-compose up

# Run question generation
docker-compose run --rm coursegen --course-code "EEE 315"

# Update embeddings
docker-compose run --rm \
  -e PYTHONPATH=/app \
  coursegen \
  python -m services.RAG.convert_to_embeddings \
  -i data/textbooks/EEE \
  --with-chroma \
  -c pdfs_bge_m3_cloudflare \
  --workers 4 \
  --resume

# Run tests
docker-compose run --rm coursegen pytest tests/ -v
```

### Production Deployment

**Docker Swarm:**
```bash
docker stack deploy -c docker-compose.yml coursegen
```

**Kubernetes:**
```bash
kompose convert
kubectl apply -f .
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TESSDATA_PREFIX` | — | Path to Tesseract data |
| `OCR_DPI` | 300 | Rendering DPI |
| `OMP_NUM_THREADS` | 4 | OCR threads |
| `CF_EMBED_MAX_BATCH` | 96 | Embedding batch size (≤100) |
| `CF_EMBED_MAX_TOKENS` | 512 | Max tokens per embedding request |
| `BILLING_ENABLED` | 0 | Track costs (1/0) |
| `CF_PRICE_PER_M_TOKENS` | 0.02 | Cloudflare pricing (USD/M tokens) |
| `COURSEGEN_CACHE_ROOT` | `<repo>/output_data` | Base directory for cache files |
| `COURSEGEN_OUTPUT_ROOT` | `<repo>/output_data` | Base directory for output files |
| `COURSEGEN_DEBUG_DUMP_DIR` | — | Directory for failed Gemini payloads |
| `COURSEGEN_DISABLE_CACHE_DAILY_RESET` | false | Skip midnight cache reset |
| `OCR_LANG` | en | OCR language hint |
| `OCR_ENGINE` | gemini | OCR engine (gemini/hybrid/easyocr) |
| `EASYOCR_GPU` | — | Enable GPU for EasyOCR |
| `EMAIL_NOTIFICATIONS_ENABLED` | false | Enable email notifications |

### API Key Load Balancer

`services/Gemini/api_key_manager.py` rotates across multiple Gemini API keys, enforces per-model quotas, and surfaces exhaustion state to upstream pipelines.

#### Core Responsibilities

- **Key discovery** — loads keys from `GeminiApiKeys` (list in `services/Gemini/gemini_api_keys.py`) or from explicit arguments.
- **Persistent usage tracking** — stores daily counters, tokens, and exhaustion flags in `output_data/data/gemini_cache/api_key_cache.json`.
- **Per-model quotas** — enforces rate limits for `flash`, `lite`, `pro`, and `embedding` model families.
- **RPM throttling** — keeps rolling timestamps per key per model to avoid exceeding per-minute limits.
- **Failure handling** — marks keys exhausted on fatal errors, escalates to email notifications, and raises a terminating `RuntimeError` when every key is exhausted.

#### Rate Limits (defaults from `rate_limit_data.py`)

| Model family | Requests per minute | Requests per day |
|--------------|---------------------|------------------|
| `lite` | 15 | 1,000 |
| `flash` | 10 | 250 |
| `pro` | 5 | 25 |
| `embedding` | 5 | 1,000 |

Adjust these in `rate_limit_data.py` if your quotas differ.

#### Configuration Steps

1. **List your keys** — edit `services/Gemini/gemini_api_keys.py` or load from environment variables / secrets manager.
2. **Persist cache directory** — ensure `output_data/data/gemini_cache` is writable and mounted persistently.
3. **(Optional) Disable daily reset** — set `COURSEGEN_DISABLE_CACHE_DAILY_RESET=true` (not recommended for production).

#### Usage

```python
from services.Gemini.gemini_service import GeminiService
service = GeminiService()  # Auto-wires ApiKeyManager + GeminiApiKeys
```

Custom configuration:

```python
from services.Gemini.api_key_manager import ApiKeyManager
from services.Gemini.gemini_service import GeminiService

manager = ApiKeyManager(["API_KEY_1", "API_KEY_2"])
service = GeminiService(api_key_manager=manager, model="gemini-2.5-flash")

response = service.generate("Summarise Fourier series.")
```

#### Inspecting Usage

```bash
jq '.' output_data/data/gemini_cache/api_key_cache.json
```

```python
from services.Gemini.api_key_manager import ApiKeyManager
mgr = ApiKeyManager()
print(mgr.cache_data["keys"])
mgr.rotate_key(model="flash")
```

#### Operational Tips

- Keep at least **twice** as many keys as concurrent workers (e.g., 10 keys for 3–4 workers).
- Rotate compromised keys by editing `api_key_cache.json` or removing them from `GeminiApiKeys`.
- Back up `api_key_cache.json` before large runs for an audit trail.

---

### Data Models

Pydantic-based schemas in `data_models/` ensure type safety, validation, and serialization across the project.

| Model | File | Purpose |
|-------|------|---------|
| `CourseOutline` | `course_outline.py` | Outline structure (modules, objectives, assessments) |
| `Question` | `question_model.py` | Question schema (type, difficulty, answer, sources, Bloom level) |
| `Document` | `document_model.py` | PDF metadata (path, size, hash, tags) |
| `OCRData` | `ocr_data_model.py` | Tesseract outputs (text, confidence, page bounds) |
| `CourseCatalog` | `course_catalog.py` | Catalog from `data/courses.json` |
| `GeminiConfig` | `gemini_config.py` | API configs (keys, models, prompts) |

---

### Services Overview

#### RAG Service (`services/RAG/`)

| Module | Purpose |
|--------|---------|
| `convert_to_embeddings.py` | Core PDF → embedding pipeline |
| `chroma_revive.py` | Initialize/resume Chroma collections; handle schema migrations |
| `chunking.py` | Semantic splitting with configurable overlap |
| `chroma_store.py` | ChromaDB read/write operations |
| `ocr_engine.py` | OCR abstraction (Gemini, EasyOCR, hybrid) |
| `billing.py` | Token counting and cost ledger |
| `log_utils.py` | Structured logging (`setup_logging`, `get_logger`, `snapshot`) |
| `progress_store.py` | JSON-based progress ledgers |
| `cache_utils.py` | Embed/OCR caching by hash |
| `inspect_chroma.py` | CLI for querying collections |
| `path_meta.py` | Metadata extraction from file paths |

#### Firestore Service (`services/Firestore/firebase_service.py`)

Cloud storage for outlines and questions:

```bash
# Setup
export GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# Usage in generators
python -m services.QuestionRag.pipelines.question_generator --with-firestore
```

#### Cloudflare Service (`services/Cloudflare/`)

- `cf_bge_service.py` — BGE-M3 client with batching and retry
- Rate limit: 96 chunks/batch; configure via `CF_EMBED_MAX_BATCH`

#### Ollama Service (`services/Ollama/ollama_service.py`)

Local embedding fallback:
```bash
# Download model
ollama pull bge-m3

# Usage
python -m services.RAG.convert_to_embeddings --embed-provider ollama
```

---

### Utils (`utils/`)

| Module | Purpose |
|--------|---------|
| `utils/Caching/cache.py` | Simple dict/file cache |
| `utils/Caching/enhanced_cache.py` | TTL-based cache for Gemini responses |
| `utils/data_cleaning/convert_and_clean.py` | Normalize text post-OCR |
| `utils/database_transfer/transfer_db.py` | Migrate Chroma collections |
| `utils/ImagesToPDF/main.py` | Convert image folders to PDFs |
| `utils/PdfCompression/main.py` | Compress large PDFs (50-80% reduction) |
| `utils/PdfCompression/upscaling.py` | Enhance low-res scans before OCR |
| `utils/Remove Duplicates/remove_duplicates.py` | Global dedup across JSONL outputs |
| `utils/logging_utils.py` | Console/file logging setup |
| `utils/metadata_extractor.py` | PDF properties extraction |
| `utils/progress_tracker.py` | CLI progress bars (tqdm) |

### Testing

PyTest suite at `tests/`:

```bash
pip install pytest
pytest tests/ -v -q
pytest --cov=services/          # coverage report
python run_ocr_sanity.py path/to/pdf  # single-file OCR test
```

Key test files: `test_batch_utils.py`, `test_chroma_revive.py`, `test_chromadb_query.py`, `test_gemini_question_gen_cache.py`, `test_filter.py`.

Mock external APIs (Gemini/Cloudflare) via `pytest-mock`. Aim for >80% coverage on new code.

---

## Troubleshooting

### Embeddings

| Symptom | Resolution |
|---------|------------|
| "No RAG context found" | Regenerate embeddings locally and rebuild the image |
| "ChromaDB connection failed" | Ensure you rebuilt after uploading the latest SQLite bundle |
| "Permission denied on embeddings" | Make `output_data/vector_database` writable (`chmod`/`chown`) |
| "Embeddings outdated" | Follow two-step refresh: `convert_to_embeddings` → `./build.sh --cleanup` |
| "Disk space full" | Check with `df -h`; embedding databases are large |
| "ChromaDB locked" | Stop any process using the database, then retry |

### Question Generation

| Symptom | Resolution |
|---------|------------|
| "Course code not found" | Check available courses in `data/textbooks/courses.json` |
| "No RAG context found" | Verify embeddings exist for the course code; lower `--rag-min-sim` |
| API errors | Verify API keys in `.env` are valid and have sufficient quota |
| 0 questions generated | Course may lack sufficient RAG context or outlines |
| Resume stuck on a subtopic | Check `output_data/cache/course_progress/`; delete a single file to reset one course |
| Duplicate questions | Delete the relevant cache entry under `output_data/cache/question_gen` and rerun with `--no-resume` |
| Key rotation stalls | Confirm `gemini_api_keys.py` is populated and cache directory is writable |
| Firestore errors | Use `--skip-firestore` to continue locally |
| Validation failures | Inspect dumped payload in `failed_responses/`; use `--structured-output` |
| Memory issues | Reduce `--theory-per-request` and `--calc-per-request`; reduce `--max-topic-workers` |
| Slowdowns | Reduce `--max-topic-workers`, bump `--request-delay`, or filter by `--topics`/`--subtopics` |

### Docker & AWS ECR

| Symptom | Resolution |
|---------|------------|
| "Error saving credentials" | Run `./build.sh --fix-credentials` |
| ECR authentication failed | Check AWS CLI configuration and permissions |
| Image not found locally | Run script auto-pulls from ECR if available |
| Permission denied on volumes | Ensure host directories have 775 permissions |
| Build fails with "invalid tag" | Use correct ECR URI format |
| AWS CLI not found | Install AWS CLI or authenticate manually |
| Out of disk space | `docker system prune -a && ./build.sh --cleanup` |
| Memory issues during build | Increase Docker memory limit in Docker Desktop |
| ChromaDB connection issues | Verify `output_data/vector_database/` exists and has content |

### Dockerfile Optimizations

- **Multi-layer caching** — optimized layer structure for faster incremental builds
- **Network resilience** — automatic retry logic for apt-get operations
- **Security** — non-root user (`appuser`) with proper permissions
- **Health checks** — built-in monitoring and health verification
- **Resource optimization** — configured for optimal memory and CPU usage
- **Dependency resolution** — fixed numpy/albumentations version conflicts
- **Path consistency** — consistent directory paths across all Docker files

---

## Best Practices

- Run question generation nightly so caches stay warm and Firestore progress stays fresh.
- Keep Gemini keys in sync across environments and mount `output_data/data/gemini_cache` in containers.
- Schedule Chroma scans after large ingestion batches so new courses pick up outlines quickly.
- Version control `courses.json` but ignore `output_data` (runtime caches).
- Review samples from each course regularly; calculation questions rely on LaTeX rendering.
- Monitor `output_data/cache/course_progress/*.json` and Firestore dashboards to catch stalled subtopics early.
- Keep prompt templates under `services/QuestionRag/resources` consistent across environments.

## Recent Improvements

- ✅ AWS ECR Deployment — full integration with Elastic Container Registry
- ✅ Cache Volumes — cache/data directories persist while embeddings ship with the image
- ✅ Enhanced Reliability — retry logic for network failures during build
- ✅ Fixed Dependencies — resolved numpy/albumentations version conflicts
- ✅ Better Error Handling — improved build script with debugging capabilities
- ✅ Path Consistency — consistent directory paths throughout
- ✅ Improved Health Checks — container verifies ChromaDB embeddings directory exists
- ✅ Optimized Docker Compose — cleaner configuration with better defaults
- ✅ Docker Credential Helper Fix — automatic resolution of credential issues
- ✅ Question Generation Fix — resolved "missing solution steps" error for calculations
- ✅ Enhanced Scripts — deployment, credential fixing, and debugging options
- ✅ Automated Embeddings Update — one-command workflow to regenerate embeddings and rebuild/deploy
- ✅ Comprehensive Documentation — complete guide for all features and use cases

## Contributing

- Follow PEP 8; add tests for new features.
- Use imperative commit messages (e.g., "Add rotation detection to OCR").
- Report issues at https://github.com/sst/opencode/issues (for tool feedback).
- See `AGENTS.md` for agent-specific instructions.

This project is licensed under MIT.
