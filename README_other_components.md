# Other Components in CourseGen

This README covers the remaining modules, utilities, data models, testing, deployment, and miscellaneous features that support the core pipelines (RAG, outlines, questions, embeddings). These components provide modularity, reliability, and extensibility for building full educational applications.

## Data Models (`data_models/`)
Pydantic-based schemas ensure type safety, validation, and serialization across the project. All models support JSON/MD export and integrate with Firestore/Chroma.

### Key Models
- **`course_model.py` & `course_outline.py`**: Course info (code, department, prerequisites) and outline structure (modules, objectives, assessments). Validates completeness (e.g., every module has LOs).
  Example:
  ```python
  from data_models.course_outline import CourseOutline
  outline = CourseOutline.parse_file("outline.md")
  assert len(outline.modules) > 0
  ```
- **`question_model.py`**: Question schema (type, difficulty, answer, sources). Supports Bloom levels and multi-option MCQs.
- **`document_model.py` & `file_data_model.py`**: PDF metadata (path, size, hash, tags like DEPARTMENT/LEVEL).
- **`ocr_data_model.py` & `ocr_response_model.py`**: Tesseract outputs (text, confidence, page bounds).
- **`course_catalog.py` & `course_embedding.py`**: Catalog from `data/courses.json`; embedding metadata.
- **`gemini_config.py`**: API configs (keys, models, prompts).

Usage: Import and validate inputs/outputs, e.g., `Question.parse_raw(json_str)`.

Customization: Extend with `BaseModel` for new fields (e.g., video links in outlines).

## Services
### RAG Service (`services/RAG/`)
Beyond `convert_to_embeddings.py`:
- **`chroma_revive.py`**: Initializes/resumes Chroma collections; handles schema migrations.
  Usage: `python services/RAG/chroma_revive.py -p chromadb_storage -c pdfs_bge_m3_cloudflare --reset` (caution: wipes data).
- **`chunking.py`**: Semantic splitting; configurable overlap/min-size.
- **`helpers.py` & `path_meta.py`**: Utility functions for tagging, hashing.
- **`log_utils.py`**: Structured logging (`setup_logging()`, `get_logger()`, `snapshot()` for progress).
- **`progress_store.py`**: JSON-based ledgers for resumability.
- **`cache_utils.py`**: Embed/OCR caching by hash.
- **`billing.py`**: Token counting and cost ledger; supports rebasing.
- **`inspect_chroma.py`**: CLI for querying collections (e.g., `--query "DSP" --top-k 5`).

### Firestore Service (`services/Firestore/firebase_service.py`)
Cloud storage for outlines/questions:
- Upload/download: `upload_outline(course_code, outline_md)`.
- Queries: Filter by tags (e.g., department="EEE").
- Setup: `GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json`.
- Usage: `--with-firestore` in generators.

### Ollama Service (`services/Ollama/ollama_service.py`)
Local embeddings/retrieval fallback:
- Models: Download `bge-m3` via Ollama.
- Usage: `--embed-provider ollama`; no cloud costs.
- Pros: Offline, fast for small batches; Cons: Lower quality than Cloudflare.

### Cloudflare Service (`services/Cloudflare/`)
- **`cf_bge_service.py`**: BGE-M3 client with batching/retry.
- **`cloudflare_service.py`**: General Workers AI helpers.
- Limits: 96 chunks/batch; monitor via env `CF_EMBED_MAX_BATCH`.

### QuestionRag Utils (`services/QuestionRag/utils/`)
- **`batch_utils.py`**: Parallel processing for generators (e.g., multi-course batches).
- **`chromadb_query.py`**: Hybrid search (BM25 + cosine); metadata filters.
- **`cloudflare_vectorize.py`**: Embed utils.
- **`courses.py`**: Load/parse `data/courses.json`.
- **`snitze.py`**: Typo? Likely `sanitize.py` for text cleaning.

Tests: `tests/test_batch_utils.py`, `test_chromadb_query.py`, `test_filter.py`.

## Utils (`utils/`)
Reusable helpers for data handling and workflows.

### Caching (`utils/Caching/`)
- **`cache.py`**: Simple dict/file cache.
- **`enhanced_cache.py`**: TTL, serialization for Gemini responses (avoids re-prompting identical chunks).

### Course Outline (`utils/course_outline/`)
- Generated MD files (e.g., `course_outline_EEE427_1_20250613_000915.md`).
- `rag_py.py`: Experimental RAG script.

### Data Cleaning (`utils/data_cleaning/`)
- **`convert_and_clean.py`**: Normalize text post-OCR (remove headers/footers).

### Database Transfer (`utils/database_transfer/`)
- **`transfer_db.py`**: Migrate Chroma collections (e.g., local to cloud).

### Images to PDF (`utils/ImagesToPDF/`)
- **`main.py`**: Convert image folders (e.g., WhatsApp scans) to PDFs for processing.
  Usage: `python utils/ImagesToPDF/main.py -i data/textbooks/PQ -o data/converted_pdfs`.

### PDF Compression (`utils/PdfCompression/`)
- **`main.py`**: Compress large PDFs (reduce size 50-80%).
- **`upscaling.py`**: Enhance low-res scans before OCR.

### Remove Duplicates (`utils/Remove Duplicates/`)
- **`remove_duplicates.py`**: Global dedup across JSONL outputs.

### General Utils
- **`logging_utils.py`**: Console/file logging setup.
- **`metadata_extractor.py`**: PDF properties extraction.
- **`progress_tracker.py`**: CLI progress bars (tqdm integration).

## Data and Specs
- **`data/textbooks/`**: Sample inputs (EEE/MTH PDFs, PPTX, images like WhatsApp scans).
- **`data/courses.json`**: Catalog {"code": "EEE471", "topics": [...], "department": "EEE"}. Backups: `.bak`, `courses_legacy.json`.
- **`data/gemini_cache/`**: API responses, `api_key_cache.json`.
- **`specs/interactive-course-question-generator/`**: Design docs (`design.md`, `requirements.md`, `tasks.md`).
- **`steering/`**: High-level (`product.md`, `structure.md`, `tech.md`).
- Outputs: `outputs.jsonl` (generated content), `chromadb_storage/` (gitignored DB).

## Testing (`tests/`)
PyTest suite for reliability:
- Run: `pip install pytest; pytest tests/ -v -q`.
- Coverage: `pytest --cov=services/`.
- Key Tests:
  - `test_batch_utils.py`: Parallel processing.
  - `test_chroma_revive.py`: DB init.
  - `test_gemini_question_gen_cache.py`: Caching.
  - `test_chromadb_query.py`: Retrieval.
  - `test_filter.py`: Metadata filtering.
- Mocking: External APIs (Gemini/Cloudflare) via `pytest-mock`.
- Add: One test per new function; aim >80% coverage.

OCR Sanity: `python run_ocr_sanity.py path/to/pdf` (single-file test).

## Deployment and Docker
- **Dockerfile**: Builds container with Tesseract, deps.
  Build: `docker build -t coursegen .`
  Run: `docker run -v $(pwd)/data:/app/data -e TESSDATA_PREFIX=/usr/share/tessdata coursegen python -m services.RAG.convert_to_embeddings -i /app/data/textbooks ...`
- **Config**: `config.py` for paths; override via env.
- **Cloud**: Deploy pipelines to GCP/Cloud Run; use Firestore for state.
- **CI/CD**: GitHub Actions for tests/lint on push.

## Batch Processing (`batch_processing.py`)
Utility for running pipelines in sequence:
```
python batch_processing.py --pipeline all --input data/textbooks --courses data/courses.json
```
Automates: Embed > Outline > Questions.

## Environment and Config
- **Python 3.10+**: Virtualenv recommended.
- **Env Vars**: See main README; secrets in `.env` (dotenv support).
- **Requirements**: `requirements.txt` (PyMuPDF, chromadb, google-generativeai, etc.).
- **Gitignore**: Caches, DBs, logs, secrets.
- **AGENTS.md**: Opencode tool instructions.

## Contributing and Maintenance
- **Style**: PEP 8, black formatter (`pip install black; black .`).
- **Commits**: Semantic (feat/fix/docs); <72 char subjects.
- **PRs**: Describe changes, add tests, update docs.
- **Issues**: Track bugs/features; link to specs.
- **Feedback**: https://github.com/sst/opencode/issues.

## Miscellaneous
- **`prompts.jsonl`**: Global prompt templates.
- **`~bashrc`**: Shell aliases? (Likely stray file).
- **Legacy**: `courses_fallback_with_topics.json.bak` for rollbacks.

These components glue the project together, enabling a full-stack educational AI workflow. For specifics, refer to inline docstrings and tests.