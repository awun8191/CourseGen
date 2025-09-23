# CourseGen: Recursive PDF Extraction, RAG, and Automated Course Content Generation

CourseGen is a comprehensive, modular pipeline for processing educational materials (PDFs, images, textbooks) through OCR, embedding, retrieval-augmented generation (RAG), and automated content creation. It supports generating course outlines, interactive questions, and more, with features like resumable processing, multi-provider API integration (Gemini, Cloudflare, Ollama), robust caching, cost tracking, and scalable architecture.

## Key Features
- **End-to-End Pipeline**: From raw PDFs to searchable embeddings and AI-generated educational content.
- **Robust OCR**: Handles scanned documents with preprocessing, rotation detection, and tunable Tesseract parameters.
- **Intelligent RAG**: Semantic search over ChromaDB with metadata filtering (e.g., by course code, department).
- **Automated Generation**: Course outlines and questions using structured Gemini prompts.
- **Load Balancing**: Rotates API keys to handle high-volume requests without rate limits.
- **Observability**: Detailed logging, progress tracking, billing ledgers, and resumability.
- **Modularity**: Separate services for RAG, Question Generation, providers (Gemini/Cloudflare/Ollama/Firestore), and utilities.
- **Scalability**: Parallel processing, batching, and caching for large datasets (e.g., university course libraries).

The project is organized into `services/` (core pipelines), `data_models/` (Pydantic schemas), `utils/` (helpers), `data/` (inputs/outputs, gitignored), and `tests/` (PyTest suite).

## Project Structure Overview
- `services/RAG/`: PDF ingestion, OCR, chunking, embedding, and ChromaDB storage. See [detailed README](README_convert_to_embeddings.md).
- `services/QuestionRag/`: RAG-based course outline and question generation. See [Course Outline README](README_course_outline_generation.md) and [Question Generation README](README_question_generation.md).
- `services/Gemini/`: API client with key load balancing. See [API Key Load Balancer README](README_api_key_load_balancer.md).
- `services/{Cloudflare, Ollama, Firestore}/`: Provider-specific clients.
- `utils/`: Caching, data cleaning, progress tracking, and more.
- `data_models/`: Typed models for courses, questions, documents, etc.
- `data/`: Sample textbooks, courses.json, caches (gitignored).
- `specs/` and `steering/`: Design documents and high-level architecture.
- `tests/`: Unit/integration tests.

## Docker Setup
Use `docker-compose.yml` for containerized runs. It mounts code, data, cache (OUTPUT_DATA2), and Firebase credentials.

Build and run:
```
docker-compose build
docker-compose run --rm coursegen bash  # Interactive shell
```

For question generation:
```
docker-compose run --rm coursegen python -m services.QuestionRag.gemini_question_gen --generate-questions --course-code EEE301
```

Ensure `.env` has API keys, and OUTPUT_DATA2/chroma exists (embeddings).

## Quick Start
1. **Setup Environment**:
   ```
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # or .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```
   Install Tesseract OCR and set `TESSDATA_PREFIX` env var (e.g., `/usr/share/tesseract-ocr/4.00/tessdata` on Linux).

2. **Configure Secrets**:
   - Gemini: `GOOGLE_API_KEY` or multiple keys in `data/gemini_cache/api_key_cache.json`.
   - Cloudflare: `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`.
   - Enable billing: `BILLING_ENABLED=1`.

3. **Process PDFs** (Embeddings):
   ```
   python -m services.RAG.convert_to_embeddings -i data/textbooks/COMPILATION/EEE --export-dir data/exported_data --cache-dir data/ocr_cache --with-chroma -c pdfs_bge_m3_cloudflare -p chromadb_storage --workers 4 --resume --ocr-dpi 450
   ```

4. **Generate Outlines**:
   ```
   python -m services.QuestionRag.pipelines.course_outline_generator --course-code EEE471 --collection pdfs_bge_m3_cloudflare --persist-dir chromadb_storage --output-dir utils/course_outline
   ```

5. **Generate Questions** (30/subtopic: 20 theory + 10 calc, RAG-required, Firestore-persisted):
    ```
    python -m services.QuestionRag.gemini_question_gen --generate-questions --course-code EEE471  # Single
    python -m services.QuestionRag.gemini_question_gen --generate-questions  # All courses
    ```
    - Skips subtopics without docs; only courses with outlines.
    - Resumes from cache (OUTPUT_DATA2/cache).
    - LaTeX for calc steps.
    - Full CLI: `--theory-per-request 10 --calc-per-request 5 --no-resume --skip-firestore` etc.

6. **Run Tests**:
   ```
   pytest tests/ -v
   ```

## Detailed Documentation
For in-depth guides:
- [Course Outline Generation](README_course_outline_generation.md): Automating syllabi from RAG-retrieved content.
- [Question Generation](README_question_generation.md): Creating MCQs, essays, and more aligned with outlines.
- [API Key Load Balancer](README_api_key_load_balancer.md): Scaling Gemini requests across multiple keys.
- [Convert to Embeddings Pipeline](README_convert_to_embeddings.md): Core PDF processing and vectorization.
- [Other Components](README_other_components.md): Data models, utils, testing, and deployment.

## Environment Variables
- `TESSDATA_PREFIX`: Path to Tesseract data.
- `OCR_DPI`: Rendering DPI (default 300).
- `OMP_NUM_THREADS`: OCR threads (default 4).
- `CF_EMBED_MAX_BATCH`: Embedding batch size (≤100).
- `BILLING_ENABLED`: Track costs (1/0).
- `CF_PRICE_PER_M_TOKENS`: Cloudflare pricing (default 0.02 USD/M tokens).

## Contributing
- Follow PEP 8; add tests for new features.
- Use imperative commit messages (e.g., "Add rotation detection to OCR").
- Report issues at https://github.com/sst/opencode/issues (for tool feedback).
- For help: Run `/help` in opencode.

See `AGENTS.md` for agent-specific instructions. This project is licensed under MIT.