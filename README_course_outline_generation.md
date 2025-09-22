# Course Outline Generation in CourseGen

This module automates the creation of detailed, structured course outlines (syllabi) from processed educational documents. It uses retrieval-augmented generation (RAG) to query relevant content from ChromaDB embeddings and synthesizes hierarchical outlines using Gemini AI. Outlines include topics, subtopics, learning objectives, assessments, prerequisites, and references, tailored to specific courses (e.g., EEE 471 - Digital Signal Processing).

## Overview
Course outlines are essential for educators but time-consuming to create manually, especially from legacy scanned PDFs. This pipeline:
- Retrieves semantically relevant chunks filtered by metadata (e.g., course code, department, level).
- Applies structured prompts to Gemini for coherent, markdown-formatted outputs.
- Validates against Pydantic schemas (`course_model.py`, `course_outline.py`).
- Supports batch generation for multiple courses from `data/courses.json`.
- Integrates caching and load balancing for efficient, scalable processing.

Key benefits:
- Reduces manual effort by 80-90% for syllabus creation.
- Ensures alignment with source materials via RAG.
- Customizable for different disciplines (e.g., engineering, humanities) via prompt templates.
- Resumable: Skips completed courses using progress trackers.

## Architecture
1. **Input**: Embeddings in ChromaDB (from `convert_to_embeddings.py`), course metadata from `data/courses.json` or CLI args.
2. **Retrieval**: `services/QuestionRag/utils/chromadb_query.py` performs hybrid search (semantic + keyword) with filters (e.g., `metadata['COURSE_CODE'] == 'EEE471'`).
3. **Generation**: `services/QuestionRag/pipelines/course_outline_generator.py`:
   - Loads prompts from `services/QuestionRag/resources/prompts_semiconductor-materials-and-properties.jsonl` (or custom).
   - Calls Gemini via `gemini_service.py` with balanced keys.
   - Structures output: Overview > Modules (Topics > Objectives > Activities) > Assessments > References.
4. **Output**: Markdown files in `utils/course_outline/` (e.g., `course_outline_EEE471_20250922.md`), plus JSON for integration.
5. **Caching**: Responses cached in `data/gemini_cache/`; metadata in `progress_store.py`.
6. **Validation**: Uses `course_outline.py` to ensure completeness (e.g., all modules have objectives).

Dependencies:
- ChromaDB collection (e.g., `pdfs_bge_m3_cloudflare`).
- Gemini API keys.
- Optional: Firestore for storing outlines (`firebase_service.py`).

## Usage
### Prerequisites
- Run embeddings pipeline first: See [Convert to Embeddings README](README_convert_to_embeddings.md).
- Configure Gemini: Set `GOOGLE_API_KEY` or use load balancer.

### CLI Command
```
python -m services.QuestionRag.pipelines.course_outline_generator [OPTIONS]
```

#### Key Options
- `--course-code STR`: Target course (e.g., "EEE471"). Required unless `--input-courses` used.
- `--input-courses PATH`: JSON file with course list (e.g., `data/courses.json`). Batch mode.
- `--collection STR`: Chroma collection name (default: "pdfs_bge_m3_cloudflare").
- `--persist-dir PATH`: ChromaDB path (default: "chromadb_storage").
- `--output-dir PATH`: Save outlines (default: "utils/course_outline").
- `--top-k INT`: Retrieval chunks (default: 50; higher for comprehensive outlines).
- `--prompt-file PATH`: Custom prompt JSONL (default: resources/prompts_...jsonl).
- `--temperature FLOAT`: Gemini creativity (0.0-1.0; default 0.3 for structured output).
- `--max-tokens INT`: Output length limit (default 4000).
- `--workers INT`: Parallel generations (default 1; use with load balancer).
- `--resume`: Skip completed courses.
- `--with-firestore`: Upload to Firestore (requires config).
- `--verbosity LEVEL`: Logging (debug/info/warn/error).

#### Single Course Example
```
python -m services.QuestionRag.pipelines.course_outline_generator \
  --course-code EEE471 \
  --collection pdfs_bge_m3_cloudflare \
  --persist-dir chromadb_storage \
  --output-dir utils/course_outline \
  --top-k 75 \
  --temperature 0.2
```
Output: `utils/course_outline/course_outline_EEE471_YYYYMMDD_HHMMSS.md`

#### Batch Example (Multiple Courses)
Populate `data/courses_batch.json`:
```json
[
  {"code": "EEE471", "department": "EEE", "level": "400"},
  {"code": "MTH313", "department": "MTH", "level": "300"}
]
```
```
python -m services.QuestionRag.pipelines.course_outline_generator \
  --input-courses data/courses_batch.json \
  --workers 2 \
  --resume
```

### Programmatic Usage
```python
from services.QuestionRag.pipelines.course_outline_generator import generate_outline

outline = generate_outline(
    course_code="EEE471",
    collection="pdfs_bge_m3_cloudflare",
    persist_dir="chromadb_storage",
    top_k=50
)
with open("outline.md", "w") as f:
    f.write(outline)
```

## Prompt Engineering
Prompts are JSONL files with templates like:
```jsonl
{"role": "system", "content": "You are an expert curriculum designer. Generate a detailed outline for {course_code} based on the provided chunks. Structure: # Title\n## Overview\n### Module 1: ...\n- Topics\n- Objectives\nEnsure alignment with engineering standards."}
{"role": "user", "content": "Chunks: {retrieved_chunks}\nGenerate outline."}
```
- Customize for domains: Add Bloom's taxonomy levels, duration estimates, or rubrics.
- Test prompts: Use `test_gemini_question_gen_cache.py` adapted for outlines.

## Output Format
Markdown structure:
```
# Course Outline: {COURSE_CODE} - {Title}

## Course Information
- **Code**: EEE471
- **Department**: EEE
- **Level**: 400
- **Credits**: 3
- **Prerequisites**: EEE313

## Overview
{Description from RAG synthesis}

## Learning Modules
### Module 1: Introduction to DSP (Weeks 1-3)
- **Topics**: Signals, systems, Fourier analysis.
- **Learning Objectives**:
  - LO1: Define discrete-time signals (Bloom: Remember).
  - LO2: Apply z-transforms to LTI systems (Bloom: Apply).
- **Activities**: Lectures, MATLAB labs.
- **Assessments**: Quiz 1 (10%).

### Module 2: ...
...

## Assessments
- Midterm: 30% (Topics 1-4)
- Final Exam: 40%
- Assignments: 20%
- Participation: 10%

## References
- "Digital Signal Processing" by Proakis (EEE471_textbook.pdf, pages 1-50).
- Lecture notes (EEE471_lectures.pdf).
```
- JSON export: Includes parsed sections for UI integration.

## Integration
- **With Question Generation**: Pipe outlines to `--input-outline` for targeted questions.
- **With Firestore**: Store for web apps (`firebase_service.py`).
- **With Courses Catalog**: Auto-generate from `data/courses.json` using `utils/courses.py`.
- **Customization Hooks**: Override `get_retrieval_query(course)` in `course_outline_generator.py` for advanced filtering.

## Testing and Validation
- Unit Tests: `pytest tests/test_gemini_question_gen_cache.py` (adapt for outlines).
- Integration: `test_chromadb_query.py` verifies retrieval.
- Manual: Compare generated outline to source PDFs; check for hallucinations (low temperature helps).
- Metrics: Coverage (e.g., % of modules with objectives), coherence score via secondary Gemini call.

## Troubleshooting
- **Poor Retrieval**: Increase `--top-k` or refine metadata filters in `path_meta.py`.
- **Gemini Rate Limits**: Use load balancer; monitor `data/gemini_cache/api_key_cache.json`.
- **Incomplete Outlines**: Adjust prompt for more structure; increase `--max-tokens`.
- **Chroma Errors**: Ensure embeddings exist (`inspect_chroma.py`); check scalar metadata.
- **Caching Issues**: Delete `data/gemini_cache/` and rerun with `--no-cache`.
- **Logs**: Set `LOG_LEVEL=DEBUG` for query traces.

## Best Practices
- Start with small `--top-k` (20) for testing, scale to 100+ for production.
- Curate `data/courses.json` with accurate metadata for better filtering.
- Review outputs manually initially; fine-tune prompts iteratively.
- For large batches, monitor costs via billing ledger.
- Version outlines with timestamps; use Git for tracking changes.

## Future Enhancements
- Multi-language support (add tessdata for non-English).
- Integration with LMS (e.g., Moodle export).
- Auto-grading alignment for generated questions.
- Visual diagrams in outlines (via PlantUML or Mermaid).

This module transforms raw documents into structured educational blueprints, enabling rapid course development.