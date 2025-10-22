# Course Outline Generation Pipeline

`services/QuestionRag/pipelines/course_outline_generator.py` produces rich course outlines (description + 8–12 modules with 5 learning objectives each) from the embeddings stored in ChromaDB. It is the authoritative source for refreshing `courses.json` and for exporting per-course outline JSON files that downstream systems can consume.

## What It Does
- **Chroma-first retrieval** – scans metadata stored with embeddings to determine which courses are available and filters chunks by department, course code, and level.
- **Structured prompting** – uses Gemini with deterministic prompts to generate markdown-ready outlines and validates the output with light schema checks.
- **Subtopic refinement** – optional RAG pass refines each module to exactly five comprehensive learning objectives using additional embedding context.
- **Resume-friendly orchestration** – caches which courses already have outlines, which are missing embeddings, and which encountered errors. Cache entries can have TTLs to force later reprocessing.
- **courses.json integration** – updates the central catalog in place (with `.bak` backup) so question generation and other services can rely on up-to-date outlines.
- **Bulk modes** – supports scanning every unique course folder present in Chroma or restricting work to a specific department.

## Components & Files
| Role | Module / Path |
| --- | --- |
| Outline generation core | `GeminiQuestionGen` (same file) |
| RAG retrieval | `ChromaQuery` & `MetaData` (`services/QuestionRag/utils/chromadb_query.py`) |
| Course store | `CourseStore` (reads/writes `courses.json`, keeps `.bak`) |
| Missing/present cache | `OutlineCache` → `OUTPUT_DATA2/cache/outline_cache_<DEPT>.json` |
| Progress log | `OutlineProgress` → `OUTPUT_DATA2/cache/outline_progress_<DEPT>.json` |
| Chroma-wide resume | `ChromaCourseProgress` → `CHROMA_OUT_DIR/chroma_progress.json` |
| Per-course exports | `CHROMA_OUT_DIR/course_outline_<CODE>_<timestamp>.json` (default `OUTPUT_DATA2/cache/outlines_by_chroma`) |

## Execution Flow
1. **Enumerate courses** – either by department code (`DepartmentRunner`) or via unique `COURSE_FOLDER` values found in Chroma (`ChromaCoursesRunner`).
2. **Check caches** – skip courses that already have outline+description unless `--force-regenerate` is provided. Missing caches respect TTLs to avoid hammering empty datasets.
3. **Retrieve context** – `ChromaQuery` runs hybrid semantic + lexical search and collects source IDs. When `allow_dept_fallback` is set, department-level material is used when course-specific hits are absent.
4. **Generate outline** – prompts Gemini (thinking mode optional) and enforces a consistent structure (description, modules, sources). Subtopics are refined with extra RAG queries if enabled.
5. **Persist** – writes back to `courses.json`, stores per-course JSON in `CHROMA_OUT_DIR`, and updates cache/progress manifests.
6. **Delay** – pacing controls (`COURSE_DELAY_S`, `TOPIC_DELAY_S`) prevent overwhelming Gemini or Chroma.

## Prerequisites
- Up-to-date embeddings in ChromaDB (`convert_to_embeddings` pipeline) with metadata fields such as `DEPARTMENT`, `COURSE_FOLDER`, `LEVEL`, etc.
- Gemini API keys configured (see `README_api_key_load_balancer.md`).
- `courses.json` with basic course metadata (code, title, level) so generated outlines can be written back.
- Optional: Firestore/other services if you extend the pipeline; current script writes locally only.

## CLI Usage
```bash
python -m services.QuestionRag.pipelines.course_outline_generator [OPTIONS]
```

| Flag | Description |
| --- | --- |
| `--scan_chroma_all / --no-scan_chroma_all` | Enumerate every course present in Chroma (default true). |
| `--department_only` | Restrict processing to the department inferred from `--department_from`. |
| `--department_from` | Seed course code used to derive the department prefix (e.g. `"EEE 315"` → `"EEE"`). |
| `--courses_json` | Path to `courses.json` (defaults through `config.load_config()`). |
| `--thinking` | Enable Gemini thinking model for richer outlines. |
| `--variation` | Allow retrieval temperature / prompt variation for more diverse module coverage (default true). |
| `--skip_existing` / `--no_skip_existing` | Skip courses that already have description + outline (default skip). |
| `--allow_dept_fallback` | When a course lacks embeddings, fall back to department-level chunks instead of marking missing. |
| `--missing_ttl_hours` | Expiration (hours) for missing-course cache markers when re-running departments. |
| `--only_missing` | Process only courses currently flagged as missing (honours TTL). |
| `--ignore_missing_cache` | Ignore cached missing flags and try again immediately. |
| `--save_each_write` | Persist `courses.json`, cache, and progress after each course (default true). |
| `--force_regenerate` | Rebuild outlines even if signatures match prior exports (Chroma scan mode). |
| `--output_dir` | Directory for per-course JSON exports during Chroma scan (default `CHROMA_OUT_DIR`). |
| `--dry_run` | Do retrieval, log hit counts, but do not call Gemini or write files. |

### Command Examples
- **Refresh every outlined course in Chroma (default mode):**
  ```bash
  python -m services.QuestionRag.pipelines.course_outline_generator --scan_chroma_all
  ```
- **Reprocess a single department with fallback to department-level embeddings:**
  ```bash
  python -m services.QuestionRag.pipelines.course_outline_generator \
    --department_only \
    --department_from "EEE 315" \
    --allow_dept_fallback \
    --missing_ttl_hours 24
  ```
- **Force regeneration for all courses regardless of cache:**
  ```bash
  python -m services.QuestionRag.pipelines.course_outline_generator \
    --scan_chroma_all \
    --force_regenerate \
    --variation \
    --thinking
  ```
- **Dry run to inspect retrieval coverage for a department:**
  ```bash
  python -m services.QuestionRag.pipelines.course_outline_generator \
    --department_only \
    --department_from "CVE 201" \
    --dry_run \
    --no_skip_existing
  ```

## Output Structure
- **courses.json** – updated in place with `description`, `outline`, and flattened `outline_sources` (a sorted list of chunk IDs).
- **Outline cache** – `OUTPUT_DATA2/cache/outline_cache_<DEPT>.json` captures which courses have embeddings (`present`) or are missing (`missing`), including timestamps.
- **Progress log** – `OUTPUT_DATA2/cache/outline_progress_<DEPT>.json` tracks status per course (`present`, `missing`, `error`, etc.) for visibility.
- **Per-course JSON export** – under `CHROMA_OUT_DIR` (default `OUTPUT_DATA2/cache/outlines_by_chroma`), each outline is saved as `course_outline_<CODE>_<timestamp>.json` for external tooling.
- **Backups** – the first run in a process writes `courses.json.bak` before modifying the file.

### Sample Outline JSON
```json
{
  "course": "EEE471",
  "description": "Advanced DSP topics with emphasis on z-Transforms and filter design.",
  "modules": [
    {
      "title": "Digital Signal Fundamentals",
      "learning_objectives": [
        "Explain discrete-time signal representations",
        "Compare energy and power signals",
        "Analyse sampling effects",
        "Apply aliasing mitigation techniques",
        "Relate time and frequency domain descriptions"
      ],
      "sources": ["EEE/400/1/EEE471/EEE471_textbook.pdf#chunk_42"]
    }
  ],
  "sources": ["EEE/400/1/EEE471/EEE471_textbook.pdf#chunk_42", "..."]
}
```

## Programmatic Usage
```python
from pathlib import Path
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

if outline:
    # Persist or hand off to other services
    print(outline["description"])
```

## Operational Notes
- **Subtopic RAG** – controlled by `GEN_QG_SUBTOPIC_RAG` environment variable (on by default). When enabled, each topic triggers an extra retrieval pass to refine learning objectives.
- **Chroma signatures** – `ChromaCoursesRunner` stores a signature (hashes, mtime counts) per course. If nothing changed, `--force_regenerate` is required to rebuild the outline.
- **Pacing knobs** – `GEN_QG_COURSE_DELAY_S`, `GEN_QG_TOPIC_DELAY_S`, and `GEN_QG_QUERY_DELAY_S` manage throughput; values are pulled from `config.py` or env vars.
- **Thinking mode** – `--thinking` swaps in the designated thinking model and uses `GEN_QG_THINK_BUDGET` tokens per request.
- **Fallback strategy** – `allow_dept_fallback` is useful when course-specific PDFs are missing; it prevents gaps in `courses.json` while you work on ingestion gaps.

## Troubleshooting
- **“No embeddings” logs** – the course folder has no vectors. Run the embeddings pipeline or use `--allow_dept_fallback`.
- **courses.json not updating** – ensure the script can write to the file; the process creates `.bak` and `.tmp` files next to it.
- **Outline cache stuck** – delete `OUTPUT_DATA2/cache/outline_cache_<DEPT>.json` to clear state or use `--ignore_missing_cache`.
- **Slow Chroma scans** – limit scope by department or by pre-filtering embeddings. The scan enumerates all metadata entries which can be large.
- **Prompt drift** – use `--variation` to re-randomize when outlines feel repetitive; disable it for more deterministic results.

## Best Practices
- Schedule Chroma scans after large ingestion batches so new courses pick up outlines quickly.
- Version control `courses.json` but ignore `OUTPUT_DATA2`—it contains runtime caches and should be mounted as a volume in production.
- Monitor cache files; a high number of `missing` entries indicates upstream ingestion gaps.
- Keep prompt templates under `services/QuestionRag/resources` consistent across environments for reproducibility.

The outline generator is the bridge between raw course materials and downstream artefacts (question generation, syllabi export, etc.). Keep the caches healthy, ensure embeddings stay fresh, and the pipeline will maintain accurate, classroom-ready outlines.
