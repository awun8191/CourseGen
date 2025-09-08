# Recursive PDF Extraction and RAG

This repository provides an end-to-end, resumable pipeline that converts folders of PDFs into searchable embeddings with robust OCR, caching, and vector database ingestion. It emphasizes observability (logs, progress ledgers), cost tracking, and predictable metadata.

## Project Structure
- `src/services/RAG`: Core indexing pipeline (OCR, chunking, embeddings, Chroma upsert)
- `src/services/Gemini`: Gemini API client and key management
- `src/services/Ollama`: Ollama embeddings client for retrieval
- `src/services/Cloudflare`: Cloudflare BGE-M3 client helpers
- `src/services/Firestore`: Firestore utilities
- `src/utils`: Logging, caching, data cleaning, progress utilities
- `src/data_models`: Pydantic models for configs and structured outputs
- `scripts/`: Helper/utility scripts (e.g., OCR debug, ingestion)
- `data/`: Example inputs/outputs, export/cache folders (gitignored)
- `chromadb_storage/`: Chroma persistence directory (gitignored)

High-level flow (per PDF):
- Detect whether the PDF has an extractable text layer; if not, OCR the pages.
- OCR quality helpers: orientation detection, OpenCV preprocessing, tunable Tesseract options.
- Chunk text (paragraph-aware), deduplicate within the file, and compute embeddings in batches.
- Write a per-file JSONL (text + metadata + embedding).
- Upsert vectors into a persistent Chroma collection.
- Track token usage and cost with a local billing ledger; optionally rebase historical costs to a new price.
- Resume safely after interrupts without redoing completed work.


## Requirements (Windows)
- Python 3.10+
- Tesseract OCR for Windows
  - Binary: `C:\Program Files\Tesseract-OCR\tesseract.exe`
  - Language data: `C:\Program Files\Tesseract-OCR\tessdata\eng.traineddata`
  - Env: `TESSDATA_PREFIX = C:\Program Files\Tesseract-OCR\tessdata`
- Optional: OpenCV (installed via requirements) for preprocessing
- Cloudflare Workers AI credentials (for embeddings) or your configured provider


## Quick start (PowerShell)
Set environment for this session (no secrets below):

```
$env:TESSDATA_PREFIX = "C:\Program Files\Tesseract-OCR\tessdata"
$env:OMP_NUM_THREADS = "4"
$env:CF_EMBED_MAX_BATCH = "96"
$env:BILLING_ENABLED = "1"
$env:CF_PRICE_PER_M_TOKENS = "0.02"     # default in code; override if needed
$env:OCR_DPI = "450"                      # higher DPI often improves OCR quality
```

Optionally persist Tesseract data path:

```
setx TESSDATA_PREFIX "C:\Program Files\Tesseract-OCR\tessdata"
```

Run the pipeline (resumable):

```
& ".\.venv\Scripts\python.exe" -m src.services.RAG.convert_to_embeddings `
  -i "C:\Users\<you>\Documents\SCHOOL\COMPILATION\EEE" `
  --export-dir "C:\Users\<you>\Documents\Recursive-PDF-EXTRACTION-AND-RAG\data\exported_data" `
  --cache-dir  "C:\Users\<you>\Documents\Recursive-PDF-EXTRACTION-AND-RAG\data\ocr_cache" `
  --workers 6 `
  --omp-threads 4 `
  --resume `
  --with-chroma `
  --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe" `
  --ocr-on-missing fallback `
  --ocr-rotate `
  --ocr-preprocess `
  --ocr-psm 6 `
  --ocr-oem 1 `
  -c pdfs_bge_m3_cloudflare `
  -p "C:\Users\<you>\Documents\Recursive-PDF-EXTRACTION-AND-RAG\chroma_db_bge_m3"
```

Set Cloudflare credentials (recommended for production):

```
$env:CLOUDFLARE_ACCOUNT_ID = "{{CLOUDFLARE_ACCOUNT_ID}}"
$env:CLOUDFLARE_API_TOKEN  = "{{CLOUDFLARE_API_TOKEN}}"
```


## CLI reference (convert_to_embeddings)
- `-i, --input-dir` (str): Root folder with PDFs (recursively processed)
- `--export-dir` (str): Output for per-file JSONL and `progress_state.json`
- `--cache-dir` (str): Local OCR/embedding caches
- `-c, --collection` (str): Chroma collection name
- `-p, --persist-dir` (str): Chroma persistence path
- `--workers` (int): Parallel processes for file-level parallelism
- `--omp-threads` (int): Threads used by OCR libraries (OpenMP)
- `--resume` (flag): Resume from progress ledger; skip completed files
- `--with-chroma` (flag): Upsert to Chroma
- `--force-ocr` (flag): Force OCR even if a text layer exists

OCR options
- `--tesseract-cmd` (str): Full path to `tesseract.exe`
- `--ocr-on-missing` (fallback|error|skip)
- `--ocr-dpi` (int): Render DPI (300/450/600)
- `--ocr-psm` (int): Page segmentation mode; `6` is a strong default
- `--ocr-oem` (int): Engine mode; `1` (LSTM) usually best
- `--ocr-extra-config` (str): Additional `-c key=value` pairs
- `--ocr-rotate` (flag): Orientation/rotation detection
- `--ocr-preprocess` (flag): OpenCV-based denoise/threshold/sharpen


## Environment variables
Cloudflare Workers AI
- `CLOUDFLARE_ACCOUNT_ID` (required for production)
- `CLOUDFLARE_API_TOKEN` (required for production)
- `CF_EMBED_MAX_BATCH` (int; sensible max ≤ 100; client caps to avoid failures)

Billing
- `BILLING_ENABLED` (1/0; default 1)
- `CF_PRICE_PER_M_TOKENS` (float; default 0.02 USD per 1M input tokens)
- `CF_BILLING_REBASE` (1/0; when set to 1, recompute historical costs in the billing state using the current price)

OCR & performance
- `TESSDATA_PREFIX` (must point to Tesseract `tessdata` folder)
- `OCR_DPI` (int; default 300)
- `OCR_LOG_EVERY` (int; log OCR progress every N pages)
- `OMP_NUM_THREADS` (int): threads used by underlying OCR libs


## Embeddings and billing
- Default embedding model: Cloudflare Workers AI `@cf/baai/bge-m3` (configurable).
- Batching is controlled via `CF_EMBED_MAX_BATCH` and internal caps.
- Billing ledger file: typically `billing_state.json` under the persist/export directory.
- Logs include token counts and incremental costs per batch and per file.
- Historical rebase: set `CF_BILLING_REBASE=1` to recompute all stored costs using the current `CF_PRICE_PER_M_TOKENS` (useful if you change pricing).


## Chroma integration
- Uses persistent Chroma collection (DuckDB/Parquet by default when using PersistentClient).
- Metadata values must be scalar (str/int/float/bool/None). Lists/objects are JSON-encoded automatically by the pipeline.


## Resume and caching
- `progress_state.json` tracks per-file status (size/mtime, completion, JSONL artifact name, and ingestion status) to allow safe resume.
- `seen_files.json` prevents reprocessing exact duplicates across runs.
- Embedding cache avoids recomputing vectors for identical chunk text.


## Metadata schema overview
Each JSONL line contains:

```
{
  "id": "<stable-sha1>",
  "text": "<chunk text>",
  "metadata": {
    "path": "<relative path>",
    "abs_path": "<absolute path>",
    "ext": ".pdf",
    "file_size": <int>,
    "file_mtime": <int>,
    "chunk_index": <int>,
    "total_chunks_in_doc": <int>,
    "file_hash": "<sha1>",
    "chunk_hash": "<sha1>",

    "DEPARTMENT": "EEE|MTH|...",         
    "LEVEL": "100|200|...",
    "SEMESTER": "1|2",
    "CATEGORY": "GENERAL|PQ|...",
    "COURSE_FOLDER": "<folder name>",
    "COURSE_CODE": "EEE|MTH|...",
    "COURSE_NUMBER": "313|...",
    "SUBCATEGORY": "<free text>",
    "FILENAME": "<file name>",
    "STEM": "<basename without ext>",
    "GROUP_KEY": "<computed grouping key>",

    "pdf_title": "<title>",
    "pdf_creation_date": "<iso>",
    "pdf_modification_date": "<iso>",

    "processing_method": "direct|ocr|unknown",
    "page_count": <int>,
    "word_count": <int>,

    "is_duplicate": false,
    "duplicate_of_index": null,
    "duplicate_of_hash": null,

    "everytag": false
  },
  "embedding": [ ... ],
  "embedding_type": "cloudflare-bge-m3"
}
```


## The "everytag" metadata: what it is, when to use it, and when not to
"everytag" is an optional boolean metadata flag you can attach to a chunk (or an entire document by applying it to all chunks) to mark it as universally relevant across your tag-based filters (DEPARTMENT/LEVEL/CATEGORY/etc.).

Intended behavior
- Retrieval/filtering layers may treat `everytag=true` chunks as globally visible, regardless of selected filters, so that truly general resources are discoverable in all views.
- Indexers may optionally duplicate such chunks logically across major facets to ensure they surface in group-specific searches without you having to assign many tags.

When to use everytag
- Generic or foundational materials that apply to multiple departments/courses/levels:
  - university-wide policy documents
  - general study skills guides, lab safety manuals, academic integrity statements
  - shared formula sheets or math/physics references used across many courses
  - onboarding/how-to-use-this-repo documentation

When NOT to use everytag
- Content scoped to a specific course, term, or assessment:
  - course lecture notes, assignments, or slides for a single course/semester
  - past questions (PQ) tied to a particular course code/number
  - instructor-specific materials or departmental notices
- If applied too broadly, `everytag` can:
  - create search noise and false positives
  - hide more relevant, specifically tagged results
  - increase storage and indexing overhead (if the system duplicates entries across facets)

Best practices
- Default is `everytag=false`; leave it unset/false unless you have a strong reason.
- Curate a small set of high-quality, universal documents for `everytag=true`.
- Avoid combining `everytag=true` with `is_duplicate=true` records; duplicates should typically be excluded from indexing.
- If your UI supports it, consider visually indicating globally-relevant results so users understand why they appear outside strict filters.


## OCR quality tips
- DPI: 300 is baseline; 450–600 improves small fonts (with more memory/time).
- PSM: `6` for a single uniform text block; `4` for single column with variable sizes; `3/1` when layout varies widely.
- OEM: `1` (LSTM) is a strong default; try `3` only if needed.
- Enable rotation detection (`--ocr-rotate`) for scanned PDFs; use preprocessing (`--ocr-preprocess`) for noisy scans.


## Monitoring
- Logs are written under `run_logs/` (e.g., `latest_run.log`).
- Billing lines show per-batch tokens and cumulative cost.
- Set `$env:OCR_LOG_EVERY = "5"` to print OCR page progress more frequently.

Tail logs in PowerShell:

```
Get-Content run_logs\latest_run.log -Wait -Tail 60
```


## Troubleshooting
- Tesseract cannot find language data: verify `TESSDATA_PREFIX` and `eng.traineddata` exist.
- OCR is poor: increase `--ocr-dpi`, enable `--ocr-rotate` and `--ocr-preprocess`, adjust `--ocr-psm`.
- Missing Cloudflare credentials: set `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`.
- Chroma metadata type errors: lists/objects are JSON-encoded by the pipeline to comply with scalar-only constraints.
- Enhanced cache warnings: if you see warnings about an enhanced cache JSON under `data\gemini_cache\`, you can delete or reinitialize it; it won’t block OCR/embeddings.


## One-file verification example
Run with a single worker to observe OCR and billing closely, piping logs to a file:

```
$env:OCR_LOG_EVERY = "5"
& ".\.venv\Scripts\python.exe" -m src.services.RAG.convert_to_embeddings `
  -i "C:\Users\<you>\Documents\SCHOOL\COMPILATION\EEE" `
  --export-dir "C:\Users\<you>\Documents\Recursive-PDF-EXTRACTION-AND-RAG\data\exported_data" `
  --cache-dir  "C:\Users\<you>\Documents\Recursive-PDF-EXTRACTION-AND-RAG\data\ocr_cache" `
  --workers 1 `
  --omp-threads 4 `
  --resume `
  --with-chroma `
  --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe" `
  --ocr-on-missing fallback `
  --ocr-rotate `
  --ocr-preprocess `
  --ocr-psm 6 `
  --ocr-oem 1 `
  -c pdfs_bge_m3_cloudflare `
  -p "C:\Users\<you>\Documents\Recursive-PDF-EXTRACTION-AND-RAG\chroma_db_bge_m3" 2>&1 | Tee-Object -FilePath "run_logs\latest_run.log"
```


## Notes
- This README uses Windows and PowerShell examples. Adjust paths/commands for macOS/Linux as needed.
- Keep secrets out of source control and console output; always set credentials via environment variables.
# CourseGen
