# Convert-to-Embeddings Pipeline

`services/RAG/convert_to_embeddings.py` converts large collections of PDFs into searchable vectors while capturing rich metadata, billing information, and resume state. It is opinionated around Cloudflare’s BGE-M3 embeddings and the CourseGen directory layout (`OUTPUT_DATA2`), but the code paths are modular enough for local experimentation.

## Highlights
- **Auto-resume** – every run persists progress to `progress_state.json`; unfinished files are picked up automatically on the next invocation.
- **Hybrid text extraction** – prefers native PDF text, falling back to Gemini OCR with optional EasyOCR hybrid mode when needed.
- **Streaming embeddings** – chunks are deduped, embedded in controllable batches, and streamed straight to JSONL to avoid high memory usage.
- **Chroma aware** – vectors are upserted into the configured Chroma collection immediately (can be skipped with `--no-chroma`).
- **Cost visibility** – token counts feed `Billing` so estimated spend per file is recorded.
- **Duplication control** – SHA-256 hashes keep per-run `seen_files.json` updated, preventing reprocessing the same binary.

## Processing Flow
1. **Discovery** – recursively walks `--input-dir`, skipping dot directories and non-PDF files.
2. **Text extraction** – attempts direct text extraction via PyMuPDF; if insufficient text is detected or `--force-ocr` is set, delegates to `services/RAG/ocr_engine.ocr_pdf`.
3. **Chunking & dedupe** – breaks text into 2-paragraph windows with sentence overlap (`chunk()`), then applies SHA1 + fuzzy dedup (`dedupe()`).
4. **Embedding** – streams batches through Cloudflare’s BGE-M3 endpoint with adaptive batch sizes and token accounting; vectors are cached per chunk hash.
5. **Export** – writes a per-PDF JSONL to the export directory and optionally upserts batches into Chroma (default) without loading the entire file into memory.
6. **Progress update** – `progress_state.json` and `seen_files.json` are updated after each file, pairing with billing and cache directories to support restarts.

## Prerequisites
- Python dependencies from `requirements.txt` (PyMuPDF, Pillow, requests, numpy, opencv-python (optional), EasyOCR (optional), google-genai (for Gemini OCR)).
- Environment variables:
  - `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` (required for embeddings).
  - Optional Cloudflare tuning: `CF_EMBED_MAX_BATCH`, `CF_EMBED_MAX_TOKENS`, `CF_EMBED_MIN_BATCH`.
  - OCR tuning: `OCR_LANG`, `OCR_ENGINE` (`gemini` \| `hybrid` \| `easyocr`), `OCR_GEMINI_FALLBACK_ENGINE`, `EASYOCR_GPU`, `OCR_MAX_IMAGE_BYTES`.
  - Storage overrides: `COURSEGEN_OUTPUT_ROOT`, `COURSEGEN_CACHE_ROOT`.
- Tesseract is **not** required for the default Gemini/EasyOCR flows, but can be used if you extend the OCR engine.

## Default Paths
When environment overrides are not supplied, directories are rooted at `<repo>/OUTPUT_DATA2`:
- Export JSONL + progress: `OUTPUT_DATA2/progress_report`
- Cache (OCR artifacts, temporary text, failed payloads): `OUTPUT_DATA2/cache`
- Chroma persistence: `OUTPUT_DATA2/emdeddings` (mounted volume in containers)
- Billing state: `<persist-dir>/billing_state.json`
- Dedup index: `<persist-dir>/seen_files.json`

## CLI Usage
```bash
python -m services.RAG.convert_to_embeddings --input-dir <PDF_ROOT> [options]
```

### Frequently Used Flags
| Flag | Description |
| --- | --- |
| `-i / --input-dir` | **Required.** Root directory containing PDFs (traversed recursively). |
| `--export-dir` | Where JSONL + progress files live (default `OUTPUT_DATA2/progress_report`). |
| `--cache-dir` | OCR + text cache root (default `OUTPUT_DATA2/cache`). |
| `--with-chroma` / `--no-chroma` | Toggle Chroma upserts (default on). |
| `-c / --collection` | Chroma collection name (default `course_embeddings`). |
| `-p / --persist-dir` | Chroma persistence directory (default `OUTPUT_DATA2/emdeddings`). |
| `--workers` | ProcessPool workers for PDF processing (default 2). |
| `--omp-threads` | OpenMP threads exposed to OCR libraries (default 2). |
| `--timeout` | Per-file processing timeout in seconds (default 1800). |
| `--max-pdfs` | Limit number of files processed from the discovery list (0 = all). |
| `--embed-batch` | Initial embedding batch size (bounded by `CF_EMBED_MAX_BATCH`, default 16). |
| `--ocr-dpi` | Render DPI when OCR is needed (default 200). |
| `--ocr-lang` | Language hint passed to OCR (`en` by default). |
| `--engine` | OCR engine preference (`gemini`, `hybrid`, `easyocr`; default `gemini`). |
| `--ocr-fallback-engine` | Local fallback used when Gemini returns empty text (`easyocr` or `hybrid`). |
| `--no-gemini-fallback` | Disable automatic local OCR fallback. |
| `--force-ocr` | Skip native text extraction even if the PDF has a text layer. |
| `--memory-limit` | Soft limit in MB for worker processes (0 = disabled). |
| `--retry-limit` | Max retry attempts per file (default 3). |

> Resume mode is always on. Deleting `progress_state.json` is the quickest way to restart a directory from scratch.

### Command Examples
- **Standard run with Chroma upserts:**
  ```bash
  python -m services.RAG.convert_to_embeddings \
    -i data/textbooks/EEE/400/1 \
    --export-dir OUTPUT_DATA2/progress_report \
    --cache-dir OUTPUT_DATA2/cache \
    --collection pdfs_bge_m3_cloudflare \
    --persist-dir OUTPUT_DATA2/emdeddings \
    --workers 4 \
    --embed-batch 32
  ```
- **OCR-heavy archive (high DPI, Gemini + EasyOCR hybrid):**
  ```bash
  python -m services.RAG.convert_to_embeddings \
    -i data/textbooks/scanned \
    --force-ocr \
    --ocr-dpi 450 \
    --engine hybrid \
    --ocr-fallback-engine easyocr \
    --workers 2 \
    --embed-batch 16
  ```
- **Dry run on a limited subset:**
  ```bash
  python -m services.RAG.convert_to_embeddings \
    -i data/textbooks/sample \
    --max-pdfs 5 \
    --no-chroma
  ```

## Output Artifacts
- **Per-PDF JSONL** – e.g. `OUTPUT_DATA2/progress_report/<stem>.jsonl.tmp` during processing, archived to `<stem>.jsonl` on completion.
- **Chroma** – vectors immediately upserted to the target collection (if enabled).
- **Progress files** – `progress_state.json` tracks file status (`pending`, `in_progress`, `completed`, `failed`, `skipped`); includes timing, chunk counts, duplicate counts, and Chroma status.
- **Billing** – `billing_state.json` accumulates total tokens and cost estimates per file.
- **Seen files** – `seen_files.json` stores SHA-256 prefixes to prevent duplicate ingestion.
- **Cache** – OCR intermediates, text snapshots, and per-chunk embedding caches live under `cache_dir`.

### Sample `progress_state.json` record
```json
{
  "files": {
    "/abs/path/EEE471_textbook.pdf": {
      "status": "completed",
      "jsonl_name": "EEE471_textbook.jsonl",
      "jsonl_archived": true,
      "chroma_upserted": true,
      "chunks": 152,
      "duplicates": 12,
      "file_size": 5242880,
      "file_mtime": 1716400000,
      "discovered_at": "2024-05-14T20:12:52+00:00",
      "started_at": "2024-05-14T20:13:05+00:00",
      "finished_at": "2024-05-14T20:17:41+00:00"
    }
  }
}
```

## Resume Behaviour
- Discoveries append to `progress_state.json` immediately, so ctrl+c mid-scan still records status.
- Completed files are marked `skipped` on subsequent runs unless the size/mtime changes.
- Failed Chroma upserts are retried the next time the script runs with `--with-chroma`, using existing JSONL files instead of reprocessing the PDF.
- Duplicates (based on SHA-256 prefix) are tagged `file_duplicate` and skipped gracefully.

## Performance Tuning Tips
- **Workers vs. OCR** – high DPI OCR is CPU-intensive; keep `--workers` low (1–2) when using 450–600 DPI to avoid thrashing.
- **Embedding batch size** – the script halves the batch on Cloudflare errors and ramps up when stable. Start with a moderate size (16–32) and tweak env vars to raise or lower the ceiling.
- **OMP threads** – adjust `--omp-threads` to match available CPU cores; the script exports the value as `OMP_NUM_THREADS`.
- **Timeouts** – use `--timeout` to prevent pathological files from hanging the pool (default 30 minutes per file).

## Troubleshooting
- **Missing text** – verify `--engine` and `--ocr-fallback-engine`; set `--force-ocr` if PDFs have broken text layers.
- **Gemini OCR API errors** – network hiccups or quota issues bubble up as `processing_error`; the retry mechanism attempts three times before marking the file failed.
- **Cloudflare rate limiting** – errors are logged and batch sizes shrink automatically; ensure account limits (`CF_EMBED_MAX_BATCH`, `CF_EMBED_MAX_TOKENS`) align with workload.
- **Progress file corruption** – delete `progress_state.json` (or specific entries) to restart; resume safety depends on this file being writable.
- **No vectors in Chroma** – confirm `--with-chroma` flag, collection name, and that the Chroma server can be created in the persist directory.
- **Out-of-memory** – reduce `--workers`, lower DPI, or raise `OCR_MAX_IMAGE_BYTES` only when necessary.

## Integrations
- After ingestion, query Chroma with `services/RAG/inspect_chroma.py` to verify recall:
  ```bash
  python services/RAG/inspect_chroma.py -c pdfs_bge_m3_cloudflare -p OUTPUT_DATA2/emdeddings --query "z-transform"
  ```
- Downstream services (question generation, outline generation) expect consistent metadata keys (`DEPARTMENT`, `LEVEL`, `COURSE_CODE`, etc.) provided by `path_meta.parse_path_meta`.
- The `Billing` log can be exported into monitoring dashboards or reconciled with Cloudflare usage for budgeting.

The convert-to-embeddings pipeline is designed for unattended ingestion jobs—mount the `OUTPUT_DATA2` tree in production environments so billing, progress, and caches persist across container runs.
