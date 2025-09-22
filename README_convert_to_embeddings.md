# Convert to Embeddings Pipeline in CourseGen

The `convert_to_embeddings.py` script is the core ingestion pipeline, transforming folders of PDFs (text-based or scanned) into searchable vector embeddings stored in ChromaDB. It handles OCR for non-extractable text, intelligent chunking, deduplication, metadata extraction, batch embedding, and resumable upserting. Designed for large-scale educational document processing (e.g., 1000+ PDFs), it emphasizes efficiency, cost tracking, and rich metadata for RAG applications.

## Overview
Processing legacy PDFs (e.g., scanned lecture notes, past questions) requires robust OCR and vectorization. This pipeline:
- Recursively discovers PDFs/images.
- Applies OCR with preprocessing (denoising, rotation) via Tesseract + OpenCV.
- Chunks text semantically (paragraph-aware), deduplicates within/across files.
- Extracts metadata (e.g., course code from path: "EEE/400/1/EEE471/file.pdf" → DEPARTMENT=EEE, LEVEL=400).
- Embeds via Cloudflare BGE-M3 (or Ollama) in batches.
- Upserts to persistent ChromaDB with scalar metadata.
- Tracks progress, billing, and caches for resumability.

Key benefits:
- Handles 10-50 PDFs/hour on modest hardware (multi-threaded).
- 95%+ OCR accuracy with tuning.
- Resumable: Interrupt and resume without reprocessing.
- Cost-aware: Logs token usage (~$0.01-0.05 per PDF).
- Metadata-rich: Enables filtered RAG (e.g., by semester/category).

## Architecture
1. **Discovery**: Scans `--input-dir` recursively; filters PDFs/images.
2. **OCR Engine** (`services/RAG/ocr_engine.py`):
   - Detects text layer; falls back to OCR if missing/poor.
   - Renders pages at DPI (300-600); detects rotation.
   - Preprocesses: Grayscale, threshold, sharpen (OpenCV).
   - Tesseract: PSM/OEM tuning; multi-page output.
3. **Text Processing** (`chunking.py`, `utils/Remove Duplicates/remove_duplicates.py`):
   - Splits into chunks (200-500 words, sentence boundaries).
   - Dedups: SHA1 hashing + fuzzy matching (threshold 0.95).
   - Cleans: Removes artifacts, normalizes whitespace.
4. **Metadata Extraction** (`path_meta.py`, `metadata_extractor.py`):
   - Parses paths for tags (DEPARTMENT, LEVEL, COURSE_CODE, etc.).
   - PDF props: Title, creation/mod dates via PyMuPDF.
   - Custom: "everytag" for universal chunks.
5. **Embedding** (`cloudflare_service.py` or `ollama_service.py`):
   - Batches chunks (≤96); embeds with BGE-M3 (1024 dim).
   - Caches vectors by chunk hash.
6. **Storage** (`chroma_store.py`):
   - Exports JSONL to `--export-dir` (one file per PDF).
   - Upserts to Chroma collection (`--collection`); persists to `--persist-dir`.
   - Handles metadata as scalars (JSON-encodes lists/objects).
7. **Tracking** (`progress_store.py`, `billing.py`):
   - `progress_state.json`: File status (processed, embedded).
   - `seen_files.json`: Global dedup.
   - Billing: Tokens/costs per batch/file.

Dependencies:
- PyMuPDF (PDF handling), pytesseract, opencv-python.
- Cloudflare Workers AI or Ollama.
- ChromaDB (persistent mode).

## Usage
### Prerequisites
- Install: `pip install -r requirements.txt`.
- Tesseract: Install binary; set `TESSDATA_PREFIX`.
- Secrets: Cloudflare env vars for embeddings.

### CLI Command
```
python -m services.RAG.convert_to_embeddings [OPTIONS]
```

#### Core Options
- `-i, --input-dir PATH`: Root folder (recursive PDFs; required).
- `--export-dir PATH`: JSONL outputs + progress (default "data/exported_data").
- `--cache-dir PATH`: OCR/embed caches (default "data/ocr_cache").
- `-c, --collection STR`: Chroma name (default "pdfs_bge_m3_cloudflare").
- `-p, --persist-dir PATH`: Chroma path (default "chromadb_storage").
- `--workers INT`: Parallel files (default 1; max CPU cores).
- `--omp-threads INT`: OCR threads (default 4).
- `--resume`: Skip completed (uses mtime/size; default false).
- `--with-chroma`: Upsert to DB (default true).
- `--force-ocr`: OCR all PDFs (ignore text layer).
- `--dry-run`: Simulate without processing.

#### OCR Options
- `--tesseract-cmd PATH`: Tesseract exe (auto-detect if unset).
- `--ocr-on-missing STR`: fallback/error/skip (default fallback).
- `--ocr-dpi INT`: Render DPI (300/450/600; default 300).
- `--ocr-psm INT`: Segmentation (3=auto,6=block; default 6).
- `--ocr-oem INT`: Engine (1=LSTM; default 1).
- `--ocr-extra-config STR`: e.g., "tessedit_char_whitelist=0123456789" (default none).
- `--ocr-rotate`: Auto-rotation (default false).
- `--ocr-preprocess`: OpenCV denoise/threshold (default false).
- `--ocr-log-every INT`: Log progress every N pages (default 10).

#### Embedding/Billing Options
- `--embed-provider STR`: cloudflare/ollama (default cloudflare).
- `--embed-model STR`: e.g., "@cf/baai/bge-m3" (default).
- `--batch-size INT`: Chunks per embed call (≤96; default 50).
- `--billing-enabled`: Track costs (default true).
- `--price-per-m-tokens FLOAT`: Provider rate (default 0.02 USD).
- `--rebase-billing`: Recalculate historical costs.

#### Advanced
- `--chunk-size INT`: Max words/chunk (default 400).
- `--chunk-overlap INT`: Overlap words (default 50).
- `--dedup-threshold FLOAT`: Fuzzy dedup sim (0.0-1.0; default 0.95).
- `--metadata-tags FILE`: Custom tag mapping JSON.
- `--filter-ext LIST`: e.g., ["pdf", "pptx"] (default pdf).
- `--verbosity LEVEL`: Logging level.

### Examples
#### Basic Single Run
```
python -m services.RAG.convert_to_embeddings \
  -i data/textbooks/COMPILATION/EEE/400/1 \
  --export-dir data/exported_data \
  --cache-dir data/ocr_cache \
  --collection pdfs_bge_m3_cloudflare \
  --persist-dir chromadb_storage \
  --workers 4 \
  --resume \
  --ocr-dpi 450 \
  --ocr-rotate \
  --ocr-preprocess
```

#### High-Quality OCR for Scanned PDFs
```
python -m services.RAG.convert_to_embeddings \
  -i data/textbooks/scanned \
  --force-ocr \
  --ocr-dpi 600 \
  --ocr-psm 6 \
  --ocr-oem 1 \
  --ocr-extra-config "preserve_interword_spaces=1" \
  --workers 2  # Lower for high DPI
```

#### Resume Interrupted Run
After Ctrl+C, rerun same command with `--resume`; skips done files.

#### Inspect Output
```
python services/RAG/inspect_chroma.py -c pdfs_bge_m3_cloudflare -p chromadb_storage --query "z-transform"
```

### Programmatic Usage
```python
from services.RAG.convert_to_embeddings import process_directory

results = process_directory(
    input_dir="data/textbooks/EEE",
    export_dir="data/exported_data",
    collection="pdfs_bge_m3_cloudflare",
    persist_dir="chromadb_storage",
    workers=4,
    resume=True
)
print(f"Processed {results['files']}, embedded {results['chunks']} chunks")
```

## Output Schema
### JSONL Files (Per PDF)
`data/exported_data/EEE471_textbook.jsonl`:
```json
{
  "id": "chunk_sha1_hash",
  "text": "The z-transform is defined as Z{x[n]} = sum x[n] z^-n ...",
  "metadata": {
    "path": "EEE/400/1/EEE471/EEE471_textbook.pdf",
    "abs_path": "/full/path/to/file.pdf",
    "ext": ".pdf",
    "file_size": 2457600,
    "file_mtime": 1695286400,
    "chunk_index": 12,
    "total_chunks_in_doc": 150,
    "file_hash": "sha1_of_pdf",
    "chunk_hash": "sha1_of_text",
    "DEPARTMENT": "EEE",
    "LEVEL": "400",
    "SEMESTER": "1",
    "CATEGORY": "TEXTBOOK",
    "COURSE_CODE": "EEE471",
    "COURSE_NUMBER": "471",
    "SUBCATEGORY": "",
    "FILENAME": "EEE471_textbook.pdf",
    "STEM": "EEE471_textbook",
    "GROUP_KEY": "EEE_400_1",
    "pdf_title": "Digital Signal Processing Notes",
    "pdf_creation_date": "2023-01-15",
    "pdf_modification_date": "2023-09-10",
    "processing_method": "ocr",
    "page_count": 120,
    "word_count": 35000,
    "is_duplicate": false,
    "duplicate_of_index": null,
    "everytag": false
  },
  "embedding": [0.123, -0.456, ..., 0.789],  // 1024 floats
  "embedding_type": "cloudflare-bge-m3"
}
```

### Progress Files
- `progress_state.json`: {"files": [{"path": "...", "status": "embedded", "chunk_count": 150}]}
- `billing_state.json`: Cumulative tokens/costs.
- `seen_files.json`: Global dedup hashes.

## Performance Tuning
- **Workers/Threads**: `--workers=CPU cores`, `--omp-threads=2-4` (balance I/O vs. CPU).
- **DPI Tradeoff**: 300 fast/accurate for text; 600 slow/better for handwriting.
- **Batch Size**: 50-96 for embeddings; monitor Cloudflare limits.
- **Memory**: 4-8GB for 10+ workers; use `--workers=1` for low RAM.
- **Resume Safety**: Relies on file mtime/size; avoid editing inputs mid-run.

Benchmark: 50-page PDF @450 DPI: ~2-5 min (OCR+embed).

## Testing and Validation
- **Sanity Check**: `python run_ocr_sanity.py data/textbooks/.../file.pdf` (single file OCR).
- **Unit Tests**: `pytest tests/test_chroma_revive.py`, `test_batch_utils.py`.
- **Integration**: Process sample dir; query Chroma for recall.
- **OCR Quality**: Compare output to ground truth; tune PSM/DPI.
- **Deduplication**: Check `is_duplicate` flags; adjust threshold.

## Troubleshooting
- **Tesseract Not Found**: Set `--tesseract-cmd` and `TESSDATA_PREFIX`; verify `eng.traineddata`.
- **Poor OCR**: Enable `--ocr-preprocess --ocr-rotate`; try PSM=3/4/6; higher DPI.
- **Cloudflare Errors**: Check `CLOUDFLARE_ACCOUNT_ID/API_TOKEN`; reduce batch size.
- **Chroma Metadata Issues**: Ensure scalars; pipeline auto-JSON-encodes.
- **Out of Memory**: Lower `--workers`, DPI, or batch size.
- **Resume Fails**: Delete corrupted `progress_state.json`; rerun without `--resume`.
- **No Embeddings**: Verify provider creds; fallback to Ollama.
- **Logs**: `run_logs/latest_run.log`; set `OCR_LOG_EVERY=1` for verbose.

Common Errors:
- "Language data not found": Fix `TESSDATA_PREFIX`.
- "Batch too large": Set `CF_EMBED_MAX_BATCH=50`.
- "Duplicate chunks": Tune `--dedup-threshold=0.98`.

## Best Practices
- **Input Prep**: Organize folders by metadata (e.g., EEE/400/1/COURSE/file.pdf) for auto-tagging.
- **Quality Control**: Spot-check OCR on 5% of files; preprocess noisy scans.
- **Cost Management**: Run with `--billing-enabled`; rebase prices periodically.
- **Storage**: Use SSD for `--persist-dir`; backup Chroma periodically.
- **Everytag**: Set true only for universal docs (e.g., safety manuals); avoid overuse.
- **Updates**: After new PDFs, run with `--resume` to append.

## Future Enhancements
- Multi-format support (PPTX, DOCX via converters).
- Advanced chunking (semantic via embeddings).
- Distributed processing (Ray/Celery).
- Auto-metadata from content (NLP tagging).

This pipeline is the backbone of CourseGen, enabling all downstream RAG features with high fidelity.