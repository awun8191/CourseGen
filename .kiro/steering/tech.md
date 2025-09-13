# Technology Stack

## Core Technologies
- **Python 3.10+**: Primary language
- **Pydantic**: Data validation and settings management
- **ChromaDB**: Vector database for embeddings storage
- **PyMuPDF (fitz)**: PDF processing and text extraction
- **Tesseract OCR**: Primary OCR engine with Windows support

## AI/ML Stack
- **Google Gemini API**: Question generation and content processing
- **Cloudflare Workers AI**: BGE-M3 embeddings via `@cf/baai/bge-m3`
- **PaddleOCR**: Alternative OCR engine
- **EasyOCR**: Fallback OCR option
- **OpenCV**: Image preprocessing for OCR quality

## Key Libraries
- **httpx**: Async HTTP client for API calls
- **Pillow (PIL)**: Image processing
- **numpy**: Numerical operations
- **pandas**: Data manipulation
- **asyncio**: Asynchronous processing
- **concurrent.futures**: Multiprocessing support

## Environment Setup

### Required Environment Variables
```bash
# Cloudflare (Required for production)
CLOUDFLARE_ACCOUNT_ID="your_account_id"
CLOUDFLARE_API_TOKEN="your_api_token"

# Google Gemini
GOOGLE_API_KEY="your_gemini_key"
GEMINI_API_KEYS="key1,key2,key3"  # Multiple keys for rotation

# Tesseract OCR (Windows)
TESSDATA_PREFIX="C:\Program Files\Tesseract-OCR\tessdata"

# Performance tuning
OMP_NUM_THREADS="4"
CF_EMBED_MAX_BATCH="96"
OCR_DPI="450"

# Billing
BILLING_ENABLED="1"
CF_PRICE_PER_M_TOKENS="0.02"
```

### Installation Requirements
- **Tesseract OCR**: Must be installed at `C:\Program Files\Tesseract-OCR\tesseract.exe` (Windows)
- **Python Virtual Environment**: Use `.venv` for dependency isolation

## Common Commands

### Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Main Pipeline Execution
```bash
# Full pipeline with resume capability
python -m services.RAG.convert_to_embeddings \
  -i "path/to/pdfs" \
  --export-dir "data/exported_data" \
  --cache-dir "data/ocr_cache" \
  --workers 6 \
  --resume \
  --with-chroma \
  --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe" \
  --ocr-on-missing fallback \
  --ocr-rotate \
  --ocr-preprocess \
  -c collection_name \
  -p "chroma_db_path"
```

### Question Generation
```bash
# Batch question generation
python batch_processing.py
```

### Development
```bash
# Run tests
python -m pytest tests/

# Check specific service
python -m services.RAG.inspect_chroma
```

## Build System
- **No formal build system**: Direct Python execution
- **Package structure**: Services-based architecture with relative imports
- **Configuration**: Environment variables + `config.json` fallback
- **Logging**: Custom logging utilities in `utils/logging_utils.py`

## Performance Considerations
- **Multiprocessing**: File-level parallelism with configurable workers
- **Caching**: Multi-tier caching (OCR, embeddings, API responses)
- **Batch Processing**: Optimized for large document collections
- **Memory Management**: Configurable image size limits and threading