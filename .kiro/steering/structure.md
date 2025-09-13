# Project Structure

## Root Directory Organization
```
├── .kiro/                          # Kiro configuration and steering
├── services/                       # Core service modules
├── data_models/                    # Pydantic data models
├── utils/                          # Utility functions
├── scripts/                        # Helper scripts
├── data/                          # Data storage (gitignored)
├── chromadb_storage/              # ChromaDB persistence (gitignored)
├── run_logs/                      # Application logs
├── config.py                      # Configuration management
├── requirements.txt               # Python dependencies
└── batch_processing.py            # Batch processing entry point
```

## Services Architecture
The project follows a **services-based architecture** with clear separation of concerns:

### `/services/RAG/`
- **Primary pipeline**: PDF processing, OCR, chunking, embeddings
- **Key files**:
  - `convert_to_embeddings.py` - Main pipeline entry point
  - `chunking.py` - Text chunking and paragraph processing
  - `helpers.py` - OCR prompts and utilities
  - `log_utils.py` - Logging utilities

### `/services/Gemini/`
- **AI integration**: Gemini API client and question generation
- **Key files**:
  - `gemini_service.py` - Main Gemini client wrapper
  - `api_key_manager.py` - API key rotation management
  - `gemini_api_keys.py` - Key configuration

### `/services/QuestionRag/`
- **Question generation**: RAG-enhanced question creation
- **Key files**:
  - `gemini_question_gen.py` - Question generation logic
  - `chromadb_query.py` - ChromaDB query interface

### `/services/Cloudflare/`
- **Embeddings**: Cloudflare Workers AI integration for BGE-M3

### `/services/Ollama/`
- **Alternative embeddings**: Ollama client for local embeddings

### `/services/Firestore/`
- **Database**: Firestore utilities and connections

## Data Models (`/data_models/`)
**Pydantic-based** data validation and structure:
- `course_model.py` - Course structure and metadata
- `course_outline.py` - Course outline generation
- `gemini_config.py` - Gemini API configuration

## Utilities (`/utils/`)
- `logging_utils.py` - Colored logging with ANSI codes
- Caching utilities
- Data cleaning helpers
- Progress tracking utilities

## Configuration Management
- **Environment variables**: Primary configuration method
- **`config.py`**: Centralized config with `config.json` fallback
- **Dataclass-based**: Type-safe configuration using `@dataclass`

## File Naming Conventions
- **Snake_case**: All Python files and directories
- **Descriptive names**: Clear purpose indication
- **Service prefixes**: Group related functionality
- **Version suffixes**: For iterative development (e.g., `_v2.py`)

## Import Patterns
- **Relative imports**: Within service modules
- **Absolute imports**: Cross-service dependencies
- **Path manipulation**: Dynamic sys.path for repository root access

## Data Storage Patterns
- **`data/` directory**: All runtime data (gitignored)
  - `exported_data/` - JSONL outputs per file
  - `ocr_cache/` - OCR result caching
  - `gemini_cache/` - API response caching
- **Progress tracking**: JSON state files for resumability
- **Metadata preservation**: Comprehensive file and processing metadata

## Course Data Structure
Based on the courses.json, the system expects:
- **Course codes**: Format like "AAE 101", "EEE 313"
- **Department mapping**: Multiple programs per course
- **Level classification**: 100, 200, 300, 400, 500 levels
- **Semester organization**: FIRST/SECOND semester designation
- **Course types**: CORE/ELECTIVE classification

## Metadata Schema
The system extracts and uses structured metadata:
- **Academic hierarchy**: DEPARTMENT → LEVEL → SEMESTER → COURSE
- **File classification**: Automatic extraction from file paths
- **Processing metadata**: OCR method, quality metrics, token counts
- **Deduplication**: Hash-based duplicate detection and tracking

## CLI and Batch Processing
- **Module execution**: `python -m services.RAG.convert_to_embeddings`
- **Argument parsing**: Comprehensive CLI with argparse
- **Batch operations**: JSONL-based batch processing for scalability
- **Resume capability**: State-based resumption after interruptions

## Logging and Monitoring
- **Structured logging**: Color-coded, descriptive log messages
- **Progress tracking**: Real-time progress indicators
- **Cost monitoring**: Token usage and billing integration
- **Error handling**: Graceful degradation and retry mechanisms