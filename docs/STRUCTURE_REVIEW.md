# CourseGen — Project Structure Review

> **Purpose:** Make CourseGen a clean, standalone, provider-agnostic RAG pipeline.  
> **Date:** 2026-09-02  
> **Status:** Proposal — no changes applied yet (awaiting approval)

---

## 1. Current Structure (as on `main` @ `f04e4df`)

```
CourseGen/
├── data/
│   └── textbooks/
│       └── courses.json              # 426 courses, 389 KB — Engineering Hub artifact
├── OUTPUT_DATA2/                     # ALL_CAPS, baked into 12 files
│   ├── emdeddings/                   # TYPO — should be "embeddings"
│   ├── cache/                        # question-gen cache (JSONL per request)
│   └── data/gemini_cache/            # duplicate cache path
├── chromadb_storage/                 # SECOND chroma path (config.py default)
├── chroma_db_bge_m3/                 # THIRD reference (.gitignore + old docs)
├── utils/
│   ├── Caching/
│   ├── ImagesToPDF/
│   ├── PdfCompression/
│   ├── data_cleaning/
│   ├── database_transfer/
│   │   └── transfer_db.py            # hardcodes name="Engineering Hub" + name="undefined"
│   ├── logging_utils.py
│   ├── metadata_extractor.py
│   └── progress_tracker.py
├── services/
│   ├── RAG/                          # pdf → ocr → chunk → embed → chroma
│   ├── QuestionRag/                  # question + outline generation
│   │   └── requirements.txt          # duplicates root (legacy google-generativeai + boto3)
│   ├── Gemini/
│   ├── Cloudflare/
│   ├── Firestore/
│   ├── Ollama/
│   └── Email/
├── firebase_functions/               # standalone Node service (question stats)
├── scripts/
│   ├── calc_probe.py
│   ├── health_check.py
│   └── list_courses.py
├── tests/                            # 7 files
├── data_models/                      # 9 pydantic models
├── build.sh                          # YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/rag:latest
├── run.sh                            # same placeholder
├── ec2_execution.sh                  # same placeholder + duplicate logic
├── quick-build.sh                    # 4th build script, CPU-only
├── docker-compose.yml                # hardcoded image + OUTPUT_DATA2 mounts
├── Dockerfile
├── Dockerfile.minimal
├── requirements.txt                  # 40+ deps, ~19 unused (see §3)
├── config.py                         # 5 env vars for 2 dirs
├── .env.example                      # 60+ vars, many overlapping
└── README.md                         # 875 lines, Engineering Hub refs throughout
```

---

## 2. Problems

| # | Problem | Impact | Files Affected |
|---|---------|--------|----------------|
| 1 | **3 chroma paths** for the same store | Only one is active depending on which env var is set; others are dead | `config.py`, `chroma_store.py`, `chromadb_query.py`, `convert_to_embeddings.py`, `Dockerfile*`, `docker-compose.yml`, `build.sh`, `.gitignore` |
| 2 | **Typo `emdeddings`** | Propagated everywhere; looks unprofessional | 12 files (py, yml, sh, Dockerfile, md) |
| 3 | **ALL_CAPS `OUTPUT_DATA2`** | Non-standard; looks like a temp dump, not a storage dir | 15+ files |
| 4 | **`courses.json` committed** | Engineering Hub-specific; breaks "standalone" claim; 389 KB committed artifact | `data/textbooks/courses.json` |
| 5 | **Hardcoded `YOUR_AWS_ACCOUNT_ID.../rag:latest`** | Placeholder never replaced; deploy scripts all broken without manual edit | `build.sh`, `run.sh`, `ec2_execution.sh`, `docker-compose.yml`, `README.md` |
| 6 | **4 build scripts** with overlapping logic | `build.sh` + `run.sh` + `ec2_execution.sh` + `quick-build.sh` — hard to know which to use | root `*.sh` |
| 7 | **5 env vars for 2 directories** | `COURSEGEN_COURSES_JSON` + `COURSEGEN_CACHE_DIR` + `CHROMA_PERSIST_DIR` + `CHROMADB_STORAGE` + `COURSEGEN_OUTPUT_ROOT` | `config.py`, `course_outline_generator.py`, `question_generator.py`, `chromadb_query.py` |
| 8 | **`utils/database_transfer/transfer_db.py`** | Hardcodes `name="Engineering Hub"` and `name="undefined"`; typos (`recieve`, `transfer_cred`) | `utils/database_transfer/transfer_db.py` |
| 9 | **Duplicate `requirements.txt`** | `services/QuestionRag/requirements.txt` lists `google-generativeai` + `boto3` separately from root | `services/QuestionRag/requirements.txt` |
| 10 | **Mixed casing** | `services/RAG` vs `utils/Caching` vs `utils/PdfCompression` — inconsistent | `services/*`, `utils/*` |
| 11 | **`firebase_functions/` at root** | Couples to Firestore question stats; not needed for standalone RAG | `firebase_functions/` |
| 12 | **19 unused deps** in `requirements.txt` | Bloats Docker image (~400 MB extra), slower installs | `requirements.txt` |

---

## 3. Unused Dependencies (19 packages, zero imports found)

| Category | Package | Status |
|----------|---------|--------|
| PDF | `pypdfium2==4.30.0` | Not imported — `PyMuPDF` is the actual PDF lib |
| Doc | `python-docx==1.2.0` | Not imported |
| AI (legacy) | `google-generativeai==0.8.5` | Legacy SDK — code uses `google-genai==1.32.0` |
| AI | `aistudio-sdk==0.3.6` | Not imported |
| AI | `openai==1.104.2` | Only string `openai_compat` mode, no `import openai` |
| DB | `google-cloud-storage==3.3.1` | Transitive via `firebase_admin` — not directly imported |
| Comms | `twilio==9.4.2` | Not imported |
| Image | `albumentations==1.4.10` | Not imported |
| Image | `albucore==0.0.13` | Not imported |
| Image | `imageio==2.37.0` | Not imported |
| Image | `tifffile==2025.8.28` | Not imported |
| Image | `shapely==2.1.1` | Not imported |
| Image | `scikit-image==0.24.0` | Not imported |
| Data | `pandas==2.3.2` | Only in `health_check.py` import-check list |
| Data | `joblib==1.5.2` | Not imported |
| Data | `networkx==3.3` | Not imported |
| Text | `beautifulsoup4==4.13.5` | Not imported |
| Text | `lxml==6.0.1` | Not imported (dep of bs4, unused) |
| Text | `bleach==6.2.0` | Not imported |
| Text | `rapidfuzz==3.14.0` | Not imported |
| Cache | `cachetools==5.5.2` | Not imported |
| Cache | `ujson==5.11.0` | Not imported |
| Cache | `orjson==3.11.3` | Not imported |
| Util | `psutil==7.0.0` | Not imported |
| Async | `aiohttp==3.9.1` | Not imported (code uses `httpx`) |
| Misc | `PyYAML==6.0.2` | Not imported |
| Misc | `typer==0.17.3` | Not imported |
| Misc | `tqdm==4.67.1` | Not imported |
| Misc | `tenacity==9.1.2` | Not imported |

**Kept (actually used):** `chromadb`, `google-genai`, `pydantic`, `PyMuPDF`, `opencv-python-headless`, `pillow`, `python-dotenv`, `requests`, `firebase_admin`, `google-cloud-firestore`, `numpy`, `httpx`

---

## 4. Proposed Generalized Structure

```
CourseGen/                                # standalone, provider-agnostic
│
├── data/
│   ├── samples/
│   │   └── courses.example.json          # 2-course example (replaces 426-course dump)
│   └── textbooks/                        # .gitkeep — user drops PDFs here (gitignored)
│
├── storage/                              # was OUTPUT_DATA2 — single, lowercase, correct spelling
│   ├── embeddings/                       # was emdeddings — ChromaDB persist dir
│   ├── cache/                            # question-gen cache
│   └── exports/                          # generated question JSONL
│
├── services/
│   ├── rag/                              # pdf → ocr → chunk → embed → chroma
│   ├── question_rag/                     # question + outline generation
│   ├── gemini/
│   ├── cloudflare/
│   ├── firestore/
│   ├── ollama/
│   └── email/
│
├── config/
│   └── settings.py                       # single settings module (replaces config.py spaghetti)
│
├── scripts/
│   ├── build.sh                          # single build (replaces build.sh + quick-build.sh)
│   ├── run.sh                            # single runner (replaces run.sh + ec2_execution.sh)
│   ├── health_check.py
│   └── migrate.py                        # generic DB migrate (replaces database_transfer/transfer_db.py)
│
├── utils/                                # keep only generic helpers
│   ├── logging_utils.py
│   ├── metadata_extractor.py
│   └── progress_tracker.py
│   # REMOVE: Caching/, ImagesToPDF/, PdfCompression/, data_cleaning/, database_transfer/
│   #         (or move to archive/ if you want to keep)
│
├── tests/
├── data_models/
├── Dockerfile
├── docker-compose.yml                    # image: coursegen:latest + ${ECR_REGISTRY} override
├── requirements.txt                      # pruned — 13 kept, 19 removed
├── .env.example                          # 5 vars → 3: GOOGLE_API_KEY, CLOUDFLARE_*, STORAGE_DIR
└── README.md                             # generic — no Engineering Hub refs
```

---

## 5. Detailed Changes

### 5.1 Paths

| Old | New | Action |
|-----|-----|--------|
| `OUTPUT_DATA2/emdeddings` | `storage/embeddings` | Rename dir + fix typo in 12 files |
| `OUTPUT_DATA2/cache` | `storage/cache` | Rename |
| `OUTPUT_DATA2/data/gemini_cache` | `storage/cache/gemini` | Deduplicate |
| `chromadb_storage/` | _(removed)_ | Was duplicate of `storage/embeddings` |
| `chroma_db_bge_m3` | _(removed)_ | Legacy name in `.gitignore` only |
| `data/textbooks/courses.json` (389 KB) | `data/samples/courses.example.json` (2 courses) + `data/textbooks/.gitkeep` | Remove committed artifact, keep example |

### 5.2 Env Vars

| Old (5 vars) | New (3 vars) | Notes |
|--------------|--------------|-------|
| `COURSEGEN_COURSES_JSON` | `COURSES_JSON` (optional override) | Default: `data/textbooks/courses.json` |
| `COURSEGEN_CACHE_DIR` | _(removed)_ | → `STORAGE_DIR/cache` |
| `CHROMA_PERSIST_DIR` | _(removed)_ | → `STORAGE_DIR/embeddings` |
| `CHROMADB_STORAGE` | _(removed)_ | Duplicate — same as above |
| `COURSEGEN_OUTPUT_ROOT` | `STORAGE_DIR` | Single root, default `storage/` |

Single `STORAGE_DIR` with sensible defaults; individual overrides only if needed.

### 5.3 Docker / Deploy

| Old | New |
|-----|-----|
| `YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/rag:latest` | `coursegen:latest` (local default) |
| Hardcoded ECR in 4 scripts + compose + README | `ECR_REGISTRY` env var — `build.sh --push` uses it, otherwise local build |
| 4 scripts (`build.sh`, `run.sh`, `ec2_execution.sh`, `quick-build.sh`) | 2 scripts (`scripts/build.sh`, `scripts/run.sh`) |

### 5.4 Code

| File | Change |
|------|--------|
| `utils/database_transfer/transfer_db.py` | Remove (Engineering Hub-specific, hardcodes `name="Engineering Hub"`) or rewrite as generic `scripts/migrate.py` with `--source` / `--dest` args |
| `services/QuestionRag/requirements.txt` | Delete — root `requirements.txt` is the single source |
| `firebase_functions/` | Keep or remove? — couples to Firestore question stats. Recommend keep but document as optional |
| `config.py` | Replace with `config/settings.py` using `STORAGE_DIR` pattern |

---

## 6. Options

### Option A — Full Restructure (breaking rename)

- Rename dirs (`OUTPUT_DATA2` → `storage`, fix `emdeddings`, lowercase `services/RAG`)
- Remove `courses.json`, `chromadb_storage` dup, 19 deps, 2 extra build scripts
- Fix hardcoded images + env var spaghetti
- **Pros:** Truly clean, standalone, professional
- **Cons:** Breaks any existing local clones / scripts that reference old paths; needs one big commit

### Option B — Minimal Fix (non-breaking)

- Keep current dir names, just fix `emdeddings` typo in place
- Remove `courses.json` → add `data/samples/courses.example.json`
- Prune 19 deps, fix hardcoded ECR → `${ECR_REGISTRY:-coursegen}:latest`
- Collapse 5 env vars → `STORAGE_DIR` but keep old vars as aliases
- **Pros:** No breaking renames; existing checkouts still work
- **Cons:** Still has `OUTPUT_DATA2` (ALL_CAPS) and mixed casing

### Recommendation

**Option A** — you asked for "very proper cleanup" and "standalone." The old paths are not worth preserving. One breaking commit now is cheaper than carrying `OUTPUT_DATA2/emdeddings` forever.

---

## 7. Approval

- [ ] Approve **Option A** (full restructure)
- [ ] Approve **Option B** (minimal fix)
- [ ] Approve with modifications (note below)

**Notes / overrides:**
> _(fill in)_

---

*Generated for Nasir — CourseGen standalone cleanup — 2026-09-02*
