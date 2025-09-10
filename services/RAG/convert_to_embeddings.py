#!/usr/bin/env python3
"""
Streamlined: PDF -> (auto OCR w/ PaddleOCR) -> chunk -> dedupe -> BGE-M3 (Cloudflare)
-> per-file JSONL -> immediate Chroma upsert -> real-time billing -> resume.

Major fixes:
- Fixed PaddleOCR configuration conflict (use_angle_cls vs use_textline_orientation)
- Fixed Windows file permission issues with progress saving
- No hardcoded Cloudflare credentials; strictly require env vars.
- Proper EasyOCR fallback wired into OCR flow.
- Retry/backoff for Cloudflare API (429/5xx/network).
- Saner OCR image-size check using nbytes; DPI ladder and env-tunable cap.
- Correct cache keying and no caching of near-empty OCR/text.
- Fixed multiprocessing status merge order so "pending" isn't resurrected.
- Safer Chroma upsert and metadata sanitization.

Env:
  CLOUDFLARE_ACCOUNT_ID          (required)
  CLOUDFLARE_API_TOKEN           (required)
  CF_PRICE_PER_M_TOKENS=0.012    # USD per 1M input tokens (BGE-M3 input price)
  CF_EMBED_MAX_BATCH=96
  OMP_NUM_THREADS=4
  BILLING_ENABLED=1
  PADDLE_LANG=en                  # optional; e.g., en, fr, de, ar, hi
  EASYOCR_GPU=0                   # optional; set 1 to enable GPU for EasyOCR
  OCR_MAX_IMAGE_BYTES=67108864    # 64MB default; adjust as needed
"""

from __future__ import annotations

import os, re, sys, json, time, argparse, signal
from dataclasses import dataclass
from pathlib import Path

# Make repository root importable when this script is executed directly
try:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
except Exception:
    pass

from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

import fitz  # PyMuPDF
from PIL import Image
import requests

# New modular imports
from services.RAG.log_utils import setup_logging, snapshot
from services.RAG.billing import Billing
from services.RAG.chroma_store import chroma_client, chroma_upsert_jsonl
from services.RAG.progress_store import (
    load_progress,
    save_progress,
    safe_file_replace,
    should_skip,
)
from services.RAG.path_meta import parse_path_meta
from services.RAG.cache_utils import sha256_file
from services.RAG.chunking import chunk, dedupe, sha1_text

# Optional OpenCV for image handling (Paddle likes numpy arrays)
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

# PaddleOCR (optional)
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception:
    PADDLE_AVAILABLE = False

# ANSI colors for logging
ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}

_COLOR_PREFIX = {
    "[ERROR]": (ANSI["red"] + ANSI["bold"], True),
    "[FAIL]": (ANSI["red"] + ANSI["bold"], True),
    "[WARN]": (ANSI["yellow"] + ANSI["bold"], False),
    "[START]": (ANSI["cyan"] + ANSI["bold"], True),
    "[PROCESS]": (ANSI["cyan"], False),
    "[META]": (ANSI["blue"], False),
    "[TEXT]": (ANSI["blue"], False),
    "[EXTRACT]": (ANSI["blue"], False),
    "[CHUNK]": (ANSI["magenta"], False),
    "[EMBED]": (ANSI["magenta"] + ANSI["bold"], True),
    "[Chroma]": (ANSI["cyan"], False),
    "[Billing]": (ANSI["cyan"], False),
    "[SKIP]": (ANSI["yellow"], True),
    "[SUCCESS]": (ANSI["green"] + ANSI["bold"], True),
    "[DONE]": (ANSI["green"] + ANSI["bold"], True),
    "[INTERRUPT]": (ANSI["yellow"] + ANSI["bold"], True),
    "[WARMUP]": (ANSI["dim"], False),
    "[OCR]": (ANSI["magenta"], False),
}

def _decorate(msg: str) -> tuple[str, bool]:
    for k, (color, pad) in _COLOR_PREFIX.items():
        if msg.startswith(k):
            return f"{color}{msg}{ANSI['reset']}", pad
    return msg, False

def log(msg: str) -> None:
    colored, pad = _decorate(msg)
    if pad:
        bar = "=" * 88
        print(f"\n{bar}\n{colored}\n{bar}\n", flush=True)
    else:
        print(colored, flush=True)

def now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

# Token counting (tiktoken; fallback ≈chars/4)
class TokenCounter:
    def __init__(self) -> None:
        self._enc = None
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    def count_batch(self, texts: List[str]) -> int:
        if self._enc:
            return sum(len(self._enc.encode(t)) for t in texts)
        return sum(max(1, len(t) // 4) for t in texts)

# Cloudflare BGE-M3 client with retry/backoff
@dataclass
class RetryCfg:
    tries: int = 5
    backoff: float = 1.5
    max_sleep: float = 20.0

class CFEmbeddings:
    def __init__(self, account_id: str, api_token: str, batch_max: int, retry: RetryCfg | None = None):
        self.url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/baai/bge-m3"
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"})
        self.batch_max = max(1, min(100, batch_max))
        self.counter = TokenCounter()
        self.retry = retry or RetryCfg()

    def close(self) -> None:
        try:
            self.s.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _post_embed(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        sleep = 0.5
        for attempt in range(1, self.retry.tries + 1):
            try:
                r = self.s.post(self.url, json=payload, timeout=90)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{r.status_code} {r.text[:200]}")
                r.raise_for_status()
                return r.json()
            except Exception:
                if attempt == self.retry.tries:
                    raise
                time.sleep(min(self.retry.max_sleep, sleep))
                sleep *= self.retry.backoff
        raise RuntimeError("unhandled retry loop")

    def embed_iter(self, texts: List[str], batch_size: int):
        """Yield (embeddings_for_batch, tokens_for_batch) to reduce RAM spikes."""
        bsz = min(batch_size, self.batch_max)
        for i in range(0, len(texts), bsz):
            sub = texts[i:i+bsz]
            payload = {"text": sub, "truncate_inputs": True}
            js = self._post_embed(payload)
            data = js.get("result", {}).get("data")
            if not isinstance(data, list):
                raise RuntimeError(f"Bad embedding response: {str(js)[:200]}")
            tokens = self.counter.count_batch(sub)
            yield data, tokens

# OCR decision
def need_ocr(doc: fitz.Document, sample_pages: int = 8, min_chars_per_page: int = 200) -> bool:
    n = min(sample_pages, len(doc))
    if n == 0:
        return True
    low = 0
    for i in range(n):
        txt = doc[i].get_text("text")
        if len(txt) < min_chars_per_page:
            low += 1
    return (low / max(1, n)) >= 0.6

# PaddleOCR (per-process singleton)
_PADDLE: Optional['PaddleOCR'] = None

def get_paddle_ocr(lang: str = "en") -> Optional['PaddleOCR']:
    global _PADDLE
    if not PADDLE_AVAILABLE:
        return None
    if _PADDLE is None:
        try:
            log(f"[INFO] Initializing PaddleOCR with language: {lang}")
            local_model_dir = './paddle_models'
            if os.path.exists(local_model_dir):
                log(f"[INFO] Using local models from {local_model_dir}")
                _PADDLE = PaddleOCR(
                    use_textline_orientation=True,
                    lang=lang,
                    det_model_dir=f'{local_model_dir}/det',
                    rec_model_dir=f'{local_model_dir}/rec',
                    cls_model_dir=f'{local_model_dir}/cls'
                )
            else:
                log(f"[INFO] Using default PaddleOCR models")
                _PADDLE = PaddleOCR(
                    use_textline_orientation=True,
                    lang=lang,
                )
            log(f"[INFO] PaddleOCR initialized successfully")
        except Exception as e:
            log(f"[WARN] Failed to initialize PaddleOCR: {e}")
            _PADDLE = None
    return _PADDLE

def _pixmap_to_numpy(pix: fitz.Pixmap) -> 'np.ndarray':
    if not OPENCV_AVAILABLE:
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return np.array(img) if 'np' in globals() else __import__('numpy').array(img)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr

def ocr_page_with_paddle_or_tesseract(page: fitz.Page, dpi: int = 300, lang: str = "en") -> str:
    ocr = get_paddle_ocr(lang=lang) if PADDLE_AVAILABLE else None

    last_img = None
    ladder = [dpi, 240, 200, 150, 100, 72] if dpi >= 240 else [dpi, 150, 100, 72]
    MAX_OCR_BYTES = int(os.getenv("OCR_MAX_IMAGE_BYTES", str(64 * 1024 * 1024)))

    for attempt_dpi in ladder:
        try:
            zoom = attempt_dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = _pixmap_to_numpy(pix)
            last_img = img

            nbytes = getattr(img, "nbytes", img.size)
            if nbytes > MAX_OCR_BYTES:
                if attempt_dpi == ladder[-1]:
                    log(f"[WARN] Page image {nbytes}B too large at {attempt_dpi} DPI, skipping OCR")
                    break
                log(f"[WARN] Page image {nbytes}B > {MAX_OCR_BYTES}B at {attempt_dpi} DPI; trying lower DPI")
                continue

            if ocr is not None:
                res = ocr.ocr(img)
                flat = []
                for blk in res:
                    if blk:
                        flat.extend(blk if isinstance(blk, list) else [blk])
                lines = []
                for line in flat:
                    try:
                        if line and len(line) >= 2:
                            box, (text, conf) = line
                            xs = [p[0] for p in box]; ys = [p[1] for p in box]
                            cx = sum(xs)/4.0; cy = sum(ys)/4.0
                            lines.append((text, float(conf), (cx, cy)))
                    except Exception:
                        continue
                lines.sort(key=lambda t: (round(t[2][1]/16.0), round(t[2][0]/16.0)))
                texts = [t for (t, conf, _) in lines if t and conf >= 0.35]
                if texts:
                    if attempt_dpi != dpi:
                        log(f"[OCR] Paddle used {attempt_dpi} DPI instead of {dpi}")
                    return "\n".join(texts)
        except Exception as e:
            if any(s in str(e).lower() for s in ("memory", "alloc")) and attempt_dpi != ladder[-1]:
                log(f"[WARN] Paddle OOM at {attempt_dpi} DPI; trying lower DPI")
                continue
            log(f"[WARN] Paddle error at {attempt_dpi} DPI: {e}")
            break

    # Last resort: Tesseract
    tess_cmd = os.getenv("TESSERACT_CMD")
    if tess_cmd and os.path.exists(tess_cmd):
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = tess_cmd

            tess_lang_map = {"en":"eng","fr":"fra","de":"deu","es":"spa","it":"ita","pt":"por","nl":"nld"}
            tess_lang = tess_lang_map.get(lang, lang)

            tess_prefix = os.getenv("TESSDATA_PREFIX")
            if tess_prefix:
                lang_path = Path(tess_prefix) / "tessdata" / f"{tess_lang}.traineddata"
                if not lang_path.exists():
                    log(f"[WARN] Tesseract missing {tess_lang}.traineddata under {lang_path.parent}; skipping Tesseract")
                    return ""
            if last_img is None:
                pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                last_img = _pixmap_to_numpy(pix)

            img_pil = Image.fromarray(last_img)
            tesseract_text = pytesseract.image_to_string(img_pil, lang=tess_lang)
            if tesseract_text.strip():
                log("[OCR] Tesseract fallback used")
                return tesseract_text.strip()
        except Exception as te:
            log(f"[WARN] Tesseract fallback failed: {te}")
    return ""

# Text extract wrapper
def extract_text(pdf_path, cache_dir, force_ocr, ocr_engine, ocr_dpi, ocr_lang):
    from services.RAG.ocr_engine import ocr_pdf

    # If force_ocr is enabled, skip text extraction check
    if force_ocr:
        engine = 'paddleocr'
        result = ocr_pdf(pdf_path, lang=ocr_lang, dpi=ocr_dpi, engine=engine)
        return result.text

    # First, try to extract text directly from PDF
    try:
        doc = fitz.open(str(pdf_path))
        extracted_text = ""

        # Check first 10 pages (or all pages if less than 10)
        pages_to_check = min(10, len(doc))
        total_chars = 0
        text_pages = 0

        for i in range(pages_to_check):
            page_text = doc[i].get_text("text").strip()
            if page_text:
                extracted_text += page_text + "\n\n"
                total_chars += len(page_text)
                text_pages += 1

        doc.close()

        # If we have sufficient text (at least 5 pages with text and reasonable character count)
        if text_pages >= 5 and total_chars > 1000:
            log(f"[TEXT] Using direct text extraction: {text_pages}/{pages_to_check} pages with text, {total_chars} chars")
            return extracted_text

        log(f"[TEXT] Insufficient text found ({text_pages}/{pages_to_check} pages, {total_chars} chars), will use OCR")

    except Exception as e:
        log(f"[WARN] Failed to extract text directly: {e}, will use OCR")

    # Fall back to OCR
    engine = ocr_engine if ocr_engine in ['gemini', 'hybrid', 'easyocr', 'paddleocr'] else 'hybrid'
    result = ocr_pdf(pdf_path, lang=ocr_lang, dpi=ocr_dpi, engine=engine)
    return result.text

# Worker init for OCR
def _worker_init_for_ocr(lang: str) -> None:
    if PADDLE_AVAILABLE:
        try:
            from services.RAG.ocr_engine import get_paddle_ocr as _warm
            _ = _warm(lang)
            log(f"[WARMUP] Worker preloaded PaddleOCR (lang={lang})")
        except Exception as e:
            log(f"[WARN] Worker OCR warmup failed: {e}")

# Per-file processing
def process_one(pdf_path: str, root: str, export_tmp: str,
                cache_dir: str, cf_acct: str, cf_token: str,
                billing_file: str, embed_batch: int,
                force_ocr: bool, ocr_engine: str,
                ocr_dpi: int, ocr_lang: str) -> Dict[str, Any]:

    path = Path(pdf_path)
    rel = str(path.resolve())
    log(f"[START] Processing {path.name}")

    meta_path = parse_path_meta(path)
    log(f"[META] Parsing metadata for {path.name}")

    log(f"[TEXT] Extracting text from {path.name}")
    text = extract_text(path, Path(cache_dir), force_ocr, ocr_engine, ocr_dpi=ocr_dpi, ocr_lang=ocr_lang)
    if not text.strip():
        return {"file": rel, "skip": True, "reason": "empty_text"}
    log(f"[EXTRACT] {path.name} chars={len(text)} snapshot='{snapshot(text)}'")

    log(f"[CHUNK] Chunking text for {path.name}")
    chunks_all = chunk(text)
    uniq, dup_map = dedupe(chunks_all)
    if not uniq:
        return {"file": rel, "skip": True, "reason": "no_chunks"}
    log(f"[CHUNK] {path.name} uniq_chunks={len(uniq)} first_chunk='{snapshot(uniq[0]) if uniq else ''}'")

    # Cloudflare embeddings
    log(f"[EMBED] Creating Cloudflare BGE-M3 embeddings for {path.name} ({len(uniq)} chunks)")
    try:
        cf = CFEmbeddings(cf_acct, cf_token, int(os.getenv("CF_EMBED_MAX_BATCH", "96")))
        log(f"[EMBED] Cloudflare client initialized with account: {cf_acct[:8]}...")
        vecs, total_tokens = [], 0
        batch_count = 0
        for emb_batch, tok_batch in cf.embed_iter(uniq, batch_size=embed_batch):
            vecs.extend(emb_batch)
            total_tokens += tok_batch
            batch_count += 1
            log(f"[EMBED] Processed batch {batch_count}: {len(emb_batch)} vectors, {tok_batch} tokens")
        if len(vecs) != len(uniq):
            return {"file": rel, "error": f"embedding_mismatch: got {len(vecs)} vectors, expected {len(uniq)}"}
        log(f"[EMBED] Successfully created {len(vecs)} Cloudflare embeddings, total tokens: {total_tokens}")
    except Exception as e:
        log(f"[ERROR] Cloudflare embedding failed: {e}")
        return {"file": rel, "error": f"cloudflare_embedding_error: {e}"}

    # Write per-file JSONL
    tmp_dir = Path(export_tmp); tmp_dir.mkdir(parents=True, exist_ok=True)
    group = meta_path["GROUP_KEY"]
    jsonl_name = f"{re.sub(r'[^A-Za-z0-9._-]+','_',group)}__{sha1_text(rel)}.jsonl"
    jsonl_tmp = tmp_dir / jsonl_name

    file_hash = sha256_file(path)[:16]
    st = path.stat()
    doc_hash = sha1_text(text)
    with jsonl_tmp.open("w", encoding="utf-8") as out:
        total = len(chunks_all)
        k = 0
        for idx, ch in enumerate(chunks_all):
            if idx in dup_map:
                continue
            chash = sha1_text(ch)
            rid = sha1_text(f"{doc_hash}:{idx}:{chash}")
            md = {
                "path": str(path),
                "chunk_index": idx,
                "total_chunks_in_doc": total,
                "file_size": st.st_size,
                "file_mtime": int(st.st_mtime),
                "file_hash": file_hash,
                "chunk_hash": chash,
                **meta_path,
            }
            out.write(json.dumps({
                "id": rid,
                "text": ch,
                "metadata": md,
                "embedding": vecs[k],
                "embedding_type": "cloudflare-bge-m3"
            }) + "\n")
            k += 1
        # duplicates
        for idx, (orig_idx, orig_h) in dup_map.items():
            ch = chunks_all[idx]
            rid = sha1_text(f"{doc_hash}:{idx}:{orig_h}:dup")
            md = {
                "path": str(path),
                "chunk_index": idx,
                "total_chunks_in_doc": total,
                "file_hash": file_hash,
                "chunk_hash": sha1_text(ch),
                "is_duplicate": True,
                "duplicate_of_index": orig_idx,
                "duplicate_of_hash": orig_h,
                "skip_index": True,
                **meta_path
            }
            out.write(json.dumps({"id": rid, "text": ch, "metadata": md}) + "\n")

    return {"file": rel, "jsonl_tmp": str(jsonl_tmp),
            "chunks": len(uniq), "dups": len(dup_map), "jsonl_name": jsonl_name,
            "total_tokens": total_tokens}

# Main
def signal_handler(signum, frame):
    log(f"[INTERRUPT] Received signal {signum}, gracefully shutting down...")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    log("[INIT] Resume functionality is ALWAYS ENABLED - script will automatically resume from previous progress")

    ap = argparse.ArgumentParser("Streamlined BGE-M3 pipeline")
    ap.add_argument("-i", "--input-dir", required=True)
    ap.add_argument("--export-dir", default="OUTPUT_DATA2/progress_report")
    ap.add_argument("--cache-dir", default="OUTPUT_DATA2/cache")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--omp-threads", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800, help="Timeout per file in seconds")
    ap.add_argument("--with-chroma", dest="with_chroma", action="store_true", default=True)
    ap.add_argument("--no-chroma", dest="with_chroma", action="store_false")
    ap.add_argument("-c", "--collection", default="course_embeddings")
    ap.add_argument("--persist-dir", default="OUTPUT_DATA2/emdeddings")
    ap.add_argument("--ocr-on-missing", choices=["fallback", "error", "skip"], default="fallback")
    ap.add_argument("--force-ocr", action="store_true")
    ap.add_argument("--max-pdfs", type=int, default=0)
    ap.add_argument("--embed-batch", type=int, default=int(os.getenv("CF_EMBED_MAX_BATCH", "96")))
    ap.add_argument("--ocr-dpi", type=int, default=200)
    ap.add_argument("--ocr-lang", default=os.getenv("PADDLE_LANG", "en"))
    ap.add_argument("--engine", default=os.getenv("OCR_ENGINE", "gemini"),
                    choices=["gemini", "hybrid", "easyocr", "paddleocr"], help="OCR engine")
    args = ap.parse_args()

    try:
        setup_logging(level=os.getenv("LOG_LEVEL", "DEBUG"))
    except Exception:
        pass

    if getattr(args, "omp_threads", 0) and args.omp_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.omp_threads)
    else:
        for _omp_var in ("OMP_THREAD_LIMIT", "KMP_DEVICE_THREAD_LIMIT", "KMP_TEAMS_THREAD_LIMIT"):
            if _omp_var in os.environ:
                os.environ.pop(_omp_var, None)

    acct = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    tok = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
    if not acct or not tok:
        log("ERROR: Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN")
        sys.exit(2)

    root = Path(args.input_dir).resolve()
    export_dir = Path(args.export_dir).resolve(); export_dir.mkdir(parents=True, exist_ok=True)
    export_tmp = export_dir / "_tmp"; export_tmp.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve(); cache_dir.mkdir(parents=True, exist_ok=True)
    persist_dir = Path(args.persist_dir).resolve(); persist_dir.mkdir(parents=True, exist_ok=True)
    billing_file = persist_dir / "billing_state.json"
    seen_index_path = persist_dir / "seen_files.json"
    progress_path = export_dir / "progress_state.json"
    billing = Billing(Path(billing_file))

    # discover PDFs
    pdfs: List[Path] = []
    ignores = {".git", "node_modules", "__pycache__", ".venv", ".idea", ".vscode", "build", "dist"}
    log(f"Scanning directory: {root}")
    for d, dirnames, files in os.walk(root):
        current_dir = Path(d)
        log(f"Scanning directory: {current_dir.name} (contains {len(files)} files, {len(dirnames)} subdirs)")

        # Filter directories but be more permissive
        original_dirnames = dirnames[:]
        dirnames[:] = [x for x in dirnames if x not in ignores and not x.startswith(".")]
        if len(original_dirnames) != len(dirnames):
            filtered = set(original_dirnames) - set(dirnames)
            log(f"Filtered directories: {filtered}")

        # Process files
        for f in files:
            if f.startswith('.') or f.startswith('._'):
                continue
            if f.lower().endswith(".pdf"):
                pdf_path = current_dir / f
                pdfs.append(pdf_path)
                log(f"Found PDF: {pdf_path.name}")

    pdfs.sort()
    log(f"Total PDFs discovered: {len(pdfs)}")
    if args.max_pdfs > 0:
        pdfs = pdfs[:args.max_pdfs]
        log(f"Limited to first {args.max_pdfs} PDFs")
    log(f"Final PDF count: {len(pdfs)}")

    # Chroma
    collection = None; client = None
    if args.with_chroma:
        client = chroma_client(str(persist_dir))
        collection = client.get_or_create_collection(name=args.collection, metadata={"hnsw:space": "cosine"})
        log(f"Chroma collection: {args.collection}")

    # progress & seen
    log("[RESUME] Loading previous progress...")
    prog = load_progress(progress_path)
    files_state = prog.setdefault("files", {})
    existing_files = len(files_state)
    log(f"[RESUME] Found {existing_files} files in progress state")

    try:
        seen = json.loads(seen_index_path.read_text(encoding="utf-8")) if seen_index_path.exists() else {}
        log(f"[RESUME] Found {len(seen)} entries in seen files index")
    except Exception as e:
        seen = {}
        log(f"[RESUME] Error loading seen files index: {e}, starting fresh")

    tasks: List[Path] = []

    for fp in pdfs:
        file_key = str(fp)
        st = fp.stat()
        fh = sha256_file(fp)[:16]

        # Initialize file state if not exists
        if file_key not in files_state:
            files_state[file_key] = {
                "status": "pending",
                "file_size": st.st_size,
                "file_mtime": int(st.st_mtime),
                "discovered_at": now_iso(),
            }
            # Save progress immediately when discovering new files
            save_progress(progress_path, prog)
            log(f"[DISCOVER] New file: {fp.name}")

        # Check if file should be skipped
        try:
            if should_skip(files_state[file_key], st.st_size, int(st.st_mtime)):
                # Update status and save progress
                files_state[file_key]["status"] = "skipped"
                files_state[file_key]["reason"] = "already_processed"
                files_state[file_key]["finished_at"] = now_iso()
                save_progress(progress_path, prog)
                log(f"[RESUME] Skipping already processed: {fp.name}")
                continue
        except Exception as e:
            log(f"[WARN] Error checking skip status for {fp.name}: {e}")

        # Check for file duplicates
        if fh in seen and seen[fh] != file_key:
            files_state[file_key]["status"] = "skipped"
            files_state[file_key]["reason"] = "file_duplicate"
            files_state[file_key]["duplicate_of"] = seen[fh]
            files_state[file_key]["finished_at"] = now_iso()
            save_progress(progress_path, prog)
            log(f"[DUPLICATE] Skipping duplicate: {fp.name} (same as {Path(seen[fh]).name})")
            continue

        # Add to processing queue
        seen[fh] = file_key
        try:
            seen_index_path.write_text(json.dumps(seen, indent=2), encoding="utf-8")
        except Exception as e:
            log(f"[WARN] Failed to update seen files index: {e}")

        tasks.append(fp)
        log(f"[QUEUE] Added to processing queue: {fp.name}")

    save_progress(progress_path, prog)

    # Resume summary
    total_files = len(pdfs)
    queued_files = len(tasks)
    skipped_files = total_files - queued_files
    log(f"[RESUME] Summary: {total_files} total files, {queued_files} queued for processing, {skipped_files} skipped (already processed)")
    if tasks:
        log(f"[RESUME] Files to process: {[fp.name for fp in tasks[:5]]}" + ("..." if len(tasks) > 5 else ""))

    # Pre-warm OCR
    if PADDLE_AVAILABLE:
        try:
            log(f"[WARMUP] Preloading PaddleOCR models (lang={args.ocr_lang})")
            _ = get_paddle_ocr(args.ocr_lang)
        except Exception as e:
            log(f"[WARN] OCR warmup failed: {e}")

    def archive_tmp(tmp: Path) -> Path:
        final = export_dir / tmp.name
        if final.exists():
            final.unlink()
        try:
            safe_file_replace(tmp, final)
        except Exception as e:
            log(f"[WARN] Failed to archive {tmp.name}, using fallback copy: {e}")
            final.write_bytes(tmp.read_bytes())
            try:
                tmp.unlink()
            except Exception:
                pass
        return final

    processed = 0
    if args.workers == 1:
        for fp in tasks:
            files_state[str(fp)]["status"] = "in_progress"
            files_state[str(fp)]["started_at"] = now_iso()
            save_progress(progress_path, prog)
            log(f"[START] Processing {fp.name} (single-threaded mode)")

            try:
                log(f"[PROCESS] Starting {fp.name} with {args.timeout}s timeout")
                start_time = time.time()
                from concurrent.futures import ProcessPoolExecutor as _PPE
                ex1 = _PPE(max_workers=1)
                fut = ex1.submit(
                    process_one,
                    str(fp), str(root), str(export_tmp), str(cache_dir),
                    acct, tok, str(billing_file), args.embed_batch,
                    args.force_ocr, args.engine, args.ocr_dpi, args.ocr_lang,
                )
                try:
                    res = fut.result(timeout=max(1, int(args.timeout)))
                except TimeoutError:
                    ex1.shutdown(wait=False, cancel_futures=True)
                    elapsed = time.time() - start_time
                    log(f"[ERROR] Timeout after {elapsed:.1f}s on {fp.name}")
                    res = {"error": "timeout", "file": str(fp)}
                else:
                    ex1.shutdown(wait=True, cancel_futures=False)
                    elapsed = time.time() - start_time
                    log(f"[PROCESS] Completed {fp.name} in {elapsed:.1f}s")
            except KeyboardInterrupt:
                log(f"[INTERRUPT] Processing interrupted for {fp.name}")
                sys.exit(0)
            except Exception as e:
                elapsed = time.time() - start_time if 'start_time' in locals() else 0
                log(f"[ERROR] Exception in {fp.name} after {elapsed:.1f}s: {e}")
                res = {"error": f"exception: {e}", "file": str(fp)}

            processed += 1

            if res.get("error"):
                files_state[str(fp)]["status"] = "failed"
                files_state[str(fp)]["error"] = res["error"]
                files_state[str(fp)]["finished_at"] = now_iso()
                save_progress(progress_path, prog)
                log(f"[FAIL] {fp.name}: {res['error']}")
                continue

            if res.get("skip"):
                files_state[str(fp)].update({"status": "skipped", "reason": res.get("reason", "unknown"),
                                             "finished_at": now_iso()})
                save_progress(progress_path, prog)
                log(f"[SKIP] {fp.name}: {res.get('reason')}")
                continue

            if "total_tokens" in res:
                ftoks, fcost = billing.add(res["file"], res["total_tokens"])
                log(f"[Billing] {Path(res['file']).name}: total tokens={ftoks:,} cost=${fcost:.6f}")

            jsonl_tmp = Path(res["jsonl_tmp"])
            jsonl_final = archive_tmp(jsonl_tmp)

            chroma_done = False
            if args.with_chroma and collection:
                try:
                    added = chroma_upsert_jsonl(jsonl_final, collection, client, batch=64)
                    chroma_done = added > 0
                    log(f"[Chroma] {fp.name}: +{added} vectors")
                except Exception as e:
                    log(f"[ERROR] ChromaDB failed for {fp.name}: {e}")
                    chroma_done = False

            files_state[str(fp)].update({
                "status": "completed",
                "jsonl_name": jsonl_final.name,
                "jsonl_archived": True,
                "chroma_upserted": chroma_done,
                "chunks": res.get("chunks", 0),
                "duplicates": res.get("dups", 0),
                "finished_at": now_iso(),
            })
            save_progress(progress_path, prog)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_worker_init_for_ocr,
                                 initargs=(args.ocr_lang,)) as ex:
            fut_map = {
                ex.submit(
                    process_one, str(fp), str(root), str(export_tmp), str(cache_dir),
                    acct, tok, str(billing_file), args.embed_batch,
                    args.force_ocr, args.engine, args.ocr_dpi, args.ocr_lang,
                ): fp for fp in tasks
            }
            for fut in as_completed(fut_map):
                fp = fut_map[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"error": f"exception: {e}", "file": str(fp)}
                processed += 1
                if res.get("error"):
                    files_state[str(fp)] = {
                        **files_state.get(str(fp), {}),
                        "status": "failed",
                        "error": res["error"],
                        "finished_at": now_iso(),
                    }
                    save_progress(progress_path, prog)
                    log(f"[FAIL] {fp.name}: {res['error']}")
                    continue
                if res.get("skip"):
                    files_state[str(fp)] = {
                        **files_state.get(str(fp), {}),
                        "status": "skipped",
                        "reason": res.get("reason", "unknown"),
                        "finished_at": now_iso(),
                    }
                    save_progress(progress_path, prog)
                    log(f"[SKIP] {fp.name}: {res.get('reason')}")
                    continue

                if "total_tokens" in res:
                    ftoks, fcost = billing.add(res["file"], res["total_tokens"])
                    log(f"[Billing] {Path(res['file']).name}: total tokens={ftoks:,} cost=${fcost:.6f}")

                jsonl_final = archive_tmp(Path(res["jsonl_tmp"]))
                chroma_done = False
                if args.with_chroma and collection:
                    try:
                        added = chroma_upsert_jsonl(jsonl_final, collection, client, batch=64)
                        chroma_done = added > 0
                        log(f"[Chroma] {fp.name}: +{added} vectors")
                    except Exception as e:
                        log(f"[ERROR] ChromaDB failed for {fp.name}: {e}")
                        chroma_done = False

                base = files_state.get(str(fp), {})
                base.update({
                    "status": "completed", "jsonl_name": jsonl_final.name,
                    "jsonl_archived": True, "chroma_upserted": chroma_done,
                    "chunks": res.get("chunks", 0), "duplicates": res.get("dups", 0),
                    "finished_at": now_iso(),
                })
                files_state[str(fp)] = base
                save_progress(progress_path, prog)

    log(f"Done. Processed this run: {processed}")

if __name__ == "__main__":
    main()
