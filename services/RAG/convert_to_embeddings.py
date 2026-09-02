#!/usr/bin/env python3
"""
Streamlined: PDF -> OCR -> chunk -> dedupe -> BGE-M3 (Cloudflare)
-> per-file JSONL -> immediate Chroma upsert -> real-time billing -> resume.

Major fixes:
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
  CF_EMBED_MAX_BATCH=64
  CF_EMBED_MAX_TOKENS=7500       # Hard cap on tokens per request batch
  CF_EMBED_MIN_BATCH=8
  OMP_NUM_THREADS=4
  BILLING_ENABLED=1
  OCR_LANG=en                    # optional; e.g., en, fr, de, ar, hi
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
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import multiprocessing as mp

import fitz  # PyMuPDF
from PIL import Image
import requests

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

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

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
                if r.status_code in (408, 429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{r.status_code} {r.text[:200]}")
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt == self.retry.tries:
                    raise
                time.sleep(min(self.retry.max_sleep, sleep))
                sleep *= self.retry.backoff
        raise RuntimeError("unhandled retry loop")

    def embed_iter(self, texts: List[str], batch_size: int):
        """Yield (embeddings_for_batch, tokens_for_batch) with adaptive batching.

        On transient CF errors (e.g., 408/429/5xx) this halves the batch size and retries
        the same window, but never below a minimum floor (default 8).
        Additionally enforces a per-request token cap (CF_EMBED_MAX_TOKENS, default 7500).
        """
        n = len(texts)
        if n == 0:
            return
        # Enforce floor via env CF_EMBED_MIN_BATCH (default 8)
        try:
            min_floor = max(1, int(os.getenv("CF_EMBED_MIN_BATCH", "8")))
        except Exception:
            min_floor = 8
        try:
            max_tokens = max(512, int(os.getenv("CF_EMBED_MAX_TOKENS", "7500")))
        except Exception:
            max_tokens = 7500
        max_bsz = min(max(min_floor, batch_size), self.batch_max)
        i = 0
        cur_bsz = max_bsz
        while i < n:
            # Greedily pack up to cur_bsz items but cap total tokens to max_tokens
            j = i
            packed: List[str] = []
            token_sum = 0
            while j < n and len(packed) < cur_bsz:
                t = texts[j]
                t_tokens = self.counter.count_batch([t])
                if not packed:
                    # Always include at least one item
                    packed.append(t)
                    token_sum += t_tokens
                    j += 1
                    continue
                if token_sum + t_tokens <= max_tokens:
                    packed.append(t)
                    token_sum += t_tokens
                    j += 1
                else:
                    break
            sub = packed
            payload = {"text": sub, "truncate_inputs": True}
            try:
                js = self._post_embed(payload)
                data = js.get("result", {}).get("data")
                if not isinstance(data, list):
                    raise RuntimeError(f"Bad embedding response: {str(js)[:200]}")
                tokens = token_sum if token_sum > 0 else self.counter.count_batch(sub)
                yield data, tokens
                i += len(sub)
                # If we had previously reduced batch size due to errors, gradually ramp back up
                if cur_bsz < max_bsz:
                    cur_bsz = min(max_bsz, cur_bsz * 2)
            except Exception as e:
                # Reduce batch size on failure; respect minimum floor
                if cur_bsz > min_floor:
                    new_bsz = max(min_floor, cur_bsz // 2)
                    log(f"[EMBED] Batch failed ({type(e).__name__}); reducing batch {cur_bsz}->{new_bsz} (min={min_floor}) and retrying")
                    cur_bsz = new_bsz
                    continue
                # cur_bsz <= min_floor -> give up
                raise

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

def _pixmap_to_numpy(pix: fitz.Pixmap) -> 'np.ndarray':
    if not OPENCV_AVAILABLE:
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return np.array(img) if 'np' in globals() else __import__('numpy').array(img)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return arr


def extract_text(pdf_path, cache_dir, force_ocr, ocr_engine, ocr_dpi, ocr_lang):
    from services.RAG.ocr_engine import ocr_pdf

    valid_engines = {"gemini", "hybrid", "easyocr"}

    # If force_ocr is enabled, skip text extraction check
    if force_ocr:
        engine = ocr_engine if ocr_engine in valid_engines else "hybrid"
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
    engine = ocr_engine if ocr_engine in valid_engines else 'hybrid'
    result = ocr_pdf(pdf_path, lang=ocr_lang, dpi=ocr_dpi, engine=engine)
    return result.text

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

    try:
        log(f"[TEXT] Extracting text from {path.name}")
        text = extract_text(path, Path(cache_dir), force_ocr, ocr_engine, ocr_dpi=ocr_dpi, ocr_lang=ocr_lang)
        if not text.strip():
            return {"file": rel, "skip": True, "reason": "empty_text"}
        log(f"[EXTRACT] {path.name} chars={len(text)} snapshot='{snapshot(text)}'")

        log(f"[CHUNK] Chunking text for {path.name}")
        # Use 2-paragraph chunks with 2-sentence overlap for better continuity
        chunks_all = chunk(text, paras_per_chunk=2, paras_overlap=0, sentence_overlap=2)
        uniq, dup_map = dedupe(chunks_all)
        if not uniq:
            return {"file": rel, "skip": True, "reason": "no_chunks"}
        log(f"[CHUNK] {path.name} uniq_chunks={len(uniq)} first_chunk='{snapshot(uniq[0]) if uniq else ''}'")

        # Compute doc hash then free raw text to reduce peak RAM
        doc_hash_for_stream = sha1_text(text)
        try:
            del text
        except Exception:
            pass

        # Cloudflare embeddings (streamed to JSONL to reduce RAM)
        log(f"[EMBED] Creating Cloudflare BGE-M3 embeddings for {path.name} ({len(uniq)} chunks) [streaming]")
        cf = None
        total_tokens = 0
        try:
            cf = CFEmbeddings(cf_acct, cf_token, int(os.getenv("CF_EMBED_MAX_BATCH", "64")))
            log(f"[EMBED] Cloudflare client initialized with account: {cf_acct[:8]}...")

            # Prepare JSONL for streaming writes with resume support
            tmp_dir = Path(export_tmp); tmp_dir.mkdir(parents=True, exist_ok=True)
            export_dir_guess = tmp_dir.parent
            group = meta_path["GROUP_KEY"]
            jsonl_name = f"{re.sub(r'[^A-Za-z0-9._-]+','_',group)}__{sha1_text(rel)}.jsonl"
            jsonl_tmp = tmp_dir / jsonl_name
            jsonl_final = export_dir_guess / jsonl_name

            # Determine resume state: copy final to tmp if resuming from archived,
            # or continue appending to existing tmp.
            seen_chunk_indexes: set[int] = set()
            if jsonl_final.exists() and not jsonl_tmp.exists():
                try:
                    shutil.copy2(jsonl_final, jsonl_tmp)
                    log(f"[RESUME] Copied archived JSONL to tmp for resume: {jsonl_name}")
                except Exception as e:
                    log(f"[WARN] Failed to copy archived JSONL for resume: {e}")
            # If either tmp or final exists, parse for seen chunk indices
            resume_path = jsonl_tmp if jsonl_tmp.exists() else (jsonl_final if jsonl_final.exists() else None)
            if resume_path and resume_path.exists():
                try:
                    with resume_path.open("r", encoding="utf-8") as rf:
                        for line in rf:
                            if not line.strip():
                                continue
                            try:
                                rec = json.loads(line)
                                md = rec.get("metadata", {}) or {}
                                idx0 = md.get("chunk_index")
                                if isinstance(idx0, int):
                                    seen_chunk_indexes.add(idx0)
                            except Exception:
                                continue
                    if seen_chunk_indexes:
                        log(f"[RESUME] Found {len(seen_chunk_indexes)} embedded chunks already in {resume_path.name}")
                except Exception as e:
                    log(f"[WARN] Failed to read existing JSONL for resume: {e}")

            file_hash = sha256_file(path)[:16]
            st = path.stat()
            doc_hash = doc_hash_for_stream

            # Build index list of unique chunk positions to align embeddings without storing them
            total = len(chunks_all)
            uniq_indices = [i for i in range(total) if i not in dup_map]

            # Filter for remaining unique chunks if resuming
            if seen_chunk_indexes:
                remaining_pairs = [(i, chunks_all[i]) for i in uniq_indices if i not in seen_chunk_indexes]
                if remaining_pairs:
                    rem_indices, rem_texts = zip(*remaining_pairs)
                    uniq_indices_remaining = list(rem_indices)
                    uniq_remaining = list(rem_texts)
                else:
                    uniq_indices_remaining = []
                    uniq_remaining = []
            else:
                uniq_indices_remaining = list(uniq_indices)
                uniq_remaining = list(uniq)

            # Open in append mode to preserve any previous progress
            with jsonl_tmp.open("a", encoding="utf-8") as out:
                k = 0
                batch_count = 0
                # BGE-M3 benefits from instruction prefixes; enable by default for documents
                use_bge_prefix = os.getenv("BGE_USE_PREFIX", "1") not in ("0", "false", "False")
                embed_source = [
                    (f"passage: {t}" if use_bge_prefix and not str(t).startswith("passage:") else t)
                    for t in uniq_remaining
                ]
                for emb_batch, tok_batch in cf.embed_iter(embed_source, batch_size=embed_batch):
                    batch_count += 1
                    total_tokens += tok_batch
                    for j, vec in enumerate(emb_batch):
                        idx = uniq_indices_remaining[k + j]
                        ch = chunks_all[idx]
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
                            "embedding": vec,
                            "embedding_type": "cloudflare-bge-m3"
                        }) + "\n")
                    log(f"[EMBED] Processed batch {batch_count}: {len(emb_batch)} vectors, {tok_batch} tokens")
                    k += len(emb_batch)

                for idx, (orig_idx, orig_h) in dup_map.items():
                    if idx in seen_chunk_indexes:
                        continue
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

            if len(uniq_remaining) != 0 and k != len(uniq_remaining):
                return {"file": rel, "error": f"embedding_mismatch: wrote {k} vectors, expected {len(uniq_remaining)}"}
            already = len(seen_chunk_indexes)
            log(f"[EMBED] Successfully streamed {k} embeddings (+{already} existing), total tokens this run: {total_tokens}")

        except Exception as e:
            # Keep partial file for resume; do not delete
            log(f"[ERROR] Cloudflare embedding failed: {e}")
            return {"file": rel, "error": f"cloudflare_embedding_error: {e}"}
        finally:
            if cf is not None:
                cf.close()

    except Exception as e:
        log(f"[ERROR] Processing failed for {path.name}: {e}")
        return {"file": rel, "error": f"processing_error: {e}"}

    # jsonl_tmp and jsonl_name already created during streaming
    return {"file": rel, "jsonl_tmp": str(jsonl_tmp),
            "chunks": len(uniq), "dups": len(dup_map), "jsonl_name": jsonl_name,
            "total_tokens": total_tokens}

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

    # Use a single, repo-root anchored default to avoid duplicate output_data trees
    default_output_root = Path(os.getenv("COURSEGEN_OUTPUT_ROOT", str((repo_root / "output_data").resolve())))
    ap.add_argument("--export-dir", default=str(default_output_root / "progress_report"))
    ap.add_argument("--cache-dir", default=str(default_output_root / "cache"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--omp-threads", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=1800, help="Timeout per file in seconds")
    ap.add_argument("--with-chroma", dest="with_chroma", action="store_true", default=True)
    ap.add_argument("--no-chroma", dest="with_chroma", action="store_false")
    ap.add_argument("-c", "--collection", default="course_embeddings")
    ap.add_argument("--persist-dir", default=str(default_output_root / "vector_database"))
    ap.add_argument("--ocr-on-missing", choices=["fallback", "error", "skip"], default="fallback")
    ap.add_argument("--force-ocr", action="store_true")
    ap.add_argument("--max-pdfs", type=int, default=0)
    ap.add_argument("--embed-batch", type=int, default=int(os.getenv("CF_EMBED_MAX_BATCH", "16")))
    ap.add_argument("--ocr-dpi", type=int, default=200)

    default_lang = (
        os.getenv("OCR_LANG")
        or os.getenv("EASYOCR_LANG")
        or os.getenv("PADDLE_LANG")
        or "en"
    )
    ap.add_argument("--ocr-lang", default=default_lang)

    engine_default = os.getenv("OCR_ENGINE", "gemini")
    if engine_default not in {"gemini", "hybrid", "easyocr"}:
        engine_default = "gemini"
    ap.add_argument(
        "--engine",
        default=engine_default,
        choices=["gemini", "hybrid", "easyocr"],
        help="OCR engine",
    )
    # Gemini fallback controls
    fallback_default = os.getenv("OCR_GEMINI_FALLBACK_ENGINE", "easyocr").lower()
    if fallback_default not in {"easyocr", "hybrid"}:
        fallback_default = "easyocr"
    ap.add_argument(
        "--ocr-fallback-engine",
        choices=["easyocr", "hybrid"],
        default=fallback_default,
        help="When using --engine gemini, choose local fallback engine if a page returns empty text.",
    )
    ap.add_argument("--no-gemini-fallback", action="store_true",
                    help="Disable auto local OCR fallback when Gemini returns empty text. If disabled, empty Gemini pages raise an error.")
    ap.add_argument("--memory-limit", type=int, default=0, help="Memory limit in MB (0 = no limit)")
    ap.add_argument("--retry-limit", type=int, default=3, help="Maximum retry attempts per file")
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

    # Force CPU-only paths in worker processes to avoid CUDA init in forks
    # Prevent EasyOCR/PyTorch from probing CUDA by default.
    os.environ.setdefault("EASYOCR_GPU", "0")
    # Hide GPUs entirely to libraries that probe CUDA at import-time
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

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

    log(f"[PATHS] export_dir={export_dir}")
    log(f"[PATHS] cache_dir={cache_dir}")
    log(f"[PATHS] persist_dir={persist_dir}")
    billing_file = persist_dir / "billing_state.json"
    seen_index_path = persist_dir / "seen_files.json"
    progress_path = export_dir / "progress_state.json"
    billing = Billing(Path(billing_file))

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

    collection = None; client = None
    if args.with_chroma:
        client = chroma_client(str(persist_dir))
        collection = client.get_or_create_collection(name=args.collection, metadata={"hnsw:space": "cosine"})
        log(f"Chroma collection: {args.collection}")

    log("[RESUME] Loading previous progress...")
    prog = load_progress(progress_path)
    files_state = prog.setdefault("files", {})
    existing_files = len(files_state)
    log(f"[RESUME] Found {existing_files} files in progress state")

    # Configure Gemini fallback behavior via env for the OCR layer
    if args.no_gemini_fallback:
        os.environ["OCR_GEMINI_AUTOFALLBACK"] = "0"
        log("[OCR] Gemini fallback: disabled")
    else:
        os.environ.setdefault("OCR_GEMINI_AUTOFALLBACK", "1")
        # Persist fallback engine for worker processes
        os.environ["OCR_GEMINI_FALLBACK_ENGINE"] = args.ocr_fallback_engine
        log(f"[OCR] Gemini fallback engine: {args.ocr_fallback_engine}")

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

        try:
            if should_skip(files_state[file_key], st.st_size, int(st.st_mtime)):
                # Do not downgrade a previously completed record to "skipped"
                if files_state[file_key].get("status") != "completed":
                    files_state[file_key]["status"] = "skipped"
                    files_state[file_key]["reason"] = "already_processed"
                    files_state[file_key]["finished_at"] = now_iso()
                # Persist any metadata updates but keep the original completion status intact
                save_progress(progress_path, prog)
                log(f"[RESUME] Skipping already processed: {fp.name}")
                continue
        except Exception as e:
            log(f"[WARN] Error checking skip status for {fp.name}: {e}")

        if fh in seen and seen[fh] != file_key:
            files_state[file_key]["status"] = "skipped"
            files_state[file_key]["reason"] = "file_duplicate"
            files_state[file_key]["duplicate_of"] = seen[fh]
            files_state[file_key]["finished_at"] = now_iso()
            save_progress(progress_path, prog)
            log(f"[DUPLICATE] Skipping duplicate: {fp.name} (same as {Path(seen[fh]).name})")
            continue

        seen[fh] = file_key
        try:
            seen_index_path.write_text(json.dumps(seen, indent=2), encoding="utf-8")
        except Exception as e:
            log(f"[WARN] Failed to update seen files index: {e}")

        tasks.append(fp)
        log(f"[QUEUE] Added to processing queue: {fp.name}")

    save_progress(progress_path, prog)

    total_files = len(pdfs)
    queued_files = len(tasks)
    skipped_files = total_files - queued_files
    log(f"[RESUME] Summary: {total_files} total files, {queued_files} queued for processing, {skipped_files} skipped (already processed)")
    if tasks:
        log(f"[RESUME] Files to process: {[fp.name for fp in tasks[:5]]}" + ("..." if len(tasks) > 5 else ""))

    # If previous runs archived JSONL files but Chroma upsert failed, re-attempt
    # those upserts now without reprocessing the PDFs. This makes resume more robust
    # when only the vector DB step failed.
    if args.with_chroma and collection:
        retry_chroma = 0
        for file_key, rec in list(files_state.items()):
            try:
                if rec.get("jsonl_archived") and not rec.get("chroma_upserted"):
                    name = rec.get("jsonl_name")
                    if not name:
                        continue
                    jsonl_path = (export_dir / name)
                    if not jsonl_path.exists():
                        continue
                    try:
                        added = chroma_upsert_jsonl(jsonl_path, collection, client, batch=64)
                        if added > 0:
                            base = rec.copy()
                            base["chroma_upserted"] = True
                            files_state[file_key] = base
                            save_progress(progress_path, prog)
                            log(f"[Chroma][RESUME] Re-upserted {added} vectors from {name}")
                            retry_chroma += 1
                    except Exception as e:
                        log(f"[ERROR][RESUME] Chroma re-upsert failed for {name}: {e}")
            except Exception:
                pass
        if retry_chroma:
            log(f"[RESUME] Recovered Chroma upserts for {retry_chroma} file(s)")

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
    # Use spawn context to avoid forking CUDA/Torch state into children
    _mp_ctx = mp.get_context("spawn")

    if args.workers == 1:
        # Use a single process pool for all files in single-threaded mode
        # This avoids the overhead of creating/destroying pools for each file
        with ProcessPoolExecutor(max_workers=1, mp_context=_mp_ctx) as ex:
            for fp in tasks:
                files_state[str(fp)]["status"] = "in_progress"
                files_state[str(fp)]["started_at"] = now_iso()
                save_progress(progress_path, prog)
                log(f"[START] Processing {fp.name} (single-threaded mode)")

                try:
                    log(f"[PROCESS] Starting {fp.name} with {args.timeout}s timeout")
                    start_time = time.time()
                    fut = ex.submit(
                        process_one,
                        str(fp), str(root), str(export_tmp), str(cache_dir),
                        acct, tok, str(billing_file), args.embed_batch,
                        args.force_ocr, args.engine, args.ocr_dpi, args.ocr_lang,
                    )
                    try:
                        res = fut.result(timeout=max(1, int(args.timeout)))
                    except TimeoutError:
                        # Cancel the future and shutdown gracefully
                        fut.cancel()
                        elapsed = time.time() - start_time
                        log(f"[ERROR] Timeout after {elapsed:.1f}s on {fp.name}")
                        res = {"error": "timeout", "file": str(fp)}
                    else:
                        elapsed = time.time() - start_time
                        log(f"[PROCESS] Completed {fp.name} in {elapsed:.1f}s")
                except KeyboardInterrupt:
                    log(f"[INTERRUPT] Processing interrupted for {fp.name}")
                    # Cancel all pending futures
                    for pending_fut in ex._futures:  # type: ignore
                        pending_fut.cancel()
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
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=_mp_ctx) as ex:
            fut_map = {}
            for fp in tasks:
                # Mark as in-progress before submitting
                files_state[str(fp)]["status"] = "in_progress"
                files_state[str(fp)]["started_at"] = now_iso()
                save_progress(progress_path, prog)
                fut = ex.submit(
                    process_one, str(fp), str(root), str(export_tmp), str(cache_dir),
                    acct, tok, str(billing_file), args.embed_batch,
                    args.force_ocr, args.engine, args.ocr_dpi, args.ocr_lang,
                )
                fut_map[fut] = fp
            
            # Track processing progress and handle failures gracefully
            completed_count = 0
            failed_count = 0
            skipped_count = 0
            
            for fut in as_completed(fut_map):
                fp = fut_map[fut]
                processed += 1
                completed_count += 1
                
                try:
                    res = fut.result()
                    if res.get("error"):
                        failed_count += 1
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
                        skipped_count += 1
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
                    
                except KeyboardInterrupt:
                    log(f"[INTERRUPT] Processing interrupted, cancelling remaining tasks...")
                    # Cancel all pending futures
                    for pending_fut in fut_map:
                        pending_fut.cancel()
                    sys.exit(0)
                except Exception as e:
                    failed_count += 1
                    log(f"[ERROR] Unexpected error processing {fp.name}: {e}")
                    files_state[str(fp)] = {
                        **files_state.get(str(fp), {}),
                        "status": "failed",
                        "error": f"unexpected_error: {e}",
                        "finished_at": now_iso(),
                    }
                    save_progress(progress_path, prog)
                    
                # Log progress periodically
                if completed_count % 5 == 0:
                    log(f"[PROGRESS] Completed {completed_count}/{len(tasks)} files - Success: {completed_count - failed_count - skipped_count}, Failed: {failed_count}, Skipped: {skipped_count}")

    log(f"Done. Processed this run: {processed}")


if __name__ == "__main__":
    main()





