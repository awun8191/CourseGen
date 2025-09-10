#!/usr/bin/env python3
"""
Streamlined: PDF -> (OCR via ocr_engine) -> chunk -> dedupe -> BGE-M3 (Cloudflare)
-> per-file JSONL -> immediate Chroma upsert -> real-time billing -> resume.

Major fixes / features:
- Uses your central OCR engine (`src.services.RAG.ocr_engine.ocr_pdf`) so you can pick
  Gemini / EasyOCR / Paddle / Hybrid via --engine or OCR_ENGINE env.
- Fixed PaddleOCR configuration conflict (use_angle_cls vs use_textline_orientation) by
  letting the ocr_engine own these details.
- No hardcoded Cloudflare credentials; strictly require env vars.
- Proper EasyOCR fallback is handled inside the shared OCR engine.
- Retry/backoff for Cloudflare API (429/5xx/network).
- Saner OCR image-size control lives in the OCR engine; DPI ladder via --dpi.
- Correct cache keying using file sha256 + mtime; avoids caching near-empty text.
- Resume: progress files per-PDF; safe temp writes with atomic replace.
- Safer Chroma upsert and metadata sanitization.
- Multiprocessing for multiple PDFs with worker warmup.

Env:
  CLOUDFLARE_ACCOUNT_ID          (required)
  CLOUDFLARE_API_TOKEN           (required)
  CF_PRICE_PER_M_TOKENS=0.012    # USD per 1M input tokens (BGE-M3 input price)
  CF_EMBED_MAX_BATCH=96
  OMP_NUM_THREADS=4
  BILLING_ENABLED=1
  PADDLE_LANG=en                  # optional; e.g., en, fr, de, ar, hi
  EASYOCR_GPU=0                   # optional; set 1 to enable GPU for EasyOCR
  OCR_ENGINE=gemini|hybrid|easyocr|paddleocr  # default overridden by CLI --engine
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
try:
    from concurrent.futures.process import BrokenProcessPool  # type: ignore
except Exception:
    class BrokenProcessPool(Exception):
        pass

# ---- Make repository root importable so "from src...." works when run directly ----
try:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    src_dir = repo_root / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
except Exception:
    pass

# ---- External deps used here ----
import requests

# ---- Project modules ----
from services.RAG.log_utils import setup_logging, snapshot
from services.RAG.billing import Billing
from services.RAG.ocr_engine import ocr_pdf
from services.RAG.chroma_store import chroma_upsert_jsonl
from services.RAG.progress_store import (
    load_progress,
    save_progress,
    safe_file_replace,
    should_skip,
)
from services.RAG.path_meta import parse_path_meta
from services.RAG.cache_utils import sha256_file, _key_ocr_page
from services.RAG.chunking import chunk, dedupe, sha1_text

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Small pretty logging helpers (terminal color)
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
    "[TEXT-SOURCE]": (ANSI["blue"] + ANSI["bold"], True),
    "[EXTRACT]": (ANSI["blue"], False),
    "[CHUNK]": (ANSI["magenta"], False),
    "[EMBED]": (ANSI["magenta"] + ANSI["bold"], True),
    "[EMBED-SAMPLE]": (ANSI["magenta"], False),
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

# ------------------------------------------------------------------------------------
# Token counting (tiktoken if available; fallback ≈ chars/4)
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

# ------------------------------------------------------------------------------------
# Cloudflare Embeddings (BGE-M3) with retry/backoff
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

# ------------------------------------------------------------------------------------
# OCR wrapper (delegates to shared ocr_engine)
def extract_text(
    path: str,
    ocr_dpi: int,
    ocr_lang: str,
    engine: str,
    cache_dir: str | None = None,
    file_sha: str | None = None,
) -> Tuple[str, Dict[str, Any]]:
    res = ocr_pdf(
        pdf_path=path,
        lang=ocr_lang,
        dpi=ocr_dpi,
        engine=engine,
        cache_dir=cache_dir,
        file_hash=file_sha,
    )
    return res.text, res.meta

# ------------------------------------------------------------------------------------
# Worker warmup: limit OpenCV threads but avoid heavy OCR initialization to save memory
def _worker_init_for_ocr(_lang: str, _use_gpu: bool = False) -> None:
    """Lightweight worker setup: limit OpenCV threads only when needed."""
    if os.getenv("OCR_ENGINE", "gemini") == "gemini":
        return
    try:
        import cv2  # type: ignore
        cv2.setNumThreads(int(os.getenv("CV2_NUM_THREADS", "1")))
        # Wire cv2 into ocr_engine lazily
        from services.RAG import ocr_engine as _ocr_engine
        _ocr_engine.cv2 = cv2  # type: ignore[attr-defined]
    except Exception:
        pass

# ------------------------------------------------------------------------------------
# Per-file processing

def sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure metadata is JSON-serializable and small."""
    def _clean(v):
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, (list, tuple)):
            return [_clean(x) for x in v][:64]
        if isinstance(v, dict):
            return {str(k): _clean(vv) for k, vv in v.items()}
        return str(v)
    return _clean(meta) if isinstance(meta, dict) else {}

def _file_id_for_chunk(file_sha: str, idx: int, text: str) -> str:
    # stable, content-aware id (sha1 over text to avoid dup collisions)
    return f"{file_sha}:{idx}:{sha1_text(text)}"

def _write_jsonl_atomic(lines: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in lines:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    safe_file_replace(tmp, out_path)

def process_one(pdf_path: str,
                collection: str,
                export_dir: str,
                cache_dir: str,
                cf_acct: str,
                cf_token: str,
                billing_file: str,
                embed_batch: int,
                ocr_engine: str,
                ocr_dpi: int,
                ocr_lang: str,
                skip_embed: bool = False) -> Dict[str, Any]:

    path = Path(pdf_path)
    rel = str(path.resolve())
    try:
        uri = path.as_uri()
    except Exception:
        uri = rel

    try:
        st0 = path.stat()
        size0 = f"{st0.st_size:,}B"
        mtime0 = int(st0.st_mtime)
    except Exception:
        size0 = "?"
        mtime0 = 0

    log(f"[START] Processing {path.name}  |  size={size0}  mtime={mtime0}  link={uri}")

    # Progress files
    export_dir_p = Path(export_dir)
    export_dir_p.mkdir(parents=True, exist_ok=True)
    progress_path = export_dir_p / f"{path.stem}.progress.json"
    jsonl_path = export_dir_p / f"{path.stem}.jsonl"

    # Load/initialize progress
    prog = load_progress(progress_path)
    if should_skip(prog, st0.st_size, mtime0):
        log(f"[SKIP] Already processed and up-to-date: {path.name}")
        return {"file": rel, "status": "skipped"}

    # File-level metadata
    meta_path = parse_path_meta(path)
    log(f"[META] {path.name}: GROUP_KEY={meta_path.get('GROUP_KEY','')}")
    file_sha = sha256_file(path)

    # ---------- OCR ----------
    log(f"[OCR] Engine={ocr_engine} DPI={ocr_dpi} Lang={ocr_lang}")
    t0 = time.time()
    text, ocr_meta = extract_text(
        rel, ocr_dpi, ocr_lang, ocr_engine, cache_dir, file_sha
    )
    t1 = time.time()
    if not text or len(text.strip()) < 10:
        save_progress(
            progress_path,
            {
                "file": rel,
                "file_size": st0.st_size,
                "file_mtime": mtime0,
                "status": "error",
                "error": "empty_ocr",
            },
        )
        raise RuntimeError(f"OCR yielded too little text for {path.name}")
    ocr_secs = t1 - t0
    log(f"[TEXT] {path.name}: chars={len(text):,}  time={ocr_secs:.1f}s  preview='{snapshot(text, 160)}'")

    # ---------- Chunk & dedupe ----------
    log("[CHUNK] Chunking + dedup")
    raw_chunks = chunk(text)
    deduped, _ = dedupe(raw_chunks)
    if not deduped:
        save_progress(
            progress_path,
            {
                "file": rel,
                "file_size": st0.st_size,
                "file_mtime": mtime0,
                "status": "error",
                "error": "no_chunks",
            },
        )
        raise RuntimeError(f"No chunks after dedupe for {path.name}")
    log(f"[CHUNK] {path.name}: raw={len(raw_chunks)}  unique={len(deduped)}")

    # ---------- Build JSONL records ----------
    base_meta = {
        "source_path": rel,
        "source_uri": uri,
        "file_sha256": file_sha,
        "mtime": mtime0,
        "collection": collection,
        "ocr": sanitize_metadata(ocr_meta),
        "path_meta": sanitize_metadata(meta_path),
        "created_at": now_iso(),
    }
    jsonl_lines: List[Dict[str, Any]] = []
    for idx, ch in enumerate(deduped):
        cid = _file_id_for_chunk(file_sha, idx, ch)
        rec = {
            "id": cid,
            "text": ch,
            "metadata": {**base_meta, "chunk_index": idx, "text_sha1": sha1_text(ch), "length": len(ch)},
        }
        jsonl_lines.append(rec)

    # Write per-file JSONL atomically
    _write_jsonl_atomic(jsonl_lines, jsonl_path)
    log(f"[EXTRACT] Wrote JSONL: {jsonl_path}  lines={len(jsonl_lines)}")

    # ---------- Embeddings (Cloudflare) ----------
    total_tokens = 0
    if not skip_embed:
        log(f"[EMBED] Cloudflare BGE-M3  batch≤{embed_batch}")
        acct = cf_acct or os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        tok = cf_token or os.getenv("CLOUDFLARE_API_TOKEN", "")
        if not acct or not tok:
            save_progress(
                progress_path,
                {
                    "file": rel,
                    "file_size": st0.st_size,
                    "file_mtime": mtime0,
                    "status": "error",
                    "error": "missing_cloudflare_creds",
                },
            )
            raise RuntimeError("Missing CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN")

        with CFEmbeddings(acct, tok, batch_max=embed_batch) as cf:
            # stream-embed and upsert per-batch to reduce RAM
            texts = [rec["text"] for rec in jsonl_lines]
            ids = [rec["id"] for rec in jsonl_lines]
            metas = [rec["metadata"] for rec in jsonl_lines]

            # We'll accumulate vectors incrementally and upsert to Chroma in slices.
            # To avoid mismatches, we upsert exactly in the same batch partitions.
            upsert_count = 0
            for (emb_batch, tok_batch), start in zip(cf.embed_iter(texts, batch_size=embed_batch),
                                                     range(0, len(texts), embed_batch)):
                end = min(start + embed_batch, len(texts))
                batch_ids = ids[start:end]
                batch_texts = texts[start:end]
                batch_metas = metas[start:end]

                # Upsert this batch into Chroma
                chroma_upsert_jsonl(
                    None,  # <- we can upsert directly with in-memory arrays (function supports path OR arrays)
                    collection_name=collection,
                    ids=batch_ids,
                    embeddings=emb_batch,
                    metadatas=batch_metas,
                    documents=batch_texts,
                )
                upsert_count += len(batch_ids)
                total_tokens += tok_batch
                if upsert_count <= 3:
                    # sample logs
                    log(f"[EMBED-SAMPLE] ids[{batch_ids[0]}], dims={len(emb_batch[0]) if emb_batch else 'n/a'}  toks≈{tok_batch}")

        # Update billing
        try:
            price_per_m = float(os.getenv("CF_PRICE_PER_M_TOKENS", "0.012"))
            usd = (total_tokens / 1_000_000.0) * price_per_m
            if int(os.getenv("BILLING_ENABLED", "1")):
                Billing.add_cost("cloudflare_embeddings", usd, meta={"tokens": total_tokens, "price_per_m": price_per_m, "file": rel})
            log(f"[Billing] Cloudflare embeddings: tokens≈{total_tokens:,}  cost≈${usd:.4f}")
        except Exception as e:
            log(f"[WARN] Billing update failed: {e}")
    else:
        log("[EMBED] Skipped by flag --skip-embed")
        # Even if skipping embedding, ensure JSONL still gets upserted as documents with no vectors, if desired.
        # Here we choose not to upsert without vectors to avoid empty collections.

    # ---------- Final upsert from file (optional path-based re-upsert) ----------
    # If you prefer a single-shot upsert from the JSONL file (instead of per-batch above),
    # uncomment below and remove the streaming upserts in the loop.
    # chroma_upsert_jsonl(jsonl_path, collection_name=collection)

    save_progress(
        progress_path,
        {
            "file": rel,
            "file_size": st0.st_size,
            "file_mtime": mtime0,
            "status": "done",
            "jsonl": str(jsonl_path),
            "tokens": total_tokens,
        },
    )

    # Clear per-page OCR cache once fully processed
    try:
        cdir = Path(cache_dir)
        pages = int(ocr_meta.get("pages", 0))
        for page in range(1, pages + 1):
            key = _key_ocr_page(file_sha, ocr_lang, ocr_dpi, ocr_engine, page)
            f = cdir / key
            if f.exists():
                f.unlink()
    except Exception as e:
        log(f"[CACHE] cleanup failed for {path.name}: {e}")
    log(f"[SUCCESS] {path.name}  chunks={len(jsonl_lines)}  tokens≈{total_tokens:,}")
    return {"file": rel, "status": "done", "chunks": len(jsonl_lines), "tokens": total_tokens, "jsonl": str(jsonl_path)}

# ------------------------------------------------------------------------------------
# CLI + Orchestration

def discover_pdfs(inputs: List[str]) -> List[str]:
    found: List[str] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            found += [str(x) for x in sorted(p.rglob("*.pdf"))]
        elif p.suffix.lower() == ".pdf":
            found.append(str(p))
    # de-dup while preserving order
    seen = set()
    uniq: List[str] = []
    for f in found:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq

def main():
    parser = argparse.ArgumentParser(description="PDF → OCR → chunk → dedupe → BGE-M3 → JSONL → Chroma (resume + billing)")
    parser.add_argument("inputs", nargs="+", help="PDF files or folders")
    parser.add_argument("-c", "--collection", default="default", help="Chroma collection name")
    parser.add_argument("-o", "--out", default="OUTPUT_DATA2/progress_report", help="Output dir for per-file JSONL and progress")
    parser.add_argument("--cache", default="OUTPUT_DATA2/cache", help="Cache dir for OCR (default: <out>/cache)")
    parser.add_argument("--persist-dir", default="OUTPUT_DATA2/emdeddings", help="Chroma persist dir (default: <out>/chroma)")
    parser.add_argument("--engine", default=os.getenv("OCR_ENGINE", "gemini"),
                        choices=["gemini", "hybrid", "easyocr", "paddleocr"], help="OCR engine (delegated to ocr_engine)")
    parser.add_argument("--dpi", type=int, default=300, help="OCR DPI for rasterization")
    parser.add_argument("--lang", default=os.getenv("PADDLE_LANG", "en"), help="OCR language hint (ocr_engine may ignore)")
    parser.add_argument("--cf-batch", type=int, default=int(os.getenv("CF_EMBED_MAX_BATCH", "96")), help="Max batch size for CF embeddings")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(3, os.cpu_count() or 1),
        help="Parallel worker processes (default: 3 or CPU count, whichever is lower)",
    )
    parser.add_argument("--skip-embed", action="store_true", help="Run OCR + JSONL only; skip embeddings/upsert")
    parser.add_argument("--timeout", type=int, default=0, help="Per-file timeout in seconds (0=disable)")
    args = parser.parse_args()
    # allow caller to override worker count; default chosen above

    setup_logging()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OCR_ENGINE", args.engine)
    # Ensure load-balanced Gemini OCR with flash-lite model and 3-page batching
    gem_svc = Path(__file__).resolve().parents[1] / "Gemini" / "gemini_service.py"
    os.environ.setdefault("GEMINI_SERVICE_PATH", str(gem_svc))
    os.environ.setdefault("GEMINI_MODEL", "gemini-2.5-flash-lite")
    os.environ.setdefault("OCR_GEMINI_PAGE_BATCH", "3")

    # Credentials
    cf_acct = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
    cf_token = os.getenv("CLOUDFLARE_API_TOKEN", "")
    if not args.skip_embed and (not cf_acct or not cf_token):
        log("[ERROR] CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are required (or use --skip-embed)")
        sys.exit(2)

    pdfs = discover_pdfs(args.inputs)
    if not pdfs:
        log("[ERROR] No PDFs found in inputs.")
        sys.exit(1)

    # Make output dirs
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Graceful Ctrl+C across pool
    interrupted = {"flag": False}
    def _sigint(_a, _b):
        interrupted["flag"] = True
        log("[INTERRUPT] Ctrl+C received; finishing current tasks...")
    signal.signal(signal.SIGINT, _sigint)

    # Serial path if workers==1 (easier to debug)
    if args.workers == 1:
        results = []
        for pdf in pdfs:
            if interrupted["flag"]:
                break
            try:
                r = process_one(pdf, args.collection, str(out_dir), str(cache_dir),
                                cf_acct, cf_token, str(out_dir / "billing.jsonl"),
                                args.cf_batch, args.engine, args.dpi, args.lang,
                                skip_embed=args.skip_embed)
                results.append(r)
            except Exception as e:
                log(f"[FAIL] {pdf}: {e}")
        log(f"[DONE] files={len(results)}")
        return

    # Parallel execution
    results = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_worker_init_for_ocr,
                             initargs=(args.lang, bool(int(os.getenv("PADDLE_GPU", "0"))))) as ex:
        futs = {}
        for pdf in pdfs:
            fut = ex.submit(
                process_one, pdf, args.collection, str(out_dir), str(cache_dir),
                cf_acct, cf_token, str(out_dir / "billing.jsonl"),
                args.cf_batch, args.engine, args.dpi, args.lang, args.skip_embed
            )
            futs[fut] = pdf

        for fut in as_completed(futs):
            pdf = futs[fut]
            if interrupted["flag"]:
                break
            try:
                if args.timeout and hasattr(fut, "result"):
                    r = fut.result(timeout=args.timeout)
                else:
                    r = fut.result()
                results.append(r)
            except TimeoutError:
                log(f"[FAIL] Timeout: {pdf}")
            except BrokenProcessPool:
                log("[ERROR] Worker pool broke; aborting.")
                break
            except KeyboardInterrupt:
                log("[INTERRUPT] KeyboardInterrupt during futures. Stopping…")
                break
            except Exception as e:
                log(f"[FAIL] {pdf}: {e}")

    log(f"[DONE] files={len(results)} processed")
    # Optionally, summarize tokens/costs here.

if __name__ == "__main__":
    main()
