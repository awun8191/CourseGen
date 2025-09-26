#!/usr/bin/env python3
# chromadb_query.py
# Production-ready Chroma retrieval helper for Engineering Hub
# - Output shape matches generator: id, snippet, meta, score, path, COURSE_FOLDER, chunk_index
# - Cloudflare Workers AI BGE-M3 query embeddings (optional)
# - Deterministic search + temperature sampling search_with_temperature
# - Fallback keyword search, dedupe, similarity normalization
# - Structured logging + env-configurable knobs

from __future__ import annotations

import os
import re
import json
import math
import random
import logging
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import time
from chromadb import PersistentClient
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not available, environment variables must be set manually
    pass

# =========================
# Logging
# =========================
logger = logging.getLogger("chroma_query")
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s"))
logger.addHandler(_handler)
logger.setLevel(os.environ.get("CHROMA_LOG_LVL", "INFO").upper())

# =========================
# Paths / Env knobs
# =========================
try:
    REPO_ROOT = Path(__file__).resolve().parents[3]
except Exception:
    REPO_ROOT = Path.cwd()

# Prefer project-wide config/env names; keep legacy envs as fallback
# Defaults match convert_to_embeddings.py and existing persisted DB
CHROMA_PATH = (
    os.environ.get("COURSEGEN_PERSIST_DIR")
    or os.environ.get("CHROMA_PERSIST_DIR")
    or str((REPO_ROOT / "OUTPUT_DATA2/emdeddings").resolve())
)
# Collection default matches convert_to_embeddings.py
CHROMA_COLLECTION = (
    os.environ.get("COURSEGEN_COLLECTION")
    or os.environ.get("CHROMA_COLLECTION")
    or "course_embeddings"
)

# Distance metric of your Chroma collection: "cosine" | "l2" | "similarity"
CHROMA_DISTANCE_METRIC = os.environ.get("CHROMA_DISTANCE_METRIC", "cosine").lower()

# Import centralized configuration
try:
    from config import load_config
    config = load_config()
    DEFAULT_TOPK = config.chroma_topk
    DEFAULT_FINAL_K = config.chroma_final_k
    DEFAULT_TAU = config.chroma_tau
    DEFAULT_MIN_SIM = config.chroma_min_sim
    DEFAULT_SEED = config.chroma_seed
except ImportError:
    # Fallback to environment variables if centralized config not available
    import os
    DEFAULT_TOPK = int(os.environ.get("CHROMA_TOPK", "50"))       # wide pool
    DEFAULT_FINAL_K = int(os.environ.get("CHROMA_FINAL_K", "8"))  # returned to LLM
    DEFAULT_TAU = float(os.environ.get("CHROMA_TAU", "0.35"))
    DEFAULT_MIN_SIM = float(os.environ.get("CHROMA_MIN_SIM", "0.60"))
    DEFAULT_SEED = os.environ.get("CHROMA_SEED")
    DEFAULT_SEED = int(DEFAULT_SEED) if (DEFAULT_SEED and DEFAULT_SEED.isdigit()) else None

# Cloudflare Workers AI / BGE-M3
CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CF_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
USE_BGE_PREFIXES = os.environ.get("CHROMA_USE_BGE_PREFIXES", "1") in ("1", "true", "True")
OFFLINE = os.environ.get("OFFLINE", "").lower() in ("1", "true", "yes")

# Basic client-side throttling and retry for CF embeddings
EMBED_QPS = float(os.environ.get("CF_EMBED_QPS", "1.0"))  # max queries per second
EMBED_MAX_RETRIES = int(os.environ.get("CF_EMBED_RETRIES", "4"))
EMBED_BACKOFF_BASE = float(os.environ.get("CF_EMBED_BACKOFF_BASE", "0.6"))
EMBED_BACKOFF_CAP = float(os.environ.get("CF_EMBED_BACKOFF_CAP", "6.0"))

# Search-level strictness and retries (default: require embeddings)
REQUIRE_EMBEDDINGS = os.environ.get("CHROMA_REQUIRE_EMBEDDINGS", "1") in ("1", "true", "True")
SEARCH_MAX_RETRIES = int(os.environ.get("CHROMA_SEARCH_RETRIES", "5"))
SEARCH_BACKOFF_BASE = float(os.environ.get("CHROMA_SEARCH_BACKOFF_BASE", "1.0"))
SEARCH_BACKOFF_CAP = float(os.environ.get("CHROMA_SEARCH_BACKOFF_CAP", "20.0"))

# HTTP
HTTP_TIMEOUT = float(os.environ.get("CHROMA_HTTP_TIMEOUT", "35"))

# =========================
# Helpers
# =========================
def _to_similarity(dist: float) -> float:
    """Convert distance to similarity in [0,1]."""
    try:
        d = float(dist)
        if CHROMA_DISTANCE_METRIC == "cosine":
            # Chroma returns cosine distances in [0,2], similar ~ 1 - d/2
            sim = 1.0 - (d / 2.0)
        elif CHROMA_DISTANCE_METRIC == "l2":
            sim = 1.0 / (1.0 + d)
        else:
            # Already similarity
            sim = d
    except Exception:
        sim = 0.0
    return max(0.0, min(1.0, sim))


def _clean_snippet(s: str, limit: int = 850) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    if len(s) > limit:
        s = s[:limit].rsplit(" ", 1)[0] + "…"
    return s


def _first_non_empty(*vals: Optional[str]) -> str:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for it in items:
        m = it.get("meta", {}) or {}
        key = (
            m.get("path") or m.get("FILENAME") or "",
            m.get("chunk_index"),
            (_clean_snippet(it.get("snippet", ""))[:64]),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


# =========================
# Metadata filter
# =========================
@dataclass
class MetaData:
    STEM: Optional[str] = None
    GROUP_KEY: Optional[str] = None
    COURSE_CODE: Optional[str] = None
    SEMESTER: Optional[str] = None
    chunk_index: Optional[int] = None
    COURSE_FOLDER: Optional[str] = None
    chunk_hash: Optional[str] = None
    file_hash: Optional[str] = None
    CATEGORY: Optional[str] = None
    path: Optional[str] = None
    DEPARTMENT: Optional[str] = None
    file_mtime: Optional[int] = None
    FILENAME: Optional[str] = None
    LEVEL: Optional[str] = None
    COURSE_NUMBER: Optional[str] = None
    file_size: Optional[int] = None
    total_chunks_in_doc: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_where(self) -> Optional[Dict[str, Any]]:
        pairs: List[tuple[str, Any]] = []
        for k, v in asdict(self).items():
            if k == "extra":
                continue
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            if isinstance(v, (list, tuple, set)):
                v = {"$in": list(v)}
            pairs.append((k, v))
        if not pairs:
            return None
        if len(pairs) == 1:
            k, v = pairs[0]
            return {k: v}
        return {"$and": [{k: v} for k, v in pairs]}

    @classmethod
    def from_partial(cls, d: Dict[str, Any]) -> "MetaData":
        field_names = set(cls.__dataclass_fields__.keys())  # type: ignore
        lower_map = {name.lower(): name for name in field_names}
        mapped: Dict[str, Any] = {}
        for k, v in d.items():
            if k in field_names:
                mapped[k] = v
                continue
            lk = k.lower()
            if lk in lower_map:
                mapped[lower_map[lk]] = v
                continue
            nk = lk.replace(" ", "_").replace("-", "_")
            if nk in lower_map:
                mapped[lower_map[nk]] = v
        return cls(**mapped)

    @staticmethod
    def make_where(d: Optional[Dict[str, Any]] = None, /, **kwargs: Any) -> Optional[Dict[str, Any]]:
        if d and any(isinstance(k, str) and k.startswith("$") for k in d.keys()):
            out = dict(d)
        else:
            data = dict(d or {})
            data.update(kwargs)
            out = MetaData.from_partial(data).to_where()
        return out


# =========================
# Chroma Query wrapper
# =========================
class ChromaQuery:
    def __init__(
        self,
        chroma_path: str = CHROMA_PATH,
        collection_name: str = CHROMA_COLLECTION,
        cf_account_id: str = CF_ACCOUNT_ID,
        cf_api_token: str = CF_API_TOKEN,
        use_bge_prefixes: bool = USE_BGE_PREFIXES,
    ):
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.CF_ACCOUNT_ID = cf_account_id
        self.CF_API_TOKEN = cf_api_token
        self.USE_BGE_PREFIXES = use_bge_prefixes
        self._embed_cache: dict[str, List[float]] = {}
        self._embed_last_ts: float = 0.0

        self.client = PersistentClient(path=self.chroma_path)
        self.col = self.client.get_or_create_collection(name=self.collection_name)
        try:
            cnt = self.col.count()
        except Exception as e:
            cnt = -1
            logger.warning(f"Chroma count() failed: {e}")
        logger.info(
            "ChromaDB path='%s' collection='%s' (count=%s)",
            self.chroma_path,
            self.collection_name,
            cnt,
        )
        logger.info("Chroma search strict embeddings=%s, QPS=%.2f", str(REQUIRE_EMBEDDINGS), EMBED_QPS)

    # ---------- Where normalization ----------
    @staticmethod
    def _normalize_where(where: Any | None) -> Optional[Dict[str, Any]]:
        if where is None:
            return None
        if isinstance(where, MetaData):
            return where.to_where()
        if isinstance(where, dict):
            return MetaData.make_where(where)
        return None

    # ---------- CF BGE-M3 embeddings ----------
    def _embed_sleep_if_needed(self):
        if EMBED_QPS <= 0:
            return
        min_interval = 1.0 / max(EMBED_QPS, 1e-6)
        now = time.time()
        wait = (self._embed_last_ts + min_interval) - now
        if wait > 0:
            time.sleep(wait)
        self._embed_last_ts = time.time()

    def _cf_bge_m3_embed(self, texts: List[str], *, input_type: str = "query") -> List[List[float]]:
        if OFFLINE:
            raise RuntimeError("OFFLINE mode enabled — remote embeddings disabled")
        if not self.CF_ACCOUNT_ID or not self.CF_API_TOKEN:
            raise RuntimeError("Missing Cloudflare credentials (CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN)")

        # Prepare prefixes
        if self.USE_BGE_PREFIXES:
            pref = f"{input_type}: "
            prefixed = [t if t.startswith(pref) else (pref + t) for t in texts]
        else:
            prefixed = list(texts)

        # Check simple cache (works best for single-text usage)
        results: List[Optional[List[float]]] = [None] * len(prefixed)
        to_request: List[str] = []
        to_request_idx: List[int] = []
        for i, t in enumerate(prefixed):
            cached = self._embed_cache.get(t)
            if cached is not None:
                results[i] = cached
            else:
                to_request.append(t)
                to_request_idx.append(i)

        if to_request:
            url = f"https://api.cloudflare.com/client/v4/accounts/{self.CF_ACCOUNT_ID}/ai/run/@cf/baai/bge-m3"
            headers = {"Authorization": f"Bearer {self.CF_API_TOKEN}", "Content-Type": "application/json"}
            payload = {"text": to_request}

            # Throttle + retry on 429
            err: Optional[Exception] = None
            for attempt in range(EMBED_MAX_RETRIES):
                try:
                    self._embed_sleep_if_needed()
                    r = requests.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
                    if r.status_code == 429:
                        # Respect Retry-After if provided
                        ra = r.headers.get("Retry-After")
                        try:
                            wait = float(ra)
                        except Exception:
                            wait = min(EMBED_BACKOFF_CAP, EMBED_BACKOFF_BASE * (2 ** attempt))
                        logger.warning(f"[embed] 429 rate-limited; retrying in {wait:.2f}s")
                        time.sleep(wait)
                        continue
                    r.raise_for_status()
                    data = r.json()
                    result = data.get("result", {})
                    arr = result.get("data") or result.get("embeddings") or result.get("vectors")
                    if arr is None:
                        raise RuntimeError(f"Unexpected CF response: {json.dumps(data)[:500]}")
                    if arr and isinstance(arr[0], dict):
                        embs = [item.get("embedding") or item.get("vector") for item in arr]
                    else:
                        embs = arr
                    # L2-normalize and cache
                    for local_i, vec in enumerate(embs):
                        x = np.array(vec, dtype=np.float32)
                        n = float(np.linalg.norm(x))
                        normed = (x / n).tolist() if n > 0 else x.tolist()
                        original_idx = to_request_idx[local_i]
                        original_text = prefixed[original_idx]
                        self._embed_cache[original_text] = normed
                        results[original_idx] = normed
                    break  # success
                except Exception as e:
                    err = e
                    wait = min(EMBED_BACKOFF_CAP, EMBED_BACKOFF_BASE * (2 ** attempt))
                    logger.warning(f"[embed] attempt {attempt+1}/{EMBED_MAX_RETRIES} failed: {e} — sleep {wait:.2f}s")
                    time.sleep(wait)

            # If failed to fill some results, raise to trigger fallback upstream
            if any(v is None for v in results):
                raise err or RuntimeError("Embedding request failed")

        # At this point, all results are filled
        return [v for v in results if v is not None]

    # ---------- Keyword fallback (offline/no-embed) ----------
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9]{3,}", (text or "").lower())

    def _keyword_fallback(self, q: str, k: int, where: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info("[Fallback] keyword search")
        tokens = self._tokenize(q) or [q.strip().lower()]
        # Try a filtered 'get', then unfiltered
        try:
            if where:
                got = self.col.get(where=where, limit=max(k * 2, 20), include=["documents", "metadatas"])
            else:
                got = self.col.get(limit=max(k * 2, 20), include=["documents", "metadatas"])
        except Exception as e:
            logger.warning(f"[Fallback] get failed: {e}")
            try:
                got = self.col.get(limit=k, include=["documents", "metadatas"])
            except Exception as e2:
                logger.error(f"[Fallback] final get failed: {e2}")
                return []

        ids = (got.get("ids") or [])
        docs = (got.get("documents") or [])
        metas = (got.get("metadatas") or [])

        # Flatten (Chroma may return nested lists)
        if ids and isinstance(ids[0], list):
            ids = ids[0]
            docs = docs[0] if docs else []
            metas = metas[0] if metas else []

        scored = []
        max_score = 0
        for _id, doc, meta in zip(ids, docs, metas):
            text = (doc or "").lower()
            score = sum(text.count(t) for t in tokens)
            max_score = max(max_score, score)
            scored.append((_id, doc, meta, score))
        out = []
        for _id, doc, meta, sc in scored:
            sim = 0.5 + 0.5 * (sc / max_score) if max_score > 0 else 0.5
            snippet = _first_non_empty(meta.get("snippet") if isinstance(meta, dict) else None, doc)
            item = {
                "id": _id,
                "snippet": _clean_snippet(snippet),
                "meta": dict(meta or {}),
                "score": float(sim),
                "path": (meta or {}).get("path"),
                "COURSE_FOLDER": (meta or {}).get("COURSE_FOLDER"),
                "chunk_index": (meta or {}).get("chunk_index"),
            }
            out.append(item)
        out.sort(key=lambda r: r["score"], reverse=True)
        return out[:k]

    # ---------- Core formatting ----------
    def _format_results(self, resp: Dict[str, Any], show_snippet: bool = True) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        ids = (resp.get("ids") or [[]])[0]
        docs = (resp.get("documents") or [[]])[0]
        metas = (resp.get("metadatas") or [[]])[0]
        dists = (resp.get("distances") or [[]])[0]
        n = max(len(ids), len(docs), len(metas), len(dists))
        for i in range(n):
            _id = ids[i] if i < len(ids) else ""
            doc = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            dist = dists[i] if i < len(dists) else 0.0
            sim = _to_similarity(dist)
            snippet = ""
            if show_snippet:
                snippet = _first_non_empty((meta or {}).get("snippet"), (meta or {}).get("text"), doc)
                snippet = _clean_snippet(snippet)
            normalized_meta = dict(meta or {})
            if "path" not in normalized_meta:
                normalized_meta["path"] = normalized_meta.get("path") or normalized_meta.get("FILENAME")
            out.append(
                {
                    "id": _id,
                    "snippet": snippet,
                    "meta": normalized_meta,
                    "score": float(sim),
                    # convenience mirrors
                    "path": normalized_meta.get("path"),
                    "COURSE_FOLDER": normalized_meta.get("COURSE_FOLDER"),
                    "chunk_index": normalized_meta.get("chunk_index"),
                }
            )
        out.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return out

    # ---------- Public API: deterministic search ----------
    def search(
        self,
        q: str,
        k: int = 10,
        where: Dict[str, Any] | MetaData | None = None,
        show_snippet: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Deterministic top-k search using explicit query embeddings (BGE-M3).
        Falls back to keyword mode when embedding or query fails.
        """
        where_norm = self._normalize_where(where)
        logger.info(f"[search] q='{q}' k={k} where={where_norm}")
        # Try vector search with CF embeddings (with search-level retries)
        last_err = None
        for attempt in range(max(1, SEARCH_MAX_RETRIES)):
            try:
                q_vec = self._cf_bge_m3_embed([q], input_type="query")
                resp = self.col.query(
                    query_embeddings=q_vec,
                    n_results=max(1, int(k)),
                    include=["documents", "metadatas", "distances"],
                    where=where_norm,
                )
                if not resp.get("documents") or not resp["documents"][0]:
                    if REQUIRE_EMBEDDINGS:
                        logger.warning("[search] empty vector result; strict mode active — no fallback")
                        return []
                    logger.warning("[search] empty vector result; using fallback")
                    return self._keyword_fallback(q, k, where_norm)
                return self._format_results(resp, show_snippet=show_snippet)
            except Exception as e:
                last_err = e
                if attempt < SEARCH_MAX_RETRIES - 1:
                    wait = min(SEARCH_BACKOFF_CAP, SEARCH_BACKOFF_BASE * (2 ** attempt))
                    logger.warning(f"[search] vector query failed (attempt {attempt+1}/{SEARCH_MAX_RETRIES}): {e} — retry in {wait:.2f}s")
                    time.sleep(wait)
                    continue
                # Final attempt exhausted
                if REQUIRE_EMBEDDINGS:
                    logger.warning(f"[search] vector query failed after retries; strict mode — no fallback: {e}")
                    return []
                logger.warning(f"[search] vector query failed: {e} — using fallback")
                return self._keyword_fallback(q, k, where_norm)

    # ---------- Public API: temperature sampling ----------
    def search_with_temperature(
        self,
        q: str,
        *,
        topk: int = DEFAULT_TOPK,
        final_k: int = DEFAULT_FINAL_K,
        tau: float = DEFAULT_TAU,
        min_sim: float = DEFAULT_MIN_SIM,
        where: Dict[str, Any] | MetaData | None = None,
        seed: Optional[int] = DEFAULT_SEED,
        show_snippet: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        1) Pull a large candidate pool (topk)
        2) Filter by min_sim (or keep a small cap if empty)
        3) Softmax over similarity/tau; sample without replacement to final_k
        4) Return chosen, sorted by similarity desc
        """
        where_norm = self._normalize_where(where)
        logger.info(f"[temp] q='{q}' topk={topk} final_k={final_k} tau={tau} min_sim={min_sim} where={where_norm}")

        base = self.search(q, k=topk, where=where_norm, show_snippet=False)
        if not base:
            return []

        sims = np.array([b["score"] for b in base], dtype=np.float64)
        keep_idx = np.where(sims >= float(min_sim))[0]
        if len(keep_idx) == 0:
            # keep a small greedy prefix if threshold filters everything
            keep_idx = np.arange(min(len(base), max(8, final_k * 3)))
        pool = [base[i] for i in keep_idx]
        pool_sims = np.array([p["score"] for p in pool], dtype=np.float64)

        # Temperature softmax
        logits = pool_sims - pool_sims.max()
        denom = max(tau, 1e-6)
        probs = np.exp(logits / denom)
        probs = probs / (probs.sum() or 1.0)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        take = min(final_k, len(pool))
        idxs = np.arange(len(pool))
        chosen_idx = np.random.choice(idxs, size=take, replace=False, p=probs)
        chosen = [pool[i] for i in chosen_idx.tolist()]
        chosen.sort(key=lambda r: r["score"], reverse=True)

        # Finish with snippets
        if show_snippet:
            for it in chosen:
                if not it.get("snippet"):
                    # hydrate a snippet quickly (we don't have the full doc here; rely on meta.snippet if present)
                    meta = it.get("meta") or {}
                    it["snippet"] = _clean_snippet(_first_non_empty(meta.get("snippet"), meta.get("text", "")))
        return chosen[:final_k]


# =========================
# Smoke test (optional)
# =========================
if __name__ == "__main__":
    cq = ChromaQuery()
    where = MetaData.make_where(COURSE_FOLDER="EEE 313")
    results = cq.search("BJT biasing worked example", k=6, where=where, show_snippet=True)
    logger.info("search() returned %d results", len(results))
    sample = cq.search_with_temperature("DC machines commutation derivation", topk=40, final_k=8, where=where, show_snippet=True)
    logger.info("search_with_temperature() returned %d results", len(sample))
