#!/usr/bin/env python3
# pip install chromadb requests numpy
from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import requests
from chromadb import PersistentClient, Where
from pathlib import Path

# ========= Config =========
# Resolve repo root and provide stable, repo-anchored defaults.
try:
    REPO_ROOT = Path(__file__).resolve().parents[2]
except Exception:
    REPO_ROOT = Path.cwd()

# Allow overrides via env; otherwise default to repo_root/OUTPUT_DATA2/emdeddings
CHROMA_PATH = os.environ.get(
    "COURSEGEN_PERSIST_DIR",
    str((REPO_ROOT / "OUTPUT_DATA2/emdeddings").resolve()),
)
COLLECTION = os.environ.get("COURSEGEN_COLLECTION", "course_embeddings")

# Cloudflare (required)
CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CF_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")

# If your docs were embedded with "passage:" prefix, keep this True
USE_BGE_PREFIXES = True

# Retrieval knobs (tune as you like)
TOPK = 50           # initial pool size from Chroma
FINAL_K = 8         # how many chunks to return to the LLM
TAU = 0.35          # temperature for softmax sampling (lower = greedier)
MIN_SIM = 0.60      # similarity floor (BGE-M3: ~0.58–0.65 is a good start)
SEED: Optional[int] = None  # set to an int for reproducibility (e.g., 42)
OFFLINE = os.environ.get("OFFLINE", "").lower() in ("1", "true", "yes")


# ========= Metadata helper =========
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

    def to_where(self) -> Optional[Dict[str, Any]]:
        """Return a Chroma 1.x-compliant `where` filter.

        - None if no fields are set
        - One field -> {field: value}
        - Many fields -> {"$and": [{f: v}, ...]}
        - list/tuple/set values become $in expressions automatically
        - dict values are treated as operator expressions (e.g., {"$in": [...]})
        """
        pairs: List[tuple[str, Any]] = []
        for k, v in self.__dict__.items():
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
    def from_dict(cls, d: Dict[str, Any]) -> "MetaData":
        keys = list(cls.__dataclass_fields__.keys())  # type: ignore
        return cls(**{k: d.get(k) for k in keys})

    @classmethod
    def from_partial(cls, d: Dict[str, Any]) -> "MetaData":
        field_names = list(cls.__dataclass_fields__.keys())  # type: ignore
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
        """Ergonomic builder for `where` filters."""
        if d and any(isinstance(k, str) and k.startswith("$") for k in d.keys()):
            return d
        data = dict(d or {})
        data.update(kwargs)
        return MetaData.from_partial(data).to_where()


# ========= Chroma + CF BGE-M3 =========
class ChromaQuery:
    def __init__(
        self,
        chroma_path: str = CHROMA_PATH,
        collection_name: str = COLLECTION,
        cf_account_id: str = CF_ACCOUNT_ID,
        cf_api_token: str = CF_API_TOKEN,
        use_bge_prefixes: bool = USE_BGE_PREFIXES,
    ):
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.CF_ACCOUNT_ID = cf_account_id
        self.CF_API_TOKEN = cf_api_token
        self.USE_BGE_PREFIXES = use_bge_prefixes

        try:
            self.client = PersistentClient(path=self.chroma_path)
            self.col = self.client.get_or_create_collection(name=self.collection_name)
            print(f"Successfully connected to ChromaDB collection: {self.collection_name}")
            print(f"Collection count: {self.col.count()}")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise

    # ---------- Utils ----------
    @staticmethod
    def _normalize(vec: List[float]) -> List[float]:
        x = np.array(vec, dtype=np.float32)
        n = float(np.linalg.norm(x))
        return (x / n).tolist() if n > 0 else x.tolist()

    @staticmethod
    def _sim_from_distance(d: float) -> float:
        return 1.0 - float(d)

    @staticmethod
    def _safe_ids(res: Dict[str, Any]) -> List[str]:
        if "ids" in res and res["ids"] and res["ids"][0]:
            return list(res["ids"][0])
        n = len(res["documents"][0]) if "documents" in res and res["documents"] else 0
        return [str(i) for i in range(n)]

    # ---------- Embeddings via Cloudflare Workers AI (BGE-M3) ----------
    def cf_bge_m3_embed(self, texts: List[str], *, input_type: str = "query") -> List[List[float]]:
        if OFFLINE:
            raise RuntimeError("OFFLINE mode: remote embeddings disabled")
        
        if not self.CF_ACCOUNT_ID or not self.CF_API_TOKEN:
            raise RuntimeError("Cloudflare credentials not set. Please set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN environment variables.")
        
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.CF_ACCOUNT_ID}/ai/run/@cf/baai/bge-m3"
        headers = {"Authorization": f"Bearer {self.CF_API_TOKEN}", "Content-Type": "application/json"}

        if self.USE_BGE_PREFIXES:
            pref = f"{input_type}: "
            texts = [t if t.startswith(pref) else (pref + t) for t in texts]

        payload = {"text": texts}
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
        except requests.exceptions.RequestException as e:
            print(f"Cloudflare API request failed: {e}")
            raise

        result = data.get("result", {})
        arr = result.get("data") or result.get("embeddings") or result.get("vectors")
        if arr is None:
            raise RuntimeError(f"Unexpected CF response: {json.dumps(data)[:500]}")

        if arr and isinstance(arr[0], dict):
            embs = [item.get("embedding") or item.get("vector") for item in arr]
        else:
            embs = arr

        return [self._normalize(v) for v in embs]

    # ---------- Simple keyword fallback (offline/no-embed path) ----------
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re
        return [t for t in re.findall(r"[A-Za-z0-9]+", text.lower()) if len(t) > 2]

    def _keyword_fallback(
        self,
        q: str,
        k: int,
        where: Where | Dict[str, Any] | MetaData | None,
    ) -> List[Dict[str, Any]]:
        print("[INFO] Using keyword fallback search")
        tokens = self._tokenize(q) or [q.strip()]
        wd = {"$contains": q.lower()}  # Simplified document search
        w = self._normalize_where(where)
        
        try:
            # Try a simpler approach first
            if w is None:
                got = self.col.get(
                    where_document=wd, 
                    limit=max(k * 2, 20), 
                    include=["documents", "metadatas"]
                )
            else:
                # Try without document filter first if where filter exists
                got = self.col.get(
                    where=w, 
                    limit=max(k * 2, 20), 
                    include=["documents", "metadatas"]
                )
        except Exception as e:
            print(f"[WARNING] Keyword search with filters failed: {e}")
            # Final fallback - just get some documents
            try:
                got = self.col.get(limit=k, include=["documents", "metadatas"])
            except Exception as e2:
                print(f"[ERROR] All fallback attempts failed: {e2}")
                return []

        ids = list(got.get("ids", []) or [])
        docs = list(got.get("documents", []) or [])
        metas = list(got.get("metadatas", []) or [])

        scored: List[Dict[str, Any]] = []
        max_score = 0
        
        for _id, doc, meta in zip(ids, docs, metas):
            text = (doc or "").lower()
            score = sum(text.count(t) for t in tokens)
            max_score = max(max_score, score)
            scored.append({"id": _id, "document": doc, "metadata": meta, "_score": score})

        # normalize score to a similarity in [0.5, 1.0]
        for it in scored:
            if max_score > 0:
                sim = 0.5 + 0.5 * (it["_score"] / max_score)
            else:
                sim = 0.5
            it["similarity"] = float(sim)
            it["distance"] = float(1.0 - sim)
            it.pop("_score", None)

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:k]

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

    # ---------- Base retrieval (no sampling) ----------
    def search(
        self,
        q: str,
        k: int = 5,
        where: Where | Dict[str, Any] | MetaData | None = None,
        show_snippet: bool = True,
    ) -> List[Dict[str, Any]]:
        print(f"[INFO] Searching for: '{q}' with k={k}")
        
        # Try to embed the query; fall back to keyword search if offline/network error
        try:
            print("[INFO] Attempting to embed query...")
            q_vec = self.cf_bge_m3_embed([q], input_type="query")
            print(f"[INFO] Successfully embedded query. Vector dimension: {len(q_vec[0])}")
        except Exception as embed_err:
            print(f"[WARNING] Embedding failed, falling back to keyword search: {embed_err}")
            return self._keyword_fallback(q, k, where)
        
        where_norm = self._normalize_where(where)
        print(f"[INFO] Normalized where filter: {where_norm}")
        
        try:
            # Start with a simpler query approach
            print("[INFO] Attempting vector similarity search...")
            res = self.col.query(
                query_embeddings=q_vec,
                n_results=k,
                include=["documents", "metadatas", "distances"],
                where=where_norm,
            )
            print(f"[INFO] Vector search successful. Found {len(res['documents'][0]) if res.get('documents') and res['documents'][0] else 0} results")
            
        except Exception as e:
            print(f"[WARNING] Vector search failed: {e}")
            
            # Try without the where filter
            if where_norm is not None:
                print("[INFO] Retrying without where filter...")
                try:
                    res = self.col.query(
                        query_embeddings=q_vec,
                        n_results=k,
                        include=["documents", "metadatas", "distances"],
                        where=None,
                    )
                    print(f"[INFO] Unfiltered vector search successful. Found {len(res['documents'][0]) if res.get('documents') and res['documents'][0] else 0} results")
                except Exception as e2:
                    print(f"[WARNING] Unfiltered vector search also failed: {e2}")
                    return self._keyword_fallback(q, k, where)
            else:
                # If no where filter was used and it still failed, try keyword fallback
                return self._keyword_fallback(q, k, where)

        # Process results
        if not res.get("documents") or not res["documents"][0]:
            print("[WARNING] No documents returned from vector search")
            return self._keyword_fallback(q, k, where)

        ids = self._safe_ids(res)
        items: List[Dict[str, Any]] = []
        
        for _id, doc, meta, dist in zip(ids, res["documents"][0], res["metadatas"][0], res["distances"][0]):
            sim = self._sim_from_distance(dist)
            items.append({
                "id": _id, 
                "document": doc, 
                "metadata": meta, 
                "distance": float(dist), 
                "similarity": sim
            })
            
            if show_snippet:
                snippet = (doc or "")[:120].replace("\n", " ")
                course_info = f"{meta.get('COURSE_CODE', 'N/A')}-{meta.get('COURSE_NUMBER', 'N/A')}"
                filename = meta.get('FILENAME', 'Unknown')
                print(f"[RESULT] sim={sim:.3f} | {course_info} | {filename}")
                print(f"         {snippet}...\n")

        # Sort by similarity (best-first)
        items.sort(key=lambda it: it["similarity"], reverse=True)
        return items

    # ---------- Temperature-weighted sampling ----------
    def search_with_temperature(
        self,
        q: str,
        *,
        topk: int = TOPK,
        final_k: int = FINAL_K,
        tau: float = TAU,
        min_sim: float = MIN_SIM,
        where: Where | Dict[str, Any] | MetaData | None = None,
        seed: Optional[int] = SEED,
        show_snippet: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        1) Pull a generous topK from Chroma
        2) Keep only items above min_sim (or fallback to top subset)
        3) Softmax over similarity/tau, sample w/out replacement
        4) Return chosen items sorted by true similarity
        """
        print(f"[INFO] Temperature sampling: topk={topk}, final_k={final_k}, tau={tau}, min_sim={min_sim}")
        
        base = self.search(q, k=topk, where=where, show_snippet=False)  # Don't show snippets twice
        if not base:
            print("[WARNING] No base results found")
            return []

        print(f"[INFO] Retrieved {len(base)} base results")
        
        sims = np.array([it["similarity"] for it in base], dtype=np.float64)
        keep_idx = np.where(sims >= min_sim)[0]
        
        if len(keep_idx) == 0:
            print(f"[INFO] No results above min_sim={min_sim}, keeping top {min(len(base), final_k * 3)} results")
            keep_idx = np.arange(min(len(base), final_k * 3))

        pool = [base[i] for i in keep_idx]
        pool_sims = np.array([it["similarity"] for it in pool], dtype=np.float64)
        pool_sims_preview = [f"{s:.3f}" for s in pool_sims[:5]]
        print(f"[INFO] Filtered pool size: {len(pool)} with similarities: {pool_sims_preview}")

        # Temperature softmax
        pool_sims_norm = pool_sims - pool_sims.max()  # stability
        denom = max(tau, 1e-6)
        p = np.exp(pool_sims_norm / denom)
        p = p / p.sum()

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        take = min(final_k, len(pool))
        choice_idx = np.random.choice(len(pool), size=take, replace=False, p=p)
        chosen = [pool[i] for i in choice_idx]
        chosen.sort(key=lambda x: x["similarity"], reverse=True)

        print(f"[INFO] Selected {len(chosen)} items for final result")
        
        if show_snippet:
            print("\n=== FINAL SELECTED RESULTS ===")
            for i, it in enumerate(chosen):
                snippet = (it["document"] or "")[:120].replace("\n", " ")
                course_info = f"{it['metadata'].get('COURSE_CODE', 'N/A')}-{it['metadata'].get('COURSE_NUMBER', 'N/A')}"
                filename = it['metadata'].get('FILENAME', 'Unknown')
                print(f"[PICKED {i+1}] sim={it['similarity']:.3f} | {course_info} | {filename}")
                print(f"            {snippet}...\n")
                
        return chosen


# ========= Example usage =========
if __name__ == "__main__":
    try:
        cq = ChromaQuery()
        
        # Test basic connection first
        print(f"\n=== ChromaDB Connection Test ===")
        print(f"Collection name: {cq.collection_name}")
        print(f"Total documents in collection: {cq.col.count()}")
        
        # Test a simple query first without filters
        print(f"\n=== Simple Query Test (No Filters) ===")
        simple_results = cq.search("circuit", k=3, where=None, show_snippet=True)
        print(f"Simple query returned {len(simple_results)} results")
        
        # Now test with filters
        print(f"\n=== Filtered Query Test ===")
        where = MetaData.make_where(COURSE_FOLDER="ENG 382")
        print(f"Using filter: {where}")
        
        query = "NUMERICAL COMPUTATIONS Course OUtline"
        
        # Temperature-based sampling
        picked = cq.search_with_temperature(
            query,
            topk=min(TOPK, 20),  # Start with smaller topk for debugging
            final_k=min(FINAL_K, 5),
            tau=TAU,
            min_sim=MIN_SIM,
            where=where,
            seed=SEED,
            show_snippet=True
        )
        
        results = picked
        
        if results:
            print(f"\n=== FINAL RESULTS SUMMARY ===")
            print(f"Query: '{query}'")
            print(f"Total results: {len(results)}")
            print(f"Average similarity: {np.mean([r['similarity'] for r in results]):.3f}")
            
            print("\nTop results metadata:")
            for i, it in enumerate(results[:3]):
                m = it.get("metadata", {}) or {}
                print(f"  {i+1}. COURSE: {m.get('COURSE_CODE')}-{m.get('COURSE_NUMBER')} | FILE: {m.get('FILENAME')} | SIM: {it['similarity']:.3f}")

            # Show assembled context
            context = "\n\n---\n\n".join(it.get("document", "") for it in results)
            print(f"\nAssembled context length: {len(context)} characters")
            
            # Show first result in detail
            if results:
                print(f"\n=== FIRST RESULT DETAIL ===")
                first = results[0]
                print(f"ID: {first['id']}")
                print(f"Similarity: {first['similarity']:.3f}")
                print(f"Metadata: {json.dumps(first['metadata'], indent=2)}")
                print(f"Document preview: {first['document'][:300]}...")
                
        else:
            print("No results found!")
            
    except Exception as e:
        print(f"[ERROR] Application failed: {e}")
        import traceback
        traceback.print_exc()
