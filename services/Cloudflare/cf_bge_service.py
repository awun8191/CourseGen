# src/services/Cloudflare/cf_bge_service.py
from __future__ import annotations

import os
import time
import json
from typing import List, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from COURSEGEN.utils.logging_utils import get_rag_logger

CF_DEFAULT_MODEL = "@cf/baai/bge-m3"           # Cloudflare Workers AI model slug
CF_MAX_BATCH = int(os.getenv("CF_EMBED_MAX_BATCH", "100"))  # max per request
CF_WORKERS = int(os.getenv("CF_EMBED_WORKERS", "8"))        # client threads
CF_TIMEOUT = int(os.getenv("CF_EMBED_TIMEOUT", "45"))       # seconds

class CloudflareBGEService:
    """
    Embedding client for Cloudflare Workers AI (BGE-M3).
    Native REST by default, optional OpenAI-compatible mode.
    Mirrors the Ollama service's embed_single/embed interface.
    """
    def __init__(
        self,
        model: str = CF_DEFAULT_MODEL,
        account_id: Optional[str] = None,
        api_token: Optional[str] = None,
        use_openai_compat: bool = False,
    ) -> None:
        self.logger = get_rag_logger("CloudflareBGEService")
        self.model = model

        # set CLOUDFLARE_ACCOUNT_ID=c1719c3cf4696ae260e6a5f57b1f3100
        # set CLOUDFLARE_API_TOKEN=U7hBTssgt-8DAi1cyr3GnwihAVKqLUa37Su2q_-e
        

# $env:CLOUDFLARE_ACCOUNT_ID='c1719c3cf4696ae260e6a5f57b1f3100'; $env:CLOUDFLARE_API_TOKEN="U7hBTssgt-8DAi1cyr3GnwihAVKqLUa37Su2q_-e"; $env:CF_EMBED_MAX_BATCH='100'; $env:OCR_MAX_IMAGE_BYTES='67108864'; $env:TESSERACT_CMD='C:\Program Files\Tesseract-OCR\tesseract.exe'; $env:PADDLE_LANG='en'; $env:BILLING_ENABLED='1'; $env:OMP_NUM_THREADS='4'; python -c "import os;import sys;sys.exit(0 if (os.getenv('CLOUDFLARE_ACCOUNT_ID') and os.getenv('CLOUDFLARE_API_TOKEN')) else 1)"; if ($LASTEXITCODE -ne 0) { Write-Error 'Missing CLOUDFLARE_* envs'; } else { python src/services/RAG/convert_to_embeddings.py -i 'C:\Users\awun8\Documents\SCHOOL\COMPILATION\EEE' --export-dir data/exported_data --cache-dir data/gemini_cache --persist-dir chroma_db_bge_m3 --resume --workers 4 --omp-threads 4 --ocr-dpi 200 --embed-batch 100 --ocr-on-missing fallback }

        self.account_id = account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID") or "c1719c3cf4696ae260e6a5f57b1f3100"
        self.api_token  = api_token  or os.getenv("CLOUDFLARE_API_TOKEN")  or "U7hBTssgt-8DAi1cyr3GnwihAVKqLUa37Su2q_-e"

        if not self.account_id or not self.api_token:
            raise RuntimeError("Cloudflare credentials missing. Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN.")

        if use_openai_compat:
            self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/v1"
            self.endpoint = f"{self.base_url}/embeddings"
            self._mode = "openai_compat"
        else:
            self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai"
            self.endpoint = f"{self.base_url}/run/{self.model}"
            self._mode = "native"

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        })

        # quick verify
        _ = self._request(["ok"])
        self.logger.success(f"Cloudflare Workers AI ready: {self.model} via {self._mode}")

        self._detected_dim: Optional[int] = None

    # --------------- public API ---------------
    def embed_single(self, text: str) -> List[float]:
        if not text or not text.strip():
            self.logger.warning("Empty text for embedding")
            return []
        embs = self._request([text])
        vec = embs[0] if embs else []
        if vec and self._detected_dim is None:
            self._detected_dim = len(vec)
            self.logger.info(f"Detected embedding dim: {self._detected_dim}")
        return vec

    def embed(self, texts: Sequence[str], target_dim: Optional[int] = None) -> List[List[float]]:
        if not texts:
            return []

        # client-side batching
        batches: List[List[str]] = []
        n = len(texts)
        for i in range(0, n, CF_MAX_BATCH):
            batches.append(list(texts[i:i+CF_MAX_BATCH]))

        self.logger.info(f"Cloudflare embedding {n} texts in {len(batches)} batch(es) with {CF_WORKERS} workers")
        results: List[Optional[List[float]]] = [None] * n

        def _run(offset: int, payload: List[str]):
            embs = self._request(payload)
            return offset, embs

        with ThreadPoolExecutor(max_workers=max(1, CF_WORKERS)) as pool:
            futs = []
            offset = 0
            for batch in batches:
                futs.append(pool.submit(_run, offset, batch))
                offset += len(batch)

            for fut in as_completed(futs):
                off, embs = fut.result()
                for j, v in enumerate(embs):
                    results[off + j] = v

        out: List[List[float]] = []
        pad_dim = target_dim or self._detected_dim or 1024
        for r in results:
            vec = r if r is not None else [0.0] * pad_dim
            if target_dim:
                if len(vec) > target_dim:
                    vec = vec[:target_dim]
                elif len(vec) < target_dim:
                    vec = vec + [0.0] * (target_dim - len(vec))
            out.append(vec)

        if out and self._detected_dim is None:
            self._detected_dim = len(out[0])
            self.logger.info(f"Detected embedding dim: {self._detected_dim}")

        self.logger.success(f"Generated {len(out)} embeddings via Cloudflare")
        return out

    def get_model_info(self) -> dict:
        return {"name": self.model, "provider": "cloudflare", "detected_dim": self._detected_dim}

    def health_check(self) -> bool:
        try:
            _ = self._request(["health check"])
            return True
        except Exception as e:
            self.logger.error(f"Cloudflare health check failed: {e}")
            return False

    # --------------- internals ---------------
    def _request(self, texts: List[str]) -> List[List[float]]:
        if self._mode == "native":
            url = self.endpoint
            body = {"text": texts}
        else:
            url = self.endpoint
            body = {"model": self.model, "input": texts}

        t0 = time.time()
        resp = self.session.post(url, data=json.dumps(body), timeout=CF_TIMEOUT)
        ms = (time.time() - t0) * 1000
        if resp.status_code >= 300:
            raise RuntimeError(f"Cloudflare API {resp.status_code}: {resp.text[:300]}")
        data = resp.json()

        if self._mode == "native":
            vectors = data.get("data") or data.get("response")
        else:
            vectors = [item.get("embedding") for item in (data.get("data") or [])]

        if not vectors or not isinstance(vectors, list):
            raise RuntimeError(f"Unexpected Cloudflare response: {str(data)[:300]}")

        self.logger.debug(f"CF batch {len(texts)} -> {len(vectors)} in {ms:.0f} ms")
        return vectors