# src/services/Ollama/ollama_service.py
"""Wrapper around the Ollama client for embedding generation."""

from __future__ import annotations
import os
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import ollama
except ImportError:
    ollama = None

from COURSEGEN.utils.logging_utils import get_rag_logger


class OllamaService:
    """Service wrapper for Ollama embedding generation with caching and error handling."""

    def __init__(
        self,
        model: str = "bge-m3:latest",
        host: str = "http://localhost:11434",
        timeout: int = 30,
        workers: Optional[int] = None,
    ) -> None:
        self.model = model
        self.host = host
        self.timeout = timeout

        # Dynamic default workers: ~2x cores, capped 32; override via env or arg
        default_workers = max(8, min(32, (os.cpu_count() or 8) * 2))
        self.workers = int(os.getenv("OLLAMA_EMBED_WORKERS", str(workers or default_workers)))

        self.logger = get_rag_logger("OllamaService")
        if ollama is None:
            raise ImportError("ollama package is required. Install with: pip install ollama")

        self.client = ollama.Client(host=host)
        self._verify_model()
        self.logger.success(f"Initialized Ollama service with model: {model}, workers={self.workers}")

    def _verify_model(self) -> None:
        try:
            test_response = self.client.embeddings(model=self.model, prompt="ping")
            if "embedding" not in test_response:
                raise ValueError(f"Model {self.model} did not return 'embedding'")
            self.logger.info(f"Model {self.model} verified successfully")
        except Exception as e:
            self.logger.error(f"Model verification failed: {e}")
            raise RuntimeError(
                f"Failed to verify Ollama model '{self.model}'. "
                f"Ensure Ollama is running and model is pulled: ollama pull {self.model}"
            ) from e

    def embed_single(self, text: str) -> List[float]:
        if not text or not text.strip():
            self.logger.warning("Empty text provided for embedding")
            return []
        try:
            response = self.client.embeddings(model=self.model, prompt=text)
            if "embedding" not in response:
                raise ValueError("Invalid response format from Ollama")
            emb = response["embedding"]
            self.logger.debug(f"Generated embedding with {len(emb)} dimensions")
            return emb
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            raise

    def _task(self, idx_text: Tuple[int, str]) -> Tuple[int, List[float]]:
        idx, text = idx_text
        try:
            emb = self.embed_single(text)
            return idx, emb
        except Exception:
            # Return empty vector; upstream will handle dimension after batch
            return idx, []

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts. Preserves order."""
        if not texts:
            return []
        max_workers = max(1, self.workers)
        results: List[Optional[List[float]]] = [None] * len(texts)
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._task, (i, t)) for i, t in enumerate(texts)]
            for fut in as_completed(futures):
                idx, emb = fut.result()
                results[idx] = emb
                completed += 1
                if completed % 10 == 0 or completed == len(texts):
                    self.logger.info(f"Ollama embeddings progress {completed}/{len(texts)}")

        # Fill any None with []
        out = [r if r is not None else [] for r in results]
        return out

    def get_model_info(self) -> dict:
        try:
            models = self.client.list()
            for model in models.get("models", []):
                if model["name"] == self.model:
                    return {
                        "name": model["name"],
                        "size": model.get("size", "unknown"),
                        "digest": model.get("digest", "unknown"),
                        "modified_at": model.get("modified_at", "unknown"),
                    }
            return {"name": self.model, "status": "not_found"}
        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            return {"name": self.model, "error": str(e)}

    def health_check(self) -> bool:
        try:
            response = self.client.embeddings(model=self.model, prompt="health check")
            return "embedding" in response
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
