"""Disk-backed cache utilities for Gemini question generation runs.

The question generation pipeline makes multiple Gemini calls per subtopic and
we want to avoid re-sending prompts if the process is interrupted.  This cache
provides a tiny JSON index that tracks per-request artifacts.  Each completed
request persists the generated questions as JSON so subsequent runs can resume
without recomputing successful batches.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .batch_utils import slugify

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheKey:
    """Identifier for a cached request."""

    course_code: str
    topic: str
    subtopic: str
    request_name: str

    def to_string(self) -> str:
        parts = [
            slugify(self.course_code),
            slugify(self.topic),
            slugify(self.subtopic),
            slugify(self.request_name),
        ]
        return "-".join(filter(None, parts))


class QuestionCache:
    """Sophisticated JSON backed cache for question generation with fine-grained progress tracking."""

    INDEX_FILE = "cache.json"

    def __init__(self, cache_dir: str | Path, namespace: str = "question_gen") -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve() / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / self.INDEX_FILE
        self._index: Dict[str, Dict[str, Any]] = {}
        self._payload_cache: Dict[str, list[dict[str, Any]]] = {}
        self._load_index()

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------
    def _load_index(self) -> None:
        if not self.index_path.exists():
            self._index = {}
            return
        try:
            self._index = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Failed to read cache index %s: %s", self.index_path, exc)
            self._index = {}

    def _save_index(self) -> None:
        self.index_path.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------
    def make_key(self, course_code: str, topic: str, subtopic: str, request_name: str) -> CacheKey:
        return CacheKey(course_code=course_code, topic=topic, subtopic=subtopic, request_name=request_name)

    def _entry_for(self, key: CacheKey) -> Optional[Dict[str, Any]]:
        return self._index.get(key.to_string())

    def get_entry(self, key: CacheKey) -> Optional[Dict[str, Any]]:
        return self._entry_for(key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self, key: CacheKey) -> Optional[list[dict[str, Any]]]:
        entry = self._entry_for(key)
        if not entry:
            return None
        status = entry.get("status")
        file_name = entry.get("file")
        if status != "complete" or not file_name:
            return None
        path = self.cache_dir / file_name
        if not path.exists():
            logger.debug("Cached file missing for %s; removing index entry", key.to_string())
            self._index.pop(key.to_string(), None)
            self._payload_cache.pop(key.to_string(), None)
            self._save_index()
            return None
        if key.to_string() in self._payload_cache:
            return self._payload_cache[key.to_string()]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                self._payload_cache[key.to_string()] = payload
            return payload
        except json.JSONDecodeError as exc:  # pragma: no cover - corrupted cache
            logger.warning("Invalid JSON in cached result %s: %s", path, exc)
            self._payload_cache.pop(key.to_string(), None)
            return None

    def store(self, key: CacheKey, records: Iterable[dict[str, Any]], meta: Optional[Dict[str, Any]] = None) -> None:
        payload = list(records)
        if not payload:
            self.mark_skipped(key, reason="empty_records", meta=meta)
            return
        filename = f"{key.to_string()}.json"
        path = self.cache_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._index[key.to_string()] = {
            "status": "complete",
            "file": filename,
            "meta": meta or {},
            "updated_at": time.time(),
        }
        self._payload_cache[key.to_string()] = payload
        self._save_index()

    def mark_skipped(
        self,
        key: CacheKey,
        *,
        reason: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._index[key.to_string()] = {
            "status": "skipped",
            "reason": reason,
            "meta": meta or {},
            "updated_at": time.time(),
        }
        self._payload_cache.pop(key.to_string(), None)
        self._save_index()

    def mark_failed(
        self,
        key: CacheKey,
        *,
        reason: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._index[key.to_string()] = {
            "status": "failed",
            "reason": reason,
            "meta": meta or {},
            "updated_at": time.time(),
        }
        self._payload_cache.pop(key.to_string(), None)
        self._save_index()

    def mark_in_progress(
        self,
        key: CacheKey,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = self._index.get(key.to_string())
        if entry and entry.get("status") == "in_progress" and entry.get("meta") == (meta or {}):
            return
        self._index[key.to_string()] = {
            "status": "in_progress",
            "meta": meta or {},
            "updated_at": time.time(),
        }
        self._save_index()

    def has_completed(self, key: CacheKey) -> bool:
        entry = self._entry_for(key)
        if not entry:
            return False
        if entry.get("status") != "complete":
            return False
        file_name = entry.get("file")
        if not file_name:
            return False
        return (self.cache_dir / file_name).exists()

    def get_status(self, key: CacheKey) -> Optional[str]:
        entry = self._entry_for(key)
        if not entry:
            return None
        return entry.get("status")

    def clear(self, key: CacheKey) -> None:
        entry = self._entry_for(key)
        if entry:
            file_name = entry.get("file")
            if file_name:
                path = self.cache_dir / file_name
                if path.exists():
                    try:
                        path.unlink()
                    except OSError:
                        logger.debug("Failed to delete cache file %s", path)
            self._index.pop(key.to_string(), None)
            self._payload_cache.pop(key.to_string(), None)
            self._save_index()

    def subtopic_request_states(self, key_prefix: CacheKey, request_names: Iterable[str]) -> Dict[str, str]:
        states: Dict[str, str] = {}
        for req_name in request_names:
            key = self.make_key(
                key_prefix.course_code,
                key_prefix.topic,
                key_prefix.subtopic,
                req_name,
            )
            entry = self._entry_for(key)
            states[req_name] = (entry or {}).get("status", "missing")
        return states

    def get_course_progress(self, course_code: str) -> Dict[str, Any]:
        """Get progress for a specific course from the cache."""
        return self._index.get(course_code, {})

    def update_subtopic_progress(
        self,
        course_code: str,
        topic: str,
        subtopic: str,
        theory_batches: List[str],
        calc_batches: List[str],
    ) -> None:
        """Update progress for a specific subtopic in the cache."""
        if course_code not in self._index:
            self._index[course_code] = {}

        if topic not in self._index[course_code]:
            self._index[course_code][topic] = {}

        self._index[course_code][topic][subtopic] = {
            "theory_batches": theory_batches,
            "calc_batches": calc_batches,
            "updated_at": time.time(),
        }
        self._save_index()

    def get_pending_batches(self, course_code: str, topic: str, subtopic: str) -> List[str]:
        """Get list of pending batch names for a subtopic."""
        course_progress = self.get_course_progress(course_code)
        topic_progress = course_progress.get(topic, {})
        subtopic_progress = topic_progress.get(subtopic, {})

        pending = []
        theory_batches = subtopic_progress.get("theory_batches", [])
        calc_batches = subtopic_progress.get("calc_batches", [])

        # Check theory batches
        for batch in theory_batches:
            if batch == "pending":
                pending.append(f"theory-{len(pending) + 1}")

        # Check calculation batches
        for batch in calc_batches:
            if batch == "pending":
                pending.append(f"calculation-{len(pending) + 1}")

        return pending

    def mark_batch_completed(self, course_code: str, topic: str, subtopic: str, batch_name: str) -> None:
        """Mark a specific batch as completed."""
        course_progress = self.get_course_progress(course_code)
        topic_progress = course_progress.get(topic, {})
        subtopic_progress = topic_progress.get(subtopic, {})

        theory_batches = subtopic_progress.get("theory_batches", [])
        calc_batches = subtopic_progress.get("calc_batches", [])

        if batch_name.startswith("theory"):
            # Find and update the theory batch
            for i, batch in enumerate(theory_batches):
                if batch == "pending":
                    theory_batches[i] = "completed"
                    break
        elif batch_name.startswith("calculation"):
            # Find and update the calculation batch
            for i, batch in enumerate(calc_batches):
                if batch == "pending":
                    calc_batches[i] = "completed"
                    break

        self.update_subtopic_progress(course_code, topic, subtopic, theory_batches, calc_batches)

    def prune(self, older_than: float | None = None) -> None:
        if older_than is None:
            return
        now = time.time()
        changed = False
        for request_id, entry in list(self._index.items()):
            updated = float(entry.get("updated_at", 0))
            if now - updated >= older_than:
                file_name = entry.get("file")
                if file_name:
                    try:
                        (self.cache_dir / file_name).unlink(missing_ok=True)
                    except Exception:  # pragma: no cover - best effort cleanup
                        pass
                self._index.pop(request_id, None)
                changed = True
        if changed:
            self._save_index()


__all__ = ["CacheKey", "QuestionCache"]
