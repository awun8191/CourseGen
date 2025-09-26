"""Course-level progress cache for resumable question generation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .batch_utils import slugify


class CourseProgressCache:
    """Persistent tracker for course/topic/subtopic generation progress."""

    def __init__(
        self,
        *,
        course_code: str,
        cache_root: Path | str,
        theory_target: int,
        calc_target: int,
    ) -> None:
        self.course_code = course_code or "unknown"
        self.theory_target = max(0, theory_target)
        self.calc_target = max(0, calc_target)

        root = Path(cache_root).expanduser().resolve() / "course_progress"
        root.mkdir(parents=True, exist_ok=True)
        slug = slugify(self.course_code) or "course"
        self.path = root / f"{slug}.json"

        self.data = self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            data = {"course": self.course_code, "topics": {}}
            self._save(data)
            return data

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            data = {"course": self.course_code, "topics": {}}

        if data.get("course") != self.course_code:
            data["course"] = self.course_code

        data.setdefault("topics", {})
        return data

    def _save(self, data: Optional[Dict[str, Any]] = None) -> None:
        payload = data or self.data
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _ensure_entry(self, topic: str, subtopic: str) -> Dict[str, Any]:
        topics = self.data.setdefault("topics", {})
        topic_entry = topics.setdefault(topic, {"subtopics": {}})
        subtopics = topic_entry.setdefault("subtopics", {})
        entry = subtopics.setdefault(
            subtopic,
            {
                "theory_progress": 0,
                "calculation_progress": 0,
                "calc_progress2": 0,
                "state": "in_progress",
                "persisted": False,
                "last_updated": time.time(),
            },
        )
        entry.setdefault("theory_progress", 0)
        entry.setdefault("calculation_progress", 0)
        entry.setdefault("calc_progress2", 0)
        entry.setdefault("state", "in_progress")
        entry.setdefault("persisted", False)
        entry.setdefault("last_updated", time.time())
        return entry

    def touch_subtopic(self, topic: str, subtopic: str) -> None:
        self._ensure_entry(topic, subtopic)
        self._save()

    # ------------------------------------------------------------------
    # Progress mutation API
    # ------------------------------------------------------------------
    def mark_request_started(self, topic: str, subtopic: str, request_name: str) -> None:
        entry = self._ensure_entry(topic, subtopic)
        if entry.get("state") == "completed":
            return
        entry.pop("error", None)
        entry["state"] = "in_progress"
        entry["current_request"] = request_name
        entry["last_updated"] = time.time()
        self._save()

    def mark_request_completed(
        self,
        topic: str,
        subtopic: str,
        request_name: str,
        count: int,
    ) -> bool:
        entry = self._ensure_entry(topic, subtopic)
        progress_key, target = self._progress_target(request_name)
        entry[progress_key] = max(entry.get(progress_key, 0), min(count, target))
        entry.pop("error", None)
        entry.pop("current_request", None)
        entry["state"] = "in_progress"
        entry["persisted"] = False
        entry["last_updated"] = time.time()

        if self._is_subtopic_complete(entry):
            entry["state"] = "completed"
            entry["completed_at"] = time.time()
        self._save()
        return entry["state"] == "completed"

    def mark_request_failed(
        self,
        topic: str,
        subtopic: str,
        request_name: str,
        error: str,
    ) -> None:
        entry = self._ensure_entry(topic, subtopic)
        progress_key, _ = self._progress_target(request_name)
        entry[progress_key] = 0
        entry["state"] = "error"
        entry["error"] = error
        entry["last_failed_request"] = request_name
        entry["persisted"] = False
        entry["last_updated"] = time.time()
        self._save()

    def mark_subtopic_error(self, topic: str, subtopic: str, error: str) -> None:
        entry = self._ensure_entry(topic, subtopic)
        entry["state"] = "error"
        entry["error"] = error
        entry["persisted"] = False
        entry["last_updated"] = time.time()
        self._save()

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------
    def should_skip(self, topic: str, subtopic: str, request_name: str) -> bool:
        entry = self._ensure_entry(topic, subtopic)
        if entry.get("state") == "completed":
            return True
        if entry.get("state") == "error":
            return False
        progress_key, target = self._progress_target(request_name)
        return entry.get(progress_key, 0) >= target

    def subtopic_state(self, topic: str, subtopic: str) -> str:
        entry = self._ensure_entry(topic, subtopic)
        return entry.get("state", "in_progress")

    def has_persisted(self, topic: str, subtopic: str) -> bool:
        entry = self._ensure_entry(topic, subtopic)
        return bool(entry.get("persisted"))

    def mark_persisted(self, topic: str, subtopic: str) -> None:
        entry = self._ensure_entry(topic, subtopic)
        entry["persisted"] = True
        entry["last_updated"] = time.time()
        self._save()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _progress_target(self, request_name: str) -> tuple[str, int]:
        lower = request_name.lower()
        if lower.startswith("theory"):
            return "theory_progress", self.theory_target
        if lower == "calculation-2":
            return "calc_progress2", self.calc_target
        if lower.startswith("calculation"):
            return "calculation_progress", self.calc_target
        key = f"{lower}_progress"
        return key, self.calc_target

    def _is_subtopic_complete(self, entry: Dict[str, Any]) -> bool:
        if self.theory_target > 0 and entry.get("theory_progress", 0) < self.theory_target:
            return False
        if self.calc_target > 0 and entry.get("calculation_progress", 0) < self.calc_target:
            return False
        if self.calc_target > 0 and entry.get("calc_progress2", 0) < self.calc_target:
            return False
        return True
