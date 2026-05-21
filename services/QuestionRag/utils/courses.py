from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from data_models.course_catalog import CourseCatalogEntry, DepartmentCatalog

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class DataFormatting:
    """
    Load & index course data from a JSON file shaped like your current `courses.json`.

    Compatible API:
        df = DataFormatting()
        course, programs = df.search_course("EEE 313")
    """

    REPO_ROOT = Path(__file__).resolve().parents[3]
    DEFAULT_RELATIVE = REPO_ROOT / "data/textbooks/courses.json"

    def __init__(self, json_path: Optional[str | Path] = None) -> None:
        self._path = self._resolve_path(json_path)
        self._raw: List[Dict[str, Any]] = self._read_json(self._path)
        self.courses: List[DepartmentCatalog] = []
        self._by_code: Dict[str, Tuple[CourseCatalogEntry, List[str]]] = {}
        self._map_data()
        logger.info("Course Data initialized: %d department groups | %d unique codes",
                    len(self.courses), len(self._by_code))

    def _resolve_path(self, provided: Optional[str | Path]) -> Path:
        env_path = Path(str(provided)) if provided else None
        if env_path:
            p = env_path.expanduser().resolve()
            if p.exists():
                return p
            logger.warning("Provided json_path does not exist: %s", p)

        env = os.environ.get("COURSEGEN_COURSES_JSON")
        if env:
            p = Path(env).expanduser().resolve()
            if p.exists():
                return p
            logger.warning("Env COURSEGEN_COURSES_JSON points to missing file: %s", p)

        rel = self.DEFAULT_RELATIVE
        if rel.exists():
            return rel

        raise FileNotFoundError(
            "courses.json not found. Set json_path, or $COURSEGEN_COURSES_JSON, "
            f"or place it at {self.DEFAULT_RELATIVE}"
        )

    def _read_json(self, path: Path) -> List[Dict[str, Any]]:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("courses.json root must be a JSON array")
            return data
        except Exception as e:
            logger.error("Failed to read %s: %s", path, e)
            raise

    @staticmethod
    def _coerce_bool(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "y")
        if isinstance(v, (int, float)):
            return bool(v)
        return False

    @staticmethod
    def _first_or_blank(seq: Any) -> str:
        if isinstance(seq, list) and seq:
            return str(seq[0])
        if isinstance(seq, str):
            return seq
        return ""

    def _make_course(self, row: Dict[str, Any]) -> CourseCatalogEntry:
        return CourseCatalogEntry(
            code=str(row.get("code", "")).strip(),
            title=str(row.get("title", "")).strip(),
            level=self._first_or_blank(row.get("levels")),
            semester=self._first_or_blank(row.get("semesters")),
            units=int(row.get("units", 0) or 0),
            type=str(row.get("type", "")).strip(),
            is_elective=self._coerce_bool(row.get("is_elective", False)),
        )

    def _map_data(self) -> None:
        """
        Build:
          - self.courses: grouped by identical offered_by_programs sets
          - self._by_code: fast lookup by course code (upper)
        """
        groups: Dict[Tuple[str, ...], List[CourseCatalogEntry]] = {}
        programs_for_code: Dict[str, List[str]] = {}

        for row in self._raw:
            programs = [str(p).strip() for p in (row.get("offered_by_programs") or []) if str(p).strip()]
            sig = tuple(sorted(programs))
            course = self._make_course(row)

            if not course.code:
                logger.warning("Skipping course with empty code: %s", course.title)
                continue

            groups.setdefault(sig, []).append(course)

            code_key = course.code.strip().lower()
            if code_key in programs_for_code and programs_for_code[code_key] != programs:
                logger.debug("Course code appears under different programs; keeping first: %s", course.code)
            programs_for_code.setdefault(code_key, programs)

            if code_key in self._by_code:
                prev, _ = self._by_code[code_key]
                if prev.title != course.title:
                    logger.warning("Duplicate code with different title detected: %s | '%s' vs '%s'",
                                   course.code, prev.title, course.title)
                continue
            self._by_code[code_key] = (course, programs)

        dept_models: List[DepartmentCatalog] = []
        for sig, courses in groups.items():
            dept_models.append(DepartmentCatalog(names=list(sig), courses=sorted(courses, key=lambda c: c.code)))
        self.courses = sorted(dept_models, key=lambda d: (d.names[0] if d.names else "", -len(d.courses)))

    def search_course(self, course_code: str) -> Tuple[CourseCatalogEntry, List[str]]:
        """
        Return (CourseModel, offered_by_programs) for matching code (case-insensitive).
        Raises ValueError if not found.
        """
        key = (course_code or "").strip().lower()
        hit = self._by_code.get(key)
        if not hit:
            raise ValueError(f"Course code not found: '{course_code}'")
        return hit

    def find_by_title(self, needle: str, limit: int = 10) -> List[Tuple[CourseCatalogEntry, List[str]]]:
        q = (needle or "").strip().lower()
        out: List[Tuple[CourseCatalogEntry, List[str]]] = []
        for course, programs in self._by_code.values():
            if q in course.title.lower():
                out.append((course, programs))
                if len(out) >= limit:
                    break
        return out

    def list_by_level(self, level: str) -> List[Tuple[CourseCatalogEntry, List[str]]]:
        lv = (level or "").strip().lower()
        return [(c, p) for c, p in self._by_code.values() if c.level.strip().lower() == lv]

    def list_by_program_contains(self, text: str) -> List[Tuple[CourseCatalogEntry, List[str]]]:
        q = (text or "").strip().lower()
        hits: List[Tuple[CourseCatalogEntry, List[str]]] = []
        for course, programs in self._by_code.values():
            if any(q in (prog or "").lower() for prog in programs):
                hits.append((course, programs))
        return hits

    def all_codes(self) -> List[str]:
        return sorted({c.code for c, _ in self._by_code.values()})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "departments": [d.to_dict() for d in self.courses],
            "index_size": len(self._by_code),
            "path": str(self._path),
        }


if __name__ == "__main__":
    import os

    json_path = os.environ.get("COURSEGEN_COURSES_JSON")
    df = DataFormatting(json_path)

    logger.info("Total department groups: %d", len(df.courses))
    logger.info("Total unique course codes: %d", len(df.all_codes()))

    try:
        course, programs = df.search_course("EEE 313")
        logger.info("Found: %s | %s | Level %s | Semester %s | Units %d | Elective=%s",
                    course.code, course.title, course.level, course.semester, course.units, course.is_elective)
        logger.info("Offered by: %s", ", ".join(programs) or "—")
    except ValueError as e:
        logger.error("%s", e)

    for c, progs in df.find_by_title("electrical machines")[:3]:
        logger.info("Title hit: %s (%s) — %s", c.title, c.code, ", ".join(progs))
