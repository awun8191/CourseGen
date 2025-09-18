from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

# -------------------------------------------------------------------
# Logging (adjust with env: PYTHONLOGGING or override in your app)
# -------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# -------------------------------------------------------------------
# Data models
# -------------------------------------------------------------------
@dataclass
class CourseModel:
    code: str = ""
    title: str = ""
    level: str = ""        # e.g., "400"
    semester: str = ""     # e.g., "First", "Second"
    units: int = 0
    type: str = ""         # e.g., "Core", "Elective"
    is_elective: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DepartmentModel:
    names: List[str]
    courses: List[CourseModel]

    def to_dict(self) -> Dict[str, Any]:
        return {"names": list(self.names), "courses": [c.to_dict() for c in self.courses]}


# -------------------------------------------------------------------
# Loader / Formatter
# -------------------------------------------------------------------
class DataFormatting:
    """
    Load & index course data from a JSON file shaped like your current `courses.json`.

    Compatible API:
        df = DataFormatting()
        course, programs = df.search_course("EEE 313")
    """

    # Default search path order:
    # 1) explicit json_path param
    # 2) $COURSEGEN_COURSES_JSON
    # 3) repo-local fallback: ./COURSEGEN/data/textbooks/courses.json (if it exists)
    # 4) last resort: your old absolute path (warned)
    DEFAULT_RELATIVE = Path("COURSEGEN/data/textbooks/courses.json")
    LEGACY_ABSOLUTE = Path("/home/user/Documents/Recursive-PDF-EXTRACTION-AND-RAG/COURSEGEN/data/textbooks/courses.json")

    def __init__(self, json_path: Optional[str | Path] = None) -> None:
        self._path = self._resolve_path(json_path)
        self._raw: List[Dict[str, Any]] = self._read_json(self._path)
        self.courses: List[DepartmentModel] = []  # grouped by identical program sets
        self._by_code: Dict[str, Tuple[CourseModel, List[str]]] = {}
        self._map_data()
        logger.info("Course Data initialized: %d department groups | %d unique codes",
                    len(self.courses), len(self._by_code))

    # ---------------- Path resolution ----------------
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

        # relative fallback
        rel = self.DEFAULT_RELATIVE.expanduser().resolve()
        if rel.exists():
            return rel

        # legacy absolute (warn loudly)
        legacy = self.LEGACY_ABSOLUTE
        if legacy.exists():
            logger.warning("Using legacy absolute path: %s", legacy)
            return legacy

        raise FileNotFoundError(
            "courses.json not found. Set json_path, or $COURSEGEN_COURSES_JSON, "
            f"or place it at {self.DEFAULT_RELATIVE}"
        )

    # ---------------- IO ----------------
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

    # ---------------- Mapping / indexing ----------------
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

    def _make_course(self, row: Dict[str, Any]) -> CourseModel:
        return CourseModel(
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
        # Group rows by program signature (order-insensitive)
        groups: Dict[Tuple[str, ...], List[CourseModel]] = {}
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

            # index by code (first one wins; warn on duplicate with different title)
            if code_key in self._by_code:
                prev, _ = self._by_code[code_key]
                if prev.title != course.title:
                    logger.warning("Duplicate code with different title detected: %s | '%s' vs '%s'",
                                   course.code, prev.title, course.title)
                continue
            self._by_code[code_key] = (course, programs)

        # Build DepartmentModel list
        dept_models: List[DepartmentModel] = []
        for sig, courses in groups.items():
            dept_models.append(DepartmentModel(names=list(sig), courses=sorted(courses, key=lambda c: c.code)))
        # Stable sort departments by first program name then by number of courses desc
        self.courses = sorted(dept_models, key=lambda d: (d.names[0] if d.names else "", -len(d.courses)))

    # ---------------- Public API (compat + extras) ----------------
    def search_course(self, course_code: str) -> Tuple[CourseModel, List[str]]:
        """
        Return (CourseModel, offered_by_programs) for matching code (case-insensitive).
        Raises ValueError if not found.
        """
        key = (course_code or "").strip().lower()
        hit = self._by_code.get(key)
        if not hit:
            raise ValueError(f"Course code not found: '{course_code}'")
        return hit

    # Helpful extras (non-breaking)
    def find_by_title(self, needle: str, limit: int = 10) -> List[Tuple[CourseModel, List[str]]]:
        q = (needle or "").strip().lower()
        out: List[Tuple[CourseModel, List[str]]] = []
        for course, programs in self._by_code.values():
            if q in course.title.lower():
                out.append((course, programs))
                if len(out) >= limit:
                    break
        return out

    def list_by_level(self, level: str) -> List[Tuple[CourseModel, List[str]]]:
        lv = (level or "").strip().lower()
        return [(c, p) for c, p in self._by_code.values() if c.level.strip().lower() == lv]

    def list_by_program_contains(self, text: str) -> List[Tuple[CourseModel, List[str]]]:
        q = (text or "").strip().lower()
        hits: List[Tuple[CourseModel, List[str]]] = []
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


# -------------------------------------------------------------------
# Script usage (safe — runs only when executed directly)
# -------------------------------------------------------------------
if __name__ == "__main__":
    import os

    # You can override with env: COURSEGEN_COURSES_JSON=/path/to/courses.json
    json_path = os.environ.get("COURSEGEN_COURSES_JSON")
    df = DataFormatting(json_path)

    logger.info("Total department groups: %d", len(df.courses))
    logger.info("Total unique course codes: %d", len(df.all_codes()))

    # Demo: search by code (keeps your original API)
    try:
        course, programs = df.search_course("EEE 313")
        logger.info("Found: %s | %s | Level %s | Semester %s | Units %d | Elective=%s",
                    course.code, course.title, course.level, course.semester, course.units, course.is_elective)
        logger.info("Offered by: %s", ", ".join(programs) or "—")
    except ValueError as e:
        logger.error("%s", e)

    # Demo: fuzzy title search
    for c, progs in df.find_by_title("electrical machines")[:3]:
        logger.info("Title hit: %s (%s) — %s", c.title, c.code, ", ".join(progs))
