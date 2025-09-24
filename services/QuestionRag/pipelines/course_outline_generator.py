"""Course outline generation pipeline using Gemini models.

This module was migrated from ``gemini_question_gen.py`` to keep outline
generation concerns separate from question generation. The implementation is
unchanged: it still produces outlines first, updates ``courses.json``, caches
embedding presence, and leaves space for follow-up question generation.
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

# Local deps
from ..utils.chromadb_query import (
    ChromaQuery,
    MetaData,
    CHROMA_PATH,
    CHROMA_COLLECTION,
)
from ..utils.courses import DataFormatting  # keeps your search_course API

# =========================
# Global Config (env-overridable)
# =========================
TEMPERATURE = float(os.environ.get("GEN_QG_TEMPERATURE", "0.15"))
TOP_P = float(os.environ.get("GEN_QG_TOP_P", "0.9"))
THINKING_BUDGET = int(os.environ.get("GEN_QG_THINK_BUDGET", "12700"))
MAX_OUTPUT_TOKENS = int(os.environ.get("GEN_QG_MAX_OUT_TOKENS", "15500"))

GEMINI_THINKING_MODEL = os.environ.get("GEN_QG_THINK_MODEL", "gemini-2.5-flash-lite")
GEMMA_MODEL = os.environ.get("GEN_QG_BASE_MODEL", "gemma-3-27b-it")

# Outline retrieval
RAG_TAU = float(os.environ.get("GEN_QG_RAG_TAU", "0.35"))
RAG_MIN_SIM = float(os.environ.get("GEN_QG_RAG_MIN_SIM", "0.60"))
RAG_TOPK_PER_QUERY = int(os.environ.get("GEN_QG_RAG_TOPK", "10"))
RAG_MAX_TOTAL = int(os.environ.get("GEN_QG_RAG_MAX", "40"))

# Subtopic refinement via embeddings
ENABLE_SUBTOPIC_RAG = os.environ.get("GEN_QG_SUBTOPIC_RAG", "1").lower() in ("1", "true", "yes")
SUB_RAG_TAU = float(os.environ.get("GEN_QG_SUB_RAG_TAU", str(RAG_TAU)))
SUB_RAG_MIN_SIM = float(os.environ.get("GEN_QG_SUB_RAG_MIN_SIM", str(RAG_MIN_SIM)))
SUB_RAG_TOPK_PER_QUERY = int(os.environ.get("GEN_QG_SUB_RAG_TOPK", "8"))
SUB_RAG_FINAL_K = int(os.environ.get("GEN_QG_SUB_RAG_FINAL_K", "8"))

# Pacing controls to avoid provider overload
COURSE_DELAY_S = float(os.environ.get("GEN_QG_COURSE_DELAY_S", "2.0"))
TOPIC_DELAY_S = float(os.environ.get("GEN_QG_TOPIC_DELAY_S", "1.0"))
QUERY_DELAY_S = float(os.environ.get("GEN_QG_QUERY_DELAY_S", "0.5"))
DELAY_JITTER_FRAC = float(os.environ.get("GEN_QG_JITTER_FRAC", "0.25"))  # 0..1 of delay

# Files
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COURSES_JSON = REPO_ROOT / "data/textbooks/courses.json"
DEFAULT_CACHE_DIR = REPO_ROOT / "OUTPUT_DATA2/cache"

COURSES_JSON = Path(
    os.environ.get("COURSEGEN_COURSES_JSON", str(DEFAULT_COURSES_JSON))
).expanduser().resolve()

CACHE_DIR = Path(
    os.environ.get("COURSEGEN_CACHE_DIR", str(DEFAULT_CACHE_DIR))
).expanduser().resolve()

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Default output dir for Chroma-wide runs
DEFAULT_OUTLINE_DIR = CACHE_DIR / "outlines_by_chroma"
CHROMA_OUT_DIR = Path(
    os.environ.get("COURSEGEN_CHROMA_OUT_DIR", str(DEFAULT_OUTLINE_DIR))
).expanduser().resolve()
CHROMA_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logger = logging.getLogger("gemini_prod")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(os.environ.get("GEN_QG_LOG_LVL", "INFO").upper())


# =========================
# Helpers
# =========================
def _accumulate_stream(stream) -> str:
    out = []
    for chunk in stream:
        t = getattr(chunk, "text", None)
        if isinstance(t, str) and t:
            out.append(t)
            continue
        parts = getattr(chunk, "parts", None)
        if parts:
            for p in parts:
                pt = getattr(p, "text", None)
                if isinstance(pt, str) and pt:
                    out.append(pt)
    return "".join(out)


def _coerce_json(text: str) -> Union[List, Dict]:
    text = (text or "").strip()
    # fenced
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        cand = m.group(1).strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
    # raw
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # first {...} or [...]
    m2 = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if m2:
        cand = m2.group(1).strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            try:
                fixed = cand.replace("\\", "\\\\")
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
    raise ValueError("Failed to parse JSON from model output")


def _retry(fn, attempts=4, base=0.8, cap=8.0):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            if i == attempts - 1:
                break
            delay = min(cap, base * (2 ** i))
            logger.warning("Retry %d/%d after error: %s (sleep %.2fs)", i + 1, attempts, e, delay)
            time.sleep(delay)
    raise last


def _dept_from_code(course_code: str) -> str:
    # "EEE 315" -> "EEE"
    return (course_code.split()[0] if course_code else "").strip().upper()


def _sleep_with_jitter(base_s: float, jitter_frac: float = DELAY_JITTER_FRAC):
    if base_s <= 0:
        return
    try:
        j = max(0.0, float(jitter_frac))
    except Exception:
        j = 0.0
    low = max(0.0, base_s * (1.0 - j))
    high = base_s * (1.0 + j)
    import random as _rnd
    dur = _rnd.uniform(low, high)
    time.sleep(dur)


# =========================
# Minimal Model Client
# =========================
class ModelClient:
    def __init__(self, is_thinking: bool = True):
        if genai is None:
            raise ImportError("google-genai package is not installed. Install with: pip install google-genai")
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise EnvironmentError("GOOGLE_API_KEY is not set")
        self.client = genai.Client(api_key=key)
        self.is_thinking = is_thinking

    def generate_json(self, prompt: str) -> Union[List, Dict]:
        model = GEMINI_THINKING_MODEL if self.is_thinking else GEMMA_MODEL
        cfg = types.GenerateContentConfig(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET) if self.is_thinking else None,
        )
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]

        def _one():
            stream = self.client.models.generate_content_stream(model=model, contents=contents, config=cfg)
            raw = _accumulate_stream(stream)
            return _coerce_json(raw)

        return _retry(_one)


# =========================
# Prompt builders (Outline first)
# =========================
def _format_rag_block_numbered(hits: List[Dict[str, Any]], max_items: int = 16) -> Tuple[str, List[str]]:
    if not hits:
        return "### STUDY MATERIAL CONTEXT\nNo study material context found.", []
    lines = ["### STUDY MATERIAL CONTEXT"]
    ids: List[str] = []
    for i, it in enumerate(hits[:max_items], start=1):
        sid = f"S{i}"
        meta = it.get("meta") or it.get("metadata") or {}
        path = meta.get("path") or meta.get("FILENAME") or "unknown"
        cf = meta.get("COURSE_FOLDER") or ""
        page = meta.get("page")
        snippet = (it.get("snippet") or it.get("document") or "").strip()
        src = path + (f" [{cf}]" if cf else "") + (f" (p.{page})" if page is not None else "")
        snippet = re.sub(r"\s+", " ", snippet)[:900]
        lines.append(f"- [{sid}] Source: {src}")
        lines.append(f"  Snippet: {snippet}")
        ids.append(sid)
    return "\n".join(lines), ids


def _prompt_outline(rag_block: str, source_ids: List[str], course_title: str, department_str: str, level: str) -> str:
    return f"""
{rag_block}

You are to produce a short course description and an 8-topic outline using ONLY the context above.
Return a single JSON OBJECT and nothing else.

REQUIREMENTS:
- Course: "{course_title}" | Department: "{department_str}" | Level: "{level}"
- "description": 2–3 sentences grounded in the context.
- "topics": EXACTLY 8 items, each:
  - "title": short and precise
  - "subtopics": EXACTLY 5 focused phrases (no numbering/markdown)
  - "sources": at least 1 valid ID from: {", ".join(source_ids)}

SCHEMA:
{{
  "description": "string",
  "topics": [
    {{
      "title": "string",
      "subtopics": ["string","string","string","string","string"],
      "sources": ["S1","S3"]
    }}
  ]
}}

VALIDATION:
- No prose outside the JSON object.
- Every "sources" entry must be a valid ID from the list above.
""".strip()


# =========================
# Generator core (Outline + Questions)
# =========================
class GeminiQuestionGen:
    def __init__(self, is_thinking: bool = True):
        self.mc = ModelClient(is_thinking=is_thinking)
        self._cq = ChromaQuery()

    # Progressive retrieval for OUTLINE:
    # 1) DEPARTMENT-only hits (broad anchor)
    # 2) DEPARTMENT + COURSE_CODE hits (specific)
    # Prefer specific; backfill with dept hits if needed
    def _retrieve_outline_hits(
        self,
        department_code: str,
        course_code: str,
        course_title: str,
        level: str,
        variation: bool = True,
        allow_dept_fallback: bool = False,
    ) -> List[Dict[str, Any]]:
        cq = self._cq
        # Focused outline queries
        queries = [
            f"\"{course_title}\" syllabus, outline, modules, topics, subtopics, objectives",
            f"{course_title} {department_code} Level {level} table of contents, overview, summary, learning outcomes",
            f"{course_title} outline topics subtopics {department_code} objectives",
        ]
        hits_dept: List[Dict[str, Any]] = []
        hits_course: List[Dict[str, Any]] = []

        # Stage 1: Course-specific queries (default)
        # IMPORTANT: In our metadata, COURSE_CODE holds only the department code (e.g., "EEE"),
        # while COURSE_FOLDER is the full course identifier (e.g., "EEE 315").
        # Filtering by COURSE_CODE=course_code ("EEE 315") would never match.
        # Use COURSE_FOLDER for course-specific retrieval, optionally also scoping by DEPARTMENT.
        where_course = MetaData(DEPARTMENT=department_code, COURSE_FOLDER=course_code).to_where()
        for i, q in enumerate(queries):
            try:
                if variation and (i % 2 == 1):
                    res = cq.search_with_temperature(q, topk=RAG_TOPK_PER_QUERY, final_k=10, tau=RAG_TAU, min_sim=RAG_MIN_SIM, where=where_course, show_snippet=True)
                else:
                    res = cq.search(q, k=RAG_TOPK_PER_QUERY, where=where_course, show_snippet=True)
                hits_course.extend(res or [])
            except Exception as e:
                logger.warning("[RAG course] %s", e)
            finally:
                _sleep_with_jitter(QUERY_DELAY_S)

        # Stage 2: Department-only fallback (only if requested and course has no hits)
        if allow_dept_fallback and not hits_course:
            where_dept = MetaData(DEPARTMENT=department_code).to_where()
            for i, q in enumerate(queries):
                try:
                    if variation and (i % 2 == 0):
                        res = cq.search_with_temperature(q, topk=RAG_TOPK_PER_QUERY, final_k=10, tau=RAG_TAU, min_sim=RAG_MIN_SIM, where=where_dept, show_snippet=True)
                    else:
                        res = cq.search(q, k=RAG_TOPK_PER_QUERY, where=where_dept, show_snippet=True)
                    hits_dept.extend(res or [])
                except Exception as e:
                    logger.warning("[RAG dept] %s", e)
                finally:
                    _sleep_with_jitter(QUERY_DELAY_S)

        # Prefer specific hits; if empty, we still return dept hits so the caller can decide to skip or use dept-wide
        # BUT: the user wants to keep track of courses without embeddings and skip them → we treat "no course hits" as missing.
        # We still return combined for transparency.
        # Merge with dedupe by (path, chunk_index, snippet head)
        def key(it):
            m = it.get("meta") or it.get("metadata") or {}
            return (m.get("path") or m.get("FILENAME") or "", m.get("chunk_index"), (it.get("snippet") or it.get("document") or "")[:64])

        seen = set()
        out: List[Dict[str, Any]] = []
        for arr in [hits_course, hits_dept]:
            for it in arr:
                k = key(it)
                if k in seen:
                    continue
                seen.add(k)
                out.append(it)

        return out[:RAG_MAX_TOTAL], len(hits_course)  # return combined and count of course-specific hits

    # Topic-level retrieval for subtopic refinement
    def _retrieve_topic_hits(
        self,
        department_code: str,
        course_code: str,
        course_title: str,
        topic_title: str,
        *,
        variation: bool = True,
    ) -> List[Dict[str, Any]]:
        queries = [
            f"Subtopics for '{topic_title}' in {course_title} syllabus, outline, modules",
            f"{topic_title} {course_title} {department_code} examples, key concepts, learning objectives",
            f"{topic_title} {course_title} {department_code} topics subtopics",
        ]
        where_course = MetaData(DEPARTMENT=department_code, COURSE_FOLDER=course_code).to_where()
        out: List[Dict[str, Any]] = []
        for i, q in enumerate(queries):
            try:
                if variation and (i % 2 == 0):
                    res = self._cq.search_with_temperature(
                        q,
                        topk=max(SUB_RAG_TOPK_PER_QUERY, 8),
                        final_k=SUB_RAG_FINAL_K,
                        tau=SUB_RAG_TAU,
                        min_sim=SUB_RAG_MIN_SIM,
                        where=where_course,
                        show_snippet=True,
                    )
                else:
                    res = self._cq.search(
                        q,
                        k=max(SUB_RAG_TOPK_PER_QUERY, 8),
                        where=where_course,
                        show_snippet=True,
                    )
                out.extend(res or [])
            except Exception as e:
                logger.warning("[RAG subtopics] %s — %s", topic_title, e)
            finally:
                _sleep_with_jitter(QUERY_DELAY_S)

        # Dedupe by (path, chunk_index, snippet head)
        def key(it):
            m = it.get("meta") or it.get("metadata") or {}
            return (m.get("path") or m.get("FILENAME") or "", m.get("chunk_index"), (it.get("snippet") or it.get("document") or "")[:64])

        seen = set()
        merged: List[Dict[str, Any]] = []
        for it in out:
            k = key(it)
            if k in seen:
                continue
            seen.add(k)
            merged.append(it)
        # keep a compact context
        return merged[:SUB_RAG_FINAL_K]

    def _prompt_subtopics(self, rag_block: str, topic_title: str, course_title: str, level: str) -> str:
        return f"""
{rag_block}

Using ONLY the study material snippets above, produce 5 precise subtopics for the topic:
Topic: "{topic_title}" (Course: "{course_title}" | Level: "{level}")

Return a single JSON ARRAY of EXACTLY 5 short strings. No numbering, no markdown, no extra text.
""".strip()

    def _format_topic_rag(self, hits: List[Dict[str, Any]]) -> str:
        if not hits:
            return "### STUDY MATERIAL CONTEXT\nNo study material context found."
        lines = ["### STUDY MATERIAL CONTEXT"]
        for it in hits:
            meta = it.get("meta") or it.get("metadata") or {}
            path = meta.get("path") or meta.get("FILENAME") or "unknown"
            cf = meta.get("COURSE_FOLDER") or ""
            page = meta.get("page")
            snippet = (it.get("snippet") or it.get("document") or "").strip()
            snippet = re.sub(r"\s+", " ", snippet)[:900]
            src = path + (f" [{cf}]" if cf else "") + (f" (p.{page})" if page is not None else "")
            lines.append(f"- Source: {src}")
            lines.append(f"  Snippet: {snippet}")
        return "\n".join(lines)

    def _refine_subtopics_with_embeddings(
        self,
        *,
        course_title: str,
        course_code: str,
        department_code: str,
        level: str,
        topics: List[Dict[str, Any]],
        variation: bool = True,
    ) -> None:
        if not ENABLE_SUBTOPIC_RAG:
            return
        for t in topics:
            title = str(t.get("title") or "").strip()
            if not title:
                continue
            try:
                topic_hits = self._retrieve_topic_hits(
                    department_code=department_code,
                    course_code=course_code,
                    course_title=course_title,
                    topic_title=title,
                    variation=variation,
                )
                rag_block = self._format_topic_rag(topic_hits)
                prompt = self._prompt_subtopics(rag_block, title, course_title, level)
                cand = self.mc.generate_json(prompt)
                if isinstance(cand, list):
                    # Coerce to 5 strings
                    out = []
                    for x in cand:
                        s = str(x or "").strip()
                        if s:
                            out.append(s)
                        if len(out) >= 5:
                            break
                    if out:
                        t["subtopics"] = (out[:5] + ["TBD"] * 5)[:5]
                        logger.info("[Subtopic RAG] Refined '%s' → %d items", title, len(out))
            except Exception as e:
                logger.warning("[Subtopic RAG] '%s' — %s", title, e)
            finally:
                _sleep_with_jitter(TOPIC_DELAY_S)

    def generate_outline_for_course(
        self,
        course_title: str,
        course_code: str,
        department_code: str,
        level: str,
        department_str_for_prompt: str,
        variation: bool = True,
        allow_dept_fallback: bool = False,
    ) -> Optional[Dict[str, Any]]:
        hits, course_hit_count = self._retrieve_outline_hits(
            department_code,
            course_code,
            course_title,
            level,
            variation=variation,
            allow_dept_fallback=allow_dept_fallback,
        )
        logger.info("[RAG] %s — course_hits=%d total_hits=%d", course_code, course_hit_count, len(hits))

        # If we strictly require course-level embeddings, skip when none found.
        # When allow_dept_fallback=True, continue with department-level hits.
        if course_hit_count == 0:
            if allow_dept_fallback and hits:
                logger.info("[Outline] No course-specific embeddings for %s — using department-level fallback", course_code)
            else:
                logger.info("[Outline] No course-specific embeddings for %s — skipping", course_code)
                return None

        rag_block, source_ids = _format_rag_block_numbered(hits, max_items=16)
        prompt = _prompt_outline(rag_block, source_ids, course_title, department_str_for_prompt, level)
        data = self.mc.generate_json(prompt)
        if not isinstance(data, dict):
            logger.warning("[Outline] Model did not return an object for %s", course_code)
            return None

        # Light fix-ups
        desc = str(data.get("description") or "").strip()
        topics = data.get("topics") or []
        if not desc or not isinstance(topics, list) or len(topics) == 0:
            logger.warning("[Outline] Missing fields for %s", course_code)
            return None

        # Ensure 8 topics, 5 subs each, keep sources as-is
        topics = topics[:8] + []
        while len(topics) < 8:
            topics.append({"title": "TBD", "subtopics": ["TBD"] * 5, "sources": source_ids[:1]})
        for t in topics:
            subs = t.get("subtopics") or []
            t["subtopics"] = (subs[:5] + ["TBD"] * 5)[:5]
            if not t.get("sources"):
                t["sources"] = source_ids[:1]

        # Optional: refine subtopics with embedding-guided RAG for each topic
        try:
            self._refine_subtopics_with_embeddings(
                course_title=course_title,
                course_code=course_code,
                department_code=department_code,
                level=level,
                topics=topics,
                variation=variation,
            )
        except Exception as e:
            logger.warning("[Subtopic RAG] Skipped refinement for %s — %s", course_code, e)

        return {"description": desc, "topics": topics}


# =========================
# Course JSON updater + cache
# =========================
class CourseStore:
    def __init__(self, json_path: Path):
        self.path = json_path
        self._load()

    def _load(self):
        if not self.path.exists():
            raise FileNotFoundError(f"courses.json not found at {self.path}")
        with self.path.open("r", encoding="utf-8") as f:
            self.data: List[Dict[str, Any]] = json.load(f)
        if not isinstance(self.data, list):
            raise ValueError("courses.json root must be a JSON array")

    def save(self, backup_once: bool = True):
        # Backup once per process run
        bak = self.path.with_suffix(self.path.suffix + ".bak")
        if backup_once and not bak.exists():
            shutil.copy2(self.path, bak)
            logger.info("Backup saved: %s", bak)

        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)
        logger.info("courses.json updated: %s", self.path)

    def iter_courses_by_department_prefix(self, dept_code: str):
        pref = (dept_code or "").upper().strip()
        for row in self.data:
            code = str(row.get("code", "")).strip()
            if code.upper().startswith(pref):
                yield row

    def update_outline(self, course_code: str, outline_obj: Dict[str, Any]):
        for row in self.data:
            if str(row.get("code", "")).strip().lower() == course_code.strip().lower():
                # Write to "description" and "outline" fields
                row["description"] = outline_obj.get("description")
                row["outline"] = outline_obj.get("topics")
                # Optional: keep "outline_sources" flattened (useful for audits)
                all_sources = sorted({s for t in outline_obj.get("topics", []) for s in (t.get("sources") or [])})
                if all_sources:
                    row["outline_sources"] = all_sources
                return True
        return False


class OutlineCache:
    """Tracks which course codes have course-specific embeddings and which do not."""
    def __init__(self, dept_code: str, cache_dir: Path = CACHE_DIR):
        self.dept = dept_code.upper().strip()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = cache_dir / f"outline_cache_{self.dept}.json"
        self._load()

    def _load(self):
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        else:
            obj = {}
        self.present: Dict[str, str] = obj.get("present", {})  # code -> ISO datetime
        self.missing: Dict[str, str] = obj.get("missing", {})  # code -> ISO datetime

    def save(self):
        obj = {
            "department": self.dept,
            "present": self.present,
            "missing": self.missing,
            "present_count": len(self.present),
            "missing_count": len(self.missing),
            "last_write": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        logger.info("Cache saved: %s (present=%d, missing=%d)", self.path, len(self.present), len(self.missing))

    def mark_present(self, course_code: str):
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.present[course_code] = ts
        if course_code in self.missing:
            self.missing.pop(course_code, None)

    def mark_missing(self, course_code: str):
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.missing[course_code] = ts
        if course_code in self.present:
            self.present.pop(course_code, None)

    def forget(self, course_code: str):
        """Clear cached state for a course so it will be re-evaluated."""
        self.present.pop(course_code, None)
        self.missing.pop(course_code, None)

    def is_missing(self, course_code: str, ttl_hours: float | int | None = None) -> bool:
        """Return True if course is marked missing and the mark is still valid.

        When ttl_hours > 0, a missing mark older than TTL is treated as expired (i.e., not missing).
        """
        if course_code not in self.missing:
            return False
        if not ttl_hours or float(ttl_hours) <= 0:
            return True
        try:
            ts = self.missing.get(course_code)
            if not ts:
                return False
            # parse simple ISO-like timestamp: YYYY-mm-ddTHH:MM:SS
            t_struct = time.strptime(ts.split(".")[0], "%Y-%m-%dT%H:%M:%S")
            t_mark = time.mktime(t_struct)
            age_sec = time.time() - t_mark
            return age_sec < (float(ttl_hours) * 3600.0)
        except Exception:
            # be permissive: if we can't parse, assume still missing
            return True

    def is_present(self, course_code: str) -> bool:
        return course_code in self.present


class OutlineProgress:
    """Lightweight per-department progress ledger for outline generation."""
    def __init__(self, dept_code: str, cache_dir: Path = CACHE_DIR):
        self.dept = dept_code.upper().strip()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = cache_dir / f"outline_progress_{self.dept}.json"
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    self._data = obj.get("courses", {}) or {}
            except Exception:
                self._data = {}

    def save(self):
        out = {
            "department": self.dept,
            "last_write": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "courses": self._data,
        }
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    def update(self, course_code: str, **fields: Any):
        row = self._data.setdefault(course_code, {})
        row.update(fields)
        row.setdefault("attempts", 0)
        row["last_attempt"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._data[course_code] = row


class ChromaCourseProgress:
    """Resume ledger for Chroma-wide runs keyed by COURSE_FOLDER.

    Tracks a compact signature to detect when a course's underlying embeddings changed.
    """
    def __init__(self, cache_dir: Path = CACHE_DIR):
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = cache_dir / "chroma_course_progress.json"
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    self._data = obj.get("courses", {}) or {}
            except Exception:
                self._data = {}

    def save(self):
        out = {
            "last_write": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "courses": self._data,
        }
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _sig_equal(a: Dict[str, Any] | None, b: Dict[str, Any] | None) -> bool:
        if not a or not b:
            return False
        # Compare essential fields; ignore timestamps/status
        return (
            a.get("max_mtime") == b.get("max_mtime")
            and a.get("doc_paths_count") == b.get("doc_paths_count")
            and (a.get("file_hashes") or []) == (b.get("file_hashes") or [])
        )

    def up_to_date(self, course_folder: str, signature: Dict[str, Any]) -> bool:
        row = self._data.get(course_folder)
        if not row:
            return False
        return self._sig_equal(row.get("signature"), signature)

    def update(self, course_folder: str, signature: Dict[str, Any], **fields: Any):
        row = self._data.setdefault(course_folder, {})
        row.update(fields)
        row["signature"] = signature
        row.setdefault("attempts", 0)
        row["last_attempt"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._data[course_folder] = row


def _sanitize_filename(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[^A-Za-z0-9_.\- ]+", "_", name)
    name = name.replace("/", "_")
    return name or "course"


# =========================
# Department Orchestrator (Outlines-first)
# =========================
class DepartmentRunner:
    def __init__(self, courses_json: Path = COURSES_JSON, is_thinking: bool = False):
        self.store = CourseStore(courses_json)
        self.gen = GeminiQuestionGen(is_thinking=is_thinking)
        self._chroma = ChromaQuery()

    def _course_has_embeddings(self, dept_code: str, course_code: str) -> bool:
        """Return True when Chroma currently has course-specific embeddings."""
        where = MetaData(DEPARTMENT=dept_code, COURSE_FOLDER=course_code).to_where()
        try:
            result = self._chroma.col.get(where=where, limit=1)
        except Exception as exc:
            logger.warning("[Cache check] %s — failed to query Chroma: %s", course_code, exc)
            return False

        ids = result.get("ids") or []
        # Flatten [[id]] structure emitted by Chroma Python client
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return bool(ids)

    def build_outlines_for_department(
        self,
        any_course_code_in_department: str,
        *,
        skip_existing: bool = True,
        variation: bool = True,
        save_each_write: bool = True,
        ignore_missing_cache: bool = False,
        allow_dept_fallback: bool = False,
        missing_ttl_hours: float = 0.0,
        only_missing: bool = False,
        dry_run: bool = False,
    ):
        dept_code = _dept_from_code(any_course_code_in_department)
        if not dept_code:
            raise ValueError("Could not infer department code from the provided course code string.")
        cache = OutlineCache(dept_code)
        prog = OutlineProgress(dept_code)

        # For prompts, we still want a readable department string; use dept code
        dept_str_for_prompt = dept_code

        count_done = count_skipped = count_missing = count_errors = 0
        t0 = time.time()

        rows = list(self.store.iter_courses_by_department_prefix(dept_code))
        logger.info("Dept %s: found %d courses to evaluate", dept_code, len(rows))
        for row in rows:
            course_code = str(row.get("code", "")).strip()
            course_title = str(row.get("title", "")).strip()
            level = ""
            lv = row.get("levels")
            if isinstance(lv, list) and lv:
                level = str(lv[0])
            elif isinstance(lv, str):
                level = lv

            # Skip if already present and we’re skipping existing
            if skip_existing and row.get("outline") and row.get("description"):
                logger.info("[Skip existing] %s — already has outline+description", course_code)
                count_skipped += 1
                prog.update(course_code, status="skipped_existing")
                continue

            if only_missing and not cache.is_missing(course_code, ttl_hours=missing_ttl_hours):
                # Only process those currently marked missing (and not expired if TTL > 0)
                logger.info("[Skip non-missing] %s — only_missing is set", course_code)
                count_skipped += 1
                prog.update(course_code, status="skipped_not_missing")
                continue

            if cache.is_missing(course_code, ttl_hours=missing_ttl_hours):
                if ignore_missing_cache:
                    logger.info("[Recheck cached-missing] %s — ignoring cache and querying again", course_code)
                else:
                    if self._course_has_embeddings(dept_code, course_code):
                        logger.info("[Cache refresh] %s — new embeddings detected; clearing missing mark", course_code)
                        cache.forget(course_code)
                        prog.update(course_code, status="recheck_after_missing")
                    else:
                        logger.info("[Skip cached-missing] %s — previously had no embeddings", course_code)
                        count_missing += 1
                        prog.update(course_code, status="missing_cached")
                        continue

            logger.info("=== Generating outline for %s — %s (Level %s, Dept %s)", course_code, course_title, level, dept_code)
            t_course = time.time()
            try:
                if dry_run:
                    # Fetch hits only to report availability
                    hits, course_hit_count = self.gen._retrieve_outline_hits(
                        dept_code,
                        course_code,
                        course_title,
                        level or "",
                        variation=variation,
                        allow_dept_fallback=allow_dept_fallback,
                    )
                    logger.info("[Dry run] %s — course_hits=%d total_hits=%d", course_code, course_hit_count, len(hits))
                    prog.update(course_code, status="dry_run", course_hit_count=course_hit_count, total_hits=len(hits))
                    continue
                outline = self.gen.generate_outline_for_course(
                    course_title=course_title,
                    course_code=course_code,
                    department_code=dept_code,
                    level=level or "",
                    department_str_for_prompt=dept_str_for_prompt,
                    variation=variation,
                    allow_dept_fallback=allow_dept_fallback,
                )
            except Exception as e:
                logger.warning("[Error] %s — %s", course_code, e)
                prog.update(course_code, status="error", last_error=str(e))
                count_errors += 1
                continue

            if outline is None:
                logger.info("[No embeddings] %s — marking as missing", course_code)
                cache.mark_missing(course_code)
                prog.update(course_code, status="missing")
                count_missing += 1
                continue

            ok = self.store.update_outline(course_code, outline)
            if ok:
                cache.mark_present(course_code)
                topics_count = len(outline.get("topics", []) or [])
                logger.info("[Updated] %s — topics=%d elapsed=%.2fs", course_code, topics_count, time.time() - t_course)
                prog.update(course_code, status="present", topics=topics_count)
                count_done += 1
                if save_each_write:
                    self.store.save(backup_once=True)
                    cache.save()
                    prog.save()
            else:
                logger.warning("[Update failed] Could not update JSON for %s", course_code)
                prog.update(course_code, status="error", last_error="update_failed")
                count_errors += 1

            # Pace between courses to avoid provider overload
            _sleep_with_jitter(COURSE_DELAY_S)

        # Final save
        self.store.save(backup_once=True)
        cache.save()
        prog.save()
        logger.info("Finished dept %s: done=%d, skipped=%d, missing=%d, errors=%d, total_time=%.2fs",
                    dept_code, count_done, count_skipped, count_missing, count_errors, time.time() - t0)


class ChromaCoursesRunner:
    """Enumerate all unique COURSE_FOLDER groups found in Chroma and generate outlines.

    Outputs per-course JSON files under CHROMA_OUT_DIR and tracks resume state via ChromaCourseProgress.
    Also attempts to update courses.json when a matching code exists.
    """
    def __init__(self, courses_json: Path | None = None, is_thinking: bool = False):
        self.store = CourseStore(courses_json) if (courses_json and Path(courses_json).exists()) else None
        self.gen = GeminiQuestionGen(is_thinking=is_thinking)
        self._chroma = ChromaQuery()

    def _flatten_metas(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            cnt = self._chroma.col.count()
        except Exception:
            cnt = None
        try:
            n = min(limit, cnt) if (limit is not None and cnt is not None) else (limit or cnt or 50000)
            got = self._chroma.col.get(limit=n, include=["metadatas"])  # type: ignore[arg-type]
        except Exception as e:
            logger.warning("[Chroma enumerate] get failed: %s", e)
            # last resort: peek
            got = self._chroma.col.peek()  # includes ids/documents/metadatas small sample
        metas = (got.get("metadatas") or [])
        if metas and isinstance(metas[0], list):
            metas = metas[0]
        # Coerce dicts only
        return [dict(m or {}) for m in metas if isinstance(m, dict)]

    @staticmethod
    def _signature_for_course(metas: List[Dict[str, Any]]) -> Dict[str, Any]:
        paths = set()
        file_hashes = set()
        max_mtime = 0
        for m in metas:
            p = (m.get("path") or m.get("FILENAME") or "").strip()
            if p:
                paths.add(p)
            fh = (m.get("file_hash") or "").strip()
            if fh:
                file_hashes.add(fh)
            try:
                mt = int(m.get("file_mtime") or 0)
                if mt > max_mtime:
                    max_mtime = mt
            except Exception:
                pass
        return {
            "doc_paths_count": len(paths),
            "file_hashes": sorted(file_hashes) if file_hashes else [],
            "max_mtime": max_mtime,
        }

    def _group_by_course_folder(self, metas: List[Dict[str, Any]]):
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for m in metas:
            cf = (m.get("COURSE_FOLDER") or m.get("COURSE_CODE") or "").strip()
            if not cf:
                continue
            groups.setdefault(cf, []).append(m)
        return groups

    def build_outlines_for_all_courses(
        self,
        *,
        skip_up_to_date: bool = True,
        variation: bool = True,
        save_each_write: bool = True,
        allow_dept_fallback: bool = False,
        dry_run: bool = False,
        output_dir: Path = CHROMA_OUT_DIR,
    ) -> None:
        t0 = time.time()
        prog = ChromaCourseProgress(cache_dir=CACHE_DIR)
        metas = self._flatten_metas()
        groups = self._group_by_course_folder(metas)
        logger.info("[Chroma scan] discovered %d course folders", len(groups))

        done = skipped = errors = 0
        output_dir.mkdir(parents=True, exist_ok=True)

        for course_folder, mlist in groups.items():
            # Signature and resume check
            sig = self._signature_for_course(mlist)
            if skip_up_to_date and prog.up_to_date(course_folder, sig):
                skipped += 1
                continue

            # Synthesize fields for prompt
            course_code = course_folder
            department_code = _dept_from_code(course_folder)
            # Try to guess title/level from metadata samples
            sample = mlist[0] if mlist else {}
            course_title = sample.get("COURSE_TITLE") or course_folder
            level = sample.get("LEVEL") or ""
            dept_str_for_prompt = sample.get("DEPARTMENT") or department_code

            try:
                prog.update(course_folder, sig, status="processing")
                if dry_run:
                    logger.info("[Dry run] Would generate outline for %s", course_folder)
                    done += 1
                    continue

                outline = self.gen.generate_outline_for_course(
                    course_title=course_title,
                    course_code=course_code,
                    department_code=department_code,
                    level=level,
                    department_str_for_prompt=dept_str_for_prompt,
                    variation=variation,
                    allow_dept_fallback=allow_dept_fallback,
                )
                if outline is None:
                    prog.update(course_folder, sig, status="missing")
                    skipped += 1
                    continue

                # Write per-course JSON
                fname = _sanitize_filename(course_folder) + ".json"
                out_path = output_dir / fname
                payload = {
                    "course_folder": course_folder,
                    "course_title": course_title,
                    "department": department_code,
                    "level": level,
                    "description": outline.get("description"),
                    "topics": outline.get("topics"),
                    "outline_sources": sorted({s for t in outline.get("topics", []) for s in (t.get("sources") or [])}),
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

                # Try to update courses.json if store provided
                if self.store is not None:
                    try:
                        updated = self.store.update_outline(course_code, outline)
                        if updated and save_each_write:
                            self.store.save(backup_once=True)
                    except Exception as e:
                        logger.warning("[Chroma scan] update courses.json failed for %s — %s", course_folder, e)

                prog.update(course_folder, sig, status="present", topics=len(outline.get("topics", []) or []))
                done += 1
                if save_each_write:
                    prog.save()
            except Exception as e:
                errors += 1
                prog.update(course_folder, sig, status="error", last_error=str(e))
                if save_each_write:
                    prog.save()
                logger.warning("[Chroma scan] %s — %s", course_folder, e)

            # Pace between courses
            _sleep_with_jitter(COURSE_DELAY_S)

        prog.save()
        logger.info("[Chroma scan] finished: done=%d skipped=%d errors=%d total=%d elapsed=%.2fs",
                    done, skipped, errors, len(groups), time.time() - t0)


# =========================
# (Optional) Questions phase — after outlines are ready
# =========================
# You can later add a pass here that reads the cache.present keys and runs your
# existing question generation routines for those course codes only.


# =========================
# CLI
# =========================
def main(argv: Optional[List[str]] = None) -> None:
    """Run the legacy CLI for generating course outlines."""

    import argparse

    parser = argparse.ArgumentParser(description="Generate course outlines first, update courses.json, cache embedding presence.")
    parser.add_argument("--department_from", required=False, help='Any course code string from the department, e.g. "EEE 315" → uses "EEE"')
    parser.add_argument("--courses_json", default=str(COURSES_JSON), help="Path to courses.json")
    parser.add_argument("--skip_existing", action="store_true", default=True, help="Skip courses that already have outline + description")
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    parser.add_argument("--variation", action="store_true", default=True, help="Allow temperature sampling for retrieval variety")
    parser.add_argument("--save_each_write", action="store_true", default=True, help="Save JSON and cache after each course")
    parser.add_argument("--ignore_missing_cache", action="store_true", default=False, help="Do not skip courses marked as missing; requery Chroma")
    parser.add_argument("--allow_dept_fallback", action="store_true", default=False, help="If no course-specific hits, use department-level hits instead of skipping")
    parser.add_argument("--missing_ttl_hours", type=float, default=0.0, help="Expiry for 'missing' cache entries; 0 means never expire")
    parser.add_argument("--only_missing", action="store_true", default=False, help="Process only courses currently marked as missing (honors TTL)")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Do not generate or write outlines; just log hit availability")
    parser.add_argument("--thinking", action="store_true", default=False, help="Use thinking model mode")
    parser.add_argument("--scan_chroma_all", action="store_true", default=True, help="Enumerate all COURSE_FOLDER in Chroma and generate outlines")
    parser.add_argument("--output_dir", default=str(CHROMA_OUT_DIR), help="Output directory for per-course JSON when scanning Chroma")
    args = parser.parse_args(argv)

    logger.info("Chroma config: path='%s' collection='%s'", CHROMA_PATH, CHROMA_COLLECTION)
    try:
        _cq = ChromaQuery()
        logger.info(
            "Chroma connected OK: path='%s' collection='%s' count=%d",
            _cq.chroma_path,
            _cq.collection_name,
            _cq.col.count(),
        )
    except Exception as e:
        logger.warning("Chroma connection check failed: %s", e)

    if args.scan_chroma_all:
        runner = ChromaCoursesRunner(courses_json=Path(args.courses_json), is_thinking=args.thinking)
        runner.build_outlines_for_all_courses(
            skip_up_to_date=args.skip_existing,
            variation=args.variation,
            save_each_write=args.save_each_write,
            allow_dept_fallback=args.allow_dept_fallback,
            dry_run=args.dry_run,
            output_dir=Path(args.output_dir),
        )
    else:
        if not args.department_from:
            parser.error("--department_from is required when not using --scan_chroma_all")
        runner = DepartmentRunner(courses_json=Path(args.courses_json), is_thinking=args.thinking)
        runner.build_outlines_for_department(
            args.department_from,
            skip_existing=args.skip_existing,
            variation=args.variation,
            save_each_write=args.save_each_write,
            ignore_missing_cache=args.ignore_missing_cache,
            allow_dept_fallback=args.allow_dept_fallback,
            missing_ttl_hours=args.missing_ttl_hours,
            only_missing=args.only_missing,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    main()


# python services/QuestionRag/course_outline_generator.py --scan_chroma_all
