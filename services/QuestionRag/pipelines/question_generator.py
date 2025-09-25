"""Gemini powered question generation pipeline using RAG + Firestore persistence."""

from __future__ import annotations

import argparse
import json
import re
import logging
import os
import random
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, Field, ValidationError

from data_models.gemini_config import GeminiConfig
from data_models.question_model import Question
from services.Gemini.gemini_service import GeminiService

from ..utils import ChromaQuery, MetaData, QuestionCache
from ..utils.batch_utils import (
    validate_answer_in_options,
    validate_options,
    write_jsonl,
)

try:  # pragma: no cover - Firestore is optional in tests
    from ...Firestore.firebase_service import FireStore  # type: ignore
except Exception:  # pragma: no cover - keep optional dependency soft
    FireStore = None  # type: ignore


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(os.environ.get("COURSEGEN_QG_LOGLEVEL", "INFO").upper())


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COURSES_JSON = Path(
    os.environ.get(
        "COURSEGEN_COURSES_JSON", str(REPO_ROOT / "data/textbooks/courses.json")
    )
).expanduser()
DEFAULT_CACHE_ROOT = Path(
    os.environ.get("COURSEGEN_CACHE_DIR", str(REPO_ROOT / "OUTPUT_DATA2/cache"))
).expanduser()
DEFAULT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = os.environ.get("COURSEGEN_QUESTION_MODEL", "gemini-2.5-flash-lite")


class QuestionGenerationError(RuntimeError):
    """Raised when the model returns invalid or incomplete data."""


@dataclass(frozen=True)
class RequestPlan:
    """Description of an individual Gemini request within a subtopic run."""

    name: str
    kind: str
    question_count: int
    difficulty_rank: int


@dataclass
class QuestionBatchConfig:
    """Configuration holder for question batch generation."""

    course_code: str
    courses_json_path: Path = DEFAULT_COURSES_JSON
    cache_dir: Path = DEFAULT_CACHE_ROOT
    rag_topk: int = 30
    rag_final_k: int = 12
    rag_tau: float = 0.35
    rag_min_similarity: float = 0.6
    rag_where: Optional[Dict[str, Any]] = None
    theory_questions_per_request: int = 10
    calc_questions_per_request: int = 5
    resume: bool = True
    store_firestore: bool = True
    request_delay_s: float = 1.5
    delay_jitter: float = 0.25
    gemini_model: str = DEFAULT_MODEL
    gemini_temperature: float = 0.25
    gemini_top_p: float = 0.85
    gemini_max_output_tokens: int = 6000
    request_attempts: int = 2
    rag_attempts: int = 2
    rag_context_limit: int = 8
    latex_wrap_steps: bool = True
    target_topics: Optional[Sequence[str]] = None
    target_subtopics: Optional[Sequence[str]] = None
    output_path: Optional[Path] = None
    custom_plan: Optional[List[RequestPlan]] = None

    def normalized_topics(self) -> Optional[set[str]]:
        if self.target_topics is None:
            return None
        return {t.strip().lower() for t in self.target_topics if str(t).strip()}

    def normalized_subtopics(self) -> Optional[set[str]]:
        if self.target_subtopics is None:
            return None
        return {t.strip().lower() for t in self.target_subtopics if str(t).strip()}

    def request_plan(self) -> List[RequestPlan]:
        if self.custom_plan is not None:
            return list(self.custom_plan)
        # Generate exactly 20 questions per subtopic: 10 theory + 10 calculation
        return [
            RequestPlan(
                name="theory-1",
                kind="theory",
                question_count=10,  # Exactly 10 theory questions per request
                difficulty_rank=4,
            ),
            RequestPlan(
                name="calculation-1",
                kind="calculation",
                question_count=10,  # Exactly 10 calculation questions per request
                difficulty_rank=6,
            ),
        ]


class GeminiGeneratedQuestion(BaseModel):
    """Schema describing the expected Gemini JSON payload."""

    question: str = Field(..., description="Main question text")
    options: List[str] = Field(
        ...,
        min_length=4,
        max_length=4,
    )
    correct_answer: str = Field(..., description="Correct option letter (A-D)")
    correct_answer_text: Optional[str] = Field(
        None, description="Correct option text (fallback if letter missing)"
    )
    explanation: str = Field(..., description="Grounded explanation")
    solution_steps: Optional[List[str]] = Field(
        default=None,
        description="Ordered list of solution steps for calculations",
    )


class GeminiQuestionBatch(BaseModel):
    questions: List[GeminiGeneratedQuestion] = Field(
        ...,
        min_length=1,
        description="Collection of questions returned from Gemini",
    )


class QuestionGenerator:
    """Generate MCQ questions using Gemini with RAG context and caching."""

    def __init__(
        self,
        *,
        gemini_service: Optional[GeminiService] = None,
        rag_client: Optional[ChromaQuery] = None,
        firestore: Optional[Any] = None,
        use_structured: Optional[bool] = None,
    ) -> None:
        # Initialize Gemini service with explicit API keys to avoid env var fallback
        if gemini_service is None:
            from services.Gemini.gemini_api_keys import GeminiApiKeys
            from services.Gemini.api_key_manager import ApiKeyManager

            gemini_keys = GeminiApiKeys()
            api_keys = gemini_keys.get_keys()
            api_key_manager = ApiKeyManager(api_keys)

            self.gemini = GeminiService(api_key_manager=api_key_manager)
        else:
            self.gemini = gemini_service

        self.rag = rag_client or ChromaQuery()
        self._firestore = firestore
        self._cache_map: Dict[Path, QuestionCache] = {}
        self._course_store: Dict[Path, List[Dict[str, Any]]] = {}
        if use_structured is None:
            env_flag = os.environ.get("COURSEGEN_USE_STRUCTURED", "0").lower()
            use_structured = env_flag in ("1", "true", "yes")
        self.use_structured = bool(use_structured)

    # ------------------------------------------------------------------
    # Public orchestrators
    # ------------------------------------------------------------------
    def generate_course_questions(self, config: QuestionBatchConfig) -> List[Question]:
        # If no course code specified, process all courses from courses.json
        if not config.course_code or config.course_code.lower() == "all":
            return self._generate_all_courses_questions(config)

        # Single course mode
        course = self._load_course(config.courses_json_path, config.course_code)
        return self._generate_single_course_questions(config, course)

    def _generate_all_courses_questions(self, config: QuestionBatchConfig) -> List[Question]:
        """Generate questions for all courses in courses.json."""
        logger.info("Generating questions for all courses in courses.json")

        courses_path = config.courses_json_path
        if not courses_path.exists():
            raise ValueError(f"Courses file not found: {courses_path}")

        # Load all courses
        data = json.loads(courses_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("courses.json must be a list of course objects")

        courses = [row for row in data if row.get("outline")]  # Only courses with outlines
        if not courses:
            logger.warning("No courses with outlines found in %s", courses_path)
            return []

        logger.info(f"Found {len(courses)} courses with outlines")

        all_results: List[Question] = []
        for course in courses:
            try:
                course_results = self._generate_single_course_questions(config, course)
                all_results.extend(course_results)
                logger.info(f"Generated {len(course_results)} questions for {course.get('code', 'unknown')}")
            except Exception as exc:
                logger.error(f"Failed to generate questions for course {course.get('code', 'unknown')}: {exc}")
                continue

        return all_results

    def _generate_single_course_questions(self, config: QuestionBatchConfig, course: Dict[str, Any]) -> List[Question]:
        """Generate questions for a single course."""
        outline = course.get("outline") or []
        normalized_topics = config.normalized_topics()
        normalized_subtopics = config.normalized_subtopics()

        results: List[Question] = []
        for topic in outline:
            topic_title = str(topic.get("title") or "").strip()
            if normalized_topics and topic_title.lower() not in normalized_topics:
                logger.debug("Skipping topic '%s' not in filter", topic_title)
                continue

            for subtopic in topic.get("subtopics") or []:
                subtopic_title = str(subtopic).strip()
                if not subtopic_title:
                    continue
                if normalized_subtopics and subtopic_title.lower() not in normalized_subtopics:
                    logger.debug(
                        "Skipping subtopic '%s' under '%s' due to filter",
                        subtopic_title,
                        topic_title,
                    )
                    continue
                generated = self._generate_for_subtopic(
                    config=config,
                    course=course,
                    topic_title=topic_title,
                    subtopic_title=subtopic_title,
                )
                results.extend(generated)
        return results

    # ------------------------------------------------------------------
    # Subtopic pipeline
    # ------------------------------------------------------------------
    def _generate_for_subtopic(
        self,
        *,
        config: QuestionBatchConfig,
        course: Dict[str, Any],
        topic_title: str,
        subtopic_title: str,
    ) -> List[Question]:
        cache = self._cache_for(config.cache_dir)
        plan = config.request_plan()
        rag_contexts = self._retrieve_rag_context(
            course=course,
            topic_title=topic_title,
            subtopic_title=subtopic_title,
            config=config,
        )

        if not rag_contexts:
            logger.warning(
                "No RAG context found for %s - %s (%s); skipping",
                course.get("code"),
                topic_title,
                subtopic_title,
            )
            meta = {
                "course_code": course.get("code"),
                "topic": topic_title,
                "subtopic": subtopic_title,
                "reason": "rag_empty",
            }
            for request in plan:
                key = cache.make_key(course.get("code", ""), topic_title, subtopic_title, request.name)
                cache.mark_skipped(key, reason="rag_empty", meta=meta)
            return []

        questions: List[Question] = []
        for idx, request in enumerate(plan):
            key = cache.make_key(
                course.get("code", ""), topic_title, subtopic_title, request.name
            )
            meta = {
                "course_code": course.get("code"),
                "topic": topic_title,
                "subtopic": subtopic_title,
                "request": request.name,
                "kind": request.kind,
                "question_count": request.question_count,
            }

            status = cache.get_status(key)
            if status in {"failed", "skipped"}:
                entry = cache.get_entry(key) or {}
                reason = str(entry.get("reason", ""))
                if status == "failed" or reason.startswith("error:"):
                    logger.info(
                        "Clearing failed cache entry for %s - %s (%s) to retry",
                        course.get("code"),
                        topic_title,
                        request.name,
                    )
                    cache.clear(key)
                    status = None

            if config.resume and cache.has_completed(key):
                cached = cache.load(key)
                if cached:
                    restored = [Question.model_validate(item) for item in cached]
                    questions.extend(restored)
                    logger.info(
                        "Loaded %d cached questions for %s - %s (%s)",
                        len(restored),
                        course.get("code"),
                        topic_title,
                        request.name,
                    )
                continue

            context_text, rag_sources = self._format_context(
                rag_contexts, limit=config.rag_context_limit, offset=idx * config.rag_context_limit
            )
            if not context_text or len(rag_sources) < 2:  # Require at least some meaningful context
                logger.warning("Insufficient RAG context (%d sources) for %s; skipping", len(rag_sources), request.name)
                cache.mark_skipped(key, reason="rag_insufficient", meta=meta)
                continue

            attempt = 0
            generated: List[Question] = []
            last_error: Optional[Exception] = None
            while attempt < max(1, config.request_attempts):
                try:
                    generated = self._call_gemini(
                        config=config,
                        course=course,
                        topic_title=topic_title,
                        subtopic_title=subtopic_title,
                        request=request,
                        context_text=context_text,
                        rag_sources=rag_sources,
                    )
                    break
                except Exception as exc:  # pragma: no cover - network dependent
                    last_error = exc
                    attempt += 1
                    if attempt >= config.request_attempts:
                        break
                    sleep_for = 1.5 * attempt
                    logger.warning(
                        "Retrying %s after error (%s); sleep %.1fs",
                        request.name,
                        exc,
                        sleep_for,
                    )
                    time.sleep(sleep_for)

            if not generated:
                reason = "request_failed"
                if last_error:
                    logger.error(
                        "Failed to generate questions for %s - %s (%s): %s",
                        course.get("code"),
                        topic_title,
                        request.name,
                        last_error,
                    )
                    # Handle API key errors more gracefully
                    error_str = str(last_error)
                    if "API_KEY" in error_str or "REDACTED_API_KEY" in error_str or "authentication" in error_str.lower():
                        reason = "api_key_error"
                        logger.error("API key authentication failed. Please check your Gemini API key configuration.")
                    else:
                        reason = f"error:{last_error}"
                cache.mark_failed(key, reason=reason, meta=meta)
                continue

            cache.store(
                key,
                [q.model_dump() for q in generated],
                meta={**meta, "rag_sources": rag_sources},
            )
            questions.extend(generated)
            self._persist_to_firestore(generated, enable=config.store_firestore)

            # Update progress after each batch completion
            self._update_progress_after_batch(
                config=config,
                course=course,
                topic_title=topic_title,
                subtopic_title=subtopic_title,
                request=request,
                completed_count=len(generated)
            )

            self._sleep_with_jitter(config.request_delay_s, config.delay_jitter)

        return questions

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _cache_for(self, cache_dir: Path) -> QuestionCache:
        resolved = cache_dir.expanduser().resolve()
        cache = self._cache_map.get(resolved)
        if cache is None:
            cache = QuestionCache(resolved)
            self._cache_map[resolved] = cache
        return cache

    def _load_course(self, courses_path: Path, course_code: str) -> Dict[str, Any]:
        resolved = courses_path.expanduser().resolve()
        if resolved not in self._course_store:
            data = json.loads(resolved.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError("courses.json must be a list of course objects")
            self._course_store[resolved] = data
        for row in self._course_store[resolved]:
            code = str(row.get("code") or "").strip().lower()
            if code == course_code.strip().lower():
                return row
        raise ValueError(f"Course code '{course_code}' not found in {courses_path}")

    def _retrieve_rag_context(
        self,
        *,
        course: Dict[str, Any],
        topic_title: str,
        subtopic_title: str,
        config: QuestionBatchConfig,
    ) -> List[Dict[str, Any]]:
        queries = [
            f"{course.get('code', '')} {course.get('title', '')} {topic_title} {subtopic_title}",
            f"{course.get('code', '')} {topic_title} {subtopic_title}",
            f"{course.get('title', '')} {subtopic_title}",
        ]

        where_candidates: List[Any] = []
        if config.rag_where:
            where_candidates.append(config.rag_where)
        course_code = str(course.get("code") or "").strip()
        if course_code:
            where_candidates.append({"COURSE_FOLDER": course_code})
            where_candidates.append({"COURSE_CODE": course_code.split()[0] if " " in course_code else course_code})
        where_candidates.append(None)

        for attempt in range(max(1, config.rag_attempts)):
            for where in where_candidates:
                metadata = MetaData.from_partial(where) if isinstance(where, dict) else where
                for query in queries:
                    try:
                        results = self.rag.search_with_temperature(
                            query,
                            topk=config.rag_topk,
                            final_k=config.rag_final_k,
                            tau=config.rag_tau,
                            min_sim=config.rag_min_similarity,
                            where=metadata,
                        )
                    except Exception as exc:  # pragma: no cover - network dependent
                        logger.warning("Chroma search failed for '%s': %s", query, exc)
                        continue
                    if results:
                        return results

                # If no results found with metadata filter, try without filter
                if where is not None:
                    for query in queries:
                        try:
                            results = self.rag.search_with_temperature(
                                query,
                                topk=config.rag_topk,
                                final_k=config.rag_final_k,
                                tau=config.rag_tau,
                                min_sim=0.3,  # Lower minimum similarity for fallback
                                where=None,  # No metadata filter
                            )
                        except Exception as exc:  # pragma: no cover - network dependent
                            logger.warning("Chroma fallback search failed for '%s': %s", query, exc)
                            continue
                        if results:
                            logger.info("Using fallback search results for '%s'", query)
                            return results

            time.sleep(0.5 * (attempt + 1))
        return []

    def _format_context(
        self,
        contexts: List[Dict[str, Any]],
        *,
        limit: int,
        offset: int,
    ) -> tuple[str, List[Dict[str, Any]]]:
        if not contexts:
            return "", []
        ordered = list(contexts)
        if limit <= 0 or limit > len(ordered):
            subset = ordered
        else:
            start = offset % len(ordered) if ordered else 0
            rotated = ordered[start:] + ordered[:start]
            subset = rotated[:limit]

        lines: List[str] = []
        sources: List[Dict[str, Any]] = []
        for idx, item in enumerate(subset, start=1):
            meta = dict(item.get("meta") or {})
            path = meta.get("path") or meta.get("FILENAME") or meta.get("COURSE_FOLDER") or ""
            snippet = str(item.get("snippet") or meta.get("snippet") or "").strip()
            score = float(item.get("score") or 0.0)
            ref_id = f"ref-{idx}"
            lines.append(
                f"[{ref_id}] path={path} score={score:.2f} chunk={meta.get('chunk_index')}\n{snippet}"
            )
            sources.append(
                {
                    "ref_id": ref_id,
                    "path": path,
                    "chunk_index": meta.get("chunk_index"),
                    "score": score,
                    "snippet": snippet,
                }
            )
        return "\n\n".join(lines), sources

    def _call_gemini(
        self,
        *,
        config: QuestionBatchConfig,
        course: Dict[str, Any],
        topic_title: str,
        subtopic_title: str,
        request: RequestPlan,
        context_text: str,
        rag_sources: List[Dict[str, Any]],
    ) -> List[Question]:
        prompt = self._build_prompt(
            course=course,
            topic_title=topic_title,
            subtopic_title=subtopic_title,
            request=request,
            context_text=context_text,
        )
        gen_config = GeminiConfig(
            temperature=config.gemini_temperature,
            top_p=config.gemini_top_p,
            max_output_tokens=config.gemini_max_output_tokens,
        )

        if self.use_structured:
            # Request structured output directly from Gemini when possible
            gen_config.response_schema = GeminiQuestionBatch

            # Ask Gemini for structured output when supported; otherwise fall back to manual parsing
            try:
                response = self.gemini.generate(
                    prompt,
                    model=config.gemini_model,
                    generation_config=gen_config,
                    response_model=GeminiQuestionBatch,
                )
            except Exception as exc:
                logger.warning(
                    "Gemini structured output failed (%s). Falling back to raw JSON parsing",
                    exc,
                )
                gen_config.response_schema = None
                response = self.gemini.generate(
                    prompt,
                    model=config.gemini_model,
                    generation_config=gen_config,
                )
        else:
            # Bypass structured output entirely when disabled
            gen_config.response_schema = None
            response = self.gemini.generate(
                prompt,
                model=config.gemini_model,
                generation_config=gen_config,
            )

        # Debug: print the raw response before validation (only in verbose mode)
        if os.environ.get("COURSEGEN_DEBUG", "").lower() == "true":
            print(f"DEBUG: Raw response type: {type(response)}")
            if isinstance(response, dict):
                print(f"DEBUG: Response keys: {response.keys()}")
                if 'result' in response:
                    print(f"DEBUG: Result content: {response['result'][:200]}...")
                if 'questions' in response:
                    print(f"DEBUG: First question keys: {response['questions'][0].keys() if response['questions'] else 'No questions'}")
                    if response['questions'] and 'solution_steps' in response['questions'][0]:
                        print(f"DEBUG: First question solution_steps: {repr(response['questions'][0]['solution_steps'])} (type: {type(response['questions'][0]['solution_steps'])})")

        # Convert to GeminiQuestionBatch
        if isinstance(response, GeminiQuestionBatch):
            batch = response
        else:
            # Handle case where response has 'result' key with raw JSON
            if isinstance(response, dict) and 'result' in response:
                raw_result = response['result']
                try:
                    batch = self._parse_batch_from_raw(raw_result)
                except (ValueError, KeyError) as exc:
                    self._dump_failed_payload(raw_result)
                    raise ValueError(
                        f"Could not parse JSON from response: {raw_result[:200]}..."
                    ) from exc
            else:
                batch = GeminiQuestionBatch.model_validate(response)

        actual_count = len(batch.questions)
        expected_count = request.question_count
        if actual_count != expected_count:
            raise QuestionGenerationError(
                f"Expected {expected_count} questions for {request.name} but received {actual_count}"
            )

        return self._convert_to_questions(
            batch.questions,
            course=course,
            topic_title=topic_title,
            subtopic_title=subtopic_title,
            request=request,
            rag_sources=rag_sources,
            wrap_latex=config.latex_wrap_steps,
        )

    @staticmethod
    def _extract_json_payload(text: str) -> Optional[str]:
        if not text:
            return None

        code_block = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text, re.DOTALL)
        if code_block:
            candidate = code_block.group(1).strip()
            if candidate:
                return candidate

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1].strip()
            if candidate:
                return candidate
        return None

    def _parse_batch_from_raw(self, raw_result: str) -> GeminiQuestionBatch:
        parsed_response = self._extract_json_content(raw_result)
        normalized = self._normalize_question_payload(parsed_response)
        return GeminiQuestionBatch.model_validate(normalized)

    def _dump_failed_payload(self, raw_result: str) -> None:
        try:
            dump_root = Path(os.environ.get("COURSEGEN_DEBUG_DUMP_DIR", str(DEFAULT_CACHE_ROOT / "failed_responses")))
            dump_root.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            random_suffix = f"{random.randint(0, 9999):04d}"
            path = dump_root / f"failed_payload_{timestamp}_{random_suffix}.json"
            path.write_text(raw_result, encoding="utf-8")
            logger.debug("Wrote failing payload to %s", path)
        except Exception as exc:
            logger.debug("Failed to write debug payload: %s", exc)

    @staticmethod
    def _strip_trailing_commas(text: str) -> str:
        if not text:
            return text

        result: list[str] = []
        in_string = False
        escaped = False
        length = len(text)
        i = 0
        while i < length:
            ch = text[i]
            if in_string:
                result.append(ch)
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue
            if ch == '"':
                in_string = True
                result.append(ch)
                i += 1
                continue
            if ch == ',':
                j = i + 1
                while j < length and text[j] in " \t\r\n":
                    j += 1
                if j < length and text[j] in '}]':
                    i += 1
                    continue
            result.append(ch)
            i += 1
        return "".join(result)

    @staticmethod
    def _repair_truncated_json(text: str) -> Optional[str]:
        if not text:
            return None

        start = None
        for idx, ch in enumerate(text):
            if ch in "[{":
                start = idx
                break
        if start is None:
            return None

        in_string = False
        escaped = False
        stack: list[str] = []
        for ch in text[start:]:
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch in "[{":
                stack.append(ch)
            elif ch in "]}":
                if stack:
                    opener = stack[-1]
                    if (opener == "[" and ch == "]") or (opener == "{" and ch == "}"):
                        stack.pop()

        repaired = text[start:]
        if in_string:
            repaired += '"'
        for opener in reversed(stack):
            repaired += ']' if opener == '[' else '}'
        return repaired

    @staticmethod
    def _simple_json_load(content: str) -> Any:
        if not content:
            raise ValueError("Empty JSON payload")
        if not isinstance(content, str):
            content = str(content)

        # Enhanced LaTeX escaping for JSON compatibility
        def _escape_latex_for_json(text: str) -> str:
            # Handle LaTeX math expressions: \(...\), \[...\], \(...\)
            text = re.sub(r'\\\(', '\\\\(', text)  # \( -> \(
            text = re.sub(r'\\\)', '\\\\)', text)  # \) -> \)
            text = re.sub(r'\\\[', '\\\\[', text)  # \[ -> \[
            text = re.sub(r'\\\]', '\\\\]', text)  # \] -> \]

            # Handle common LaTeX commands and symbols
            text = re.sub(r'\\(?![\\/bfnrt\"u\\\\])', r'\\\\', text)

            return text

        def _attempt_load(candidate: str) -> Optional[Any]:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None

        def _add_candidate(candidates: list[str], value: Optional[str]) -> None:
            if not value:
                return
            if value not in candidates:
                candidates.append(value)

        candidates: list[str] = []
        _add_candidate(candidates, content)
        _add_candidate(candidates, QuestionGenerator._strip_trailing_commas(content))

        sanitized = _escape_latex_for_json(content)
        _add_candidate(candidates, sanitized)
        _add_candidate(candidates, QuestionGenerator._strip_trailing_commas(sanitized))

        doubled = sanitized.replace("\\", "\\\\")
        _add_candidate(candidates, doubled)
        _add_candidate(candidates, QuestionGenerator._strip_trailing_commas(doubled))

        for candidate in candidates:
            result = _attempt_load(candidate)
            if result is not None:
                return result

        for candidate in candidates:
            repaired = QuestionGenerator._repair_truncated_json(candidate)
            if not repaired or repaired == candidate:
                continue
            result = _attempt_load(repaired)
            if result is not None:
                return result

        raise ValueError("Failed to parse JSON content")

    def _extract_json_content(self, text: str) -> Any:
        if not text:
            raise ValueError("No JSON content in empty response")

        # Prefer fenced code blocks
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            snippet = match.group(1).strip()
            if snippet:
                return self._simple_json_load(snippet)

        # Try raw payload
        trimmed = text.strip()
        try:
            return json.loads(trimmed)
        except json.JSONDecodeError:
            pass

        # Fallback to the first JSON-looking object
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            snippet = brace_match.group(0).strip()
            return self._simple_json_load(snippet)

        raise ValueError(f"No JSON found in response: {text[:200]}...")

    @staticmethod
    def _normalize_solution_steps(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(step).strip() for step in value if str(step).strip()]
        if isinstance(value, tuple):
            return [str(step).strip() for step in value if str(step).strip()]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return [str(step).strip() for step in parsed if str(step).strip()]
                except json.JSONDecodeError:
                    pass
            return [text]
        # Fallback: wrap anything else in a list
        return [str(value).strip()]

    def _normalize_question_payload(self, payload: Any) -> Dict[str, Any]:
        if isinstance(payload, list):
            normalized = []
            for item in payload:
                if isinstance(item, dict):
                    item["solution_steps"] = self._normalize_solution_steps(
                        item.get("solution_steps")
                    )
                    normalized.append(item)
            return {"questions": normalized}

        if not isinstance(payload, dict):
            return {"questions": []}

        questions = payload.get("questions")
        if isinstance(questions, list):
            for question in questions:
                if not isinstance(question, dict):
                    continue
                question["solution_steps"] = self._normalize_solution_steps(
                    question.get("solution_steps")
                )
            return payload

        # Handle single-question payloads
        if isinstance(payload, dict) and "question" in payload:
            normalized_question = dict(payload)
            normalized_question["solution_steps"] = self._normalize_solution_steps(
                normalized_question.get("solution_steps")
            )
            return {"questions": [normalized_question]}

        return payload

    def _convert_to_questions(
        self,
        llm_questions: Iterable[GeminiGeneratedQuestion],
        *,
        course: Dict[str, Any],
        topic_title: str,
        subtopic_title: str,
        request: RequestPlan,
        rag_sources: List[Dict[str, Any]],
        wrap_latex: bool,
    ) -> List[Question]:
        if not rag_sources:
            raise QuestionGenerationError("RAG sources are required for question generation")

        questions: List[Question] = []
        level = self._first(course.get("levels"))
        semester = self._first(course.get("semesters"))
        course_code = str(course.get("code") or "")
        course_title = str(course.get("title") or "")

        for idx, item in enumerate(llm_questions, start=1):
            options = [str(opt).strip() for opt in item.options]
            validate_options(options)

            if any(not option for option in options):
                raise QuestionGenerationError("Options must not be empty")

            normalized_options = {option.lower() for option in options}
            if len(normalized_options) != len(options):
                raise QuestionGenerationError("Options must be unique")

            answer_letter, answer_text = self._normalize_answer(
                item.correct_answer,
                item.correct_answer_text,
                options,
            )
            validate_answer_in_options(answer_text, options)

            question_text = str(item.question or "").strip()
            if not question_text:
                raise QuestionGenerationError("Question text is empty")

            explanation = str(item.explanation or "").strip()
            if not explanation:
                raise QuestionGenerationError("Explanation is required")
            steps = [str(step).strip() for step in (item.solution_steps or []) if str(step).strip()]
            if request.kind == "calculation":
                steps = self._ensure_latex_steps(steps, wrap_latex=wrap_latex)
                # Allow empty solution steps for calculation questions instead of raising error
                if not steps:
                    steps = []
            else:
                # For theory questions, ensure solution_steps is an empty list, not an empty string
                steps = []

            question = Question(
                course_code=course_code,
                course_name=course_title,
                topic_name=topic_title,
                subtopic_name=subtopic_title,
                level=level,
                semester=semester,
                question_type=request.kind,
                difficulty_ranking=request.difficulty_rank,
                difficulty=self._difficulty_from_rank(request.difficulty_rank),
                question=question_text,
                options=options,
                correct_answer=answer_letter,
                correct_answer_text=answer_text,
                explanation=explanation,
                solution_steps=steps,
                rag_sources=[dict(src) for src in rag_sources],
                extra_metadata={
                    "request_name": request.name,
                    "question_index": idx,
                    "generated_at": time.time(),
                },
            )
            questions.append(question)
        return questions

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    def _difficulty_from_rank(self, rank: int) -> str:
        if rank <= 3:
            return "Easy"
        if rank <= 6:
            return "Medium"
        return "Hard"

    def _first(self, value: Any) -> Optional[str]:
        if isinstance(value, list) and value:
            return str(value[0])
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    def _normalize_answer(
        self,
        answer_value: Any,
        answer_text_value: Optional[str],
        options: List[str],
    ) -> tuple[str, str]:
        letters = ["A", "B", "C", "D"]
        if answer_text_value:
            text = answer_text_value.strip()
            for idx, option in enumerate(options):
                if text.lower() == option.lower():
                    return letters[idx], option
        if isinstance(answer_value, int):
            idx = answer_value - 1
            if 0 <= idx < len(options):
                return letters[idx], options[idx]
        if isinstance(answer_value, str):
            cleaned = answer_value.strip().upper()
            for idx, letter in enumerate(letters):
                if cleaned in {letter, f"OPTION {letter}", f"{letter}.", f"{letter})"}:
                    return letter, options[idx]
            for idx, option in enumerate(options):
                if cleaned.lower() == option.lower():
                    return letters[idx], option
        raise QuestionGenerationError("Unable to determine correct answer letter")

    def _ensure_latex_steps(self, steps: List[str], *, wrap_latex: bool) -> List[str]:
        if not steps:
            return []
        formatted: List[str] = []
        for step in steps[:8]:
            clean = step.strip()
            if not clean:
                continue
            if not wrap_latex:
                formatted.append(clean)
                continue
            if clean.startswith("$") or clean.startswith("\\("):
                formatted.append(clean)
            else:
                formatted.append(f"\\({clean}\\)")
        return formatted

    @staticmethod
    def _format_latex_for_display(latex_text: str) -> str:
        """Format LaTeX text for proper display, handling escaped backslashes."""
        if not latex_text:
            return latex_text

        # Convert escaped backslashes back to single backslashes for display
        # This handles cases where JSON had \\( -> \( for display
        text = latex_text.replace("\\\\", "\\")

        # Ensure proper LaTeX delimiters for display
        if "\\(" in text and not text.startswith("\\("):
            # If contains inline math but not wrapped, wrap it
            text = f"\\({text}\\)"

        return text



        

    def _build_prompt(
        self,
        *,
        course: Dict[str, Any],
        topic_title: str,
        subtopic_title: str,
        request: RequestPlan,
        context_text: str,
    ) -> str:
        level = self._first(course.get("levels")) or "Unknown"
        semester = self._first(course.get("semesters")) or "Unknown"

        base_guidance = textwrap.dedent("""
        - Questions must be original, unambiguous, and self-contained.
        - Provide exactly four distinct options labelled A, B, C, D.
        - 'correct_answer' must be one of "A", "B", "C", or "D" and match 'correct_answer_text'.
        - Explanations should help students understand why the answer is correct and reference key formulas when relevant.
        - Wrap every formula or symbol in `$...$` with double-escaped commands (e.g., `$\\omega = 2\\pi f$`).
        """).strip()


        latex_guidance = textwrap.dedent("""
        - Use LaTeX for every mathematical expression.
        - Wrap inline math with `$...$` and multi-line math with `$$...$$` so the renderer treats it correctly.
        - Because the output is JSON, ESCAPE every backslash twice (e.g., write `\\frac{a}{b}` to render `$\frac{a}{b}$`).
        - Example inline: `$\\frac{12}{4} = 3 \\text{Ohms}$`.
        - Example integral: `$\\int_{0}^{1} x^2 \\, dx$`.
        - Ensure explanations and solution steps follow the same `$`-delimited, double-escaped format.
        """).strip()


        if request.kind == "calculation":
            steps_guidance = textwrap.dedent("""
            - Treat every question as calculation-focused with appropriate numeric work.
            - 'solution_steps' must be a JSON array containing 3–7 concise steps.
            - Steps should include formula selection, substitution, computation, and conclusion.
            """).strip()
        else:
            steps_guidance = textwrap.dedent("""
            - Emphasize conceptual understanding and qualitative reasoning.
            - 'solution_steps' must be an empty JSON array []. Do not place text, null, or placeholders there.
            """).strip()

        prompt = textwrap.dedent(f"""
Generate {request.question_count} unique, curriculum-aligned multiple-choice questions (MCQs) for the course "{course.get('title','')}" ({course.get('code','')}) on the topic "{topic_title}" (subtopic: "{subtopic_title}"). Use ONLY the provided extracts for grounding; do not quote them verbatim. Keep each question self-contained and original.

### REQUIRED OUTPUT (VALID JSON ONLY)
- Return a single JSON object and nothing else. No markdown, no comments, no surrounding prose.
- Top-level format:
  {{
    "questions": [
      {{
        "question": "<string>",
        "options": ["<string>", "<string>", "<string>", "<string>"],
        "correct_answer": "A" | "B" | "C" | "D",
        "correct_answer_text": "<string>",
        "explanation": "<string>",
        "solution_steps": ["<step1>", "<step2>", "..."]
      }}
    ]
  }}
- The JSON MUST parse as-is. Do not include extra keys or null placeholders.

### OPTIONS / ANSWER RULES
- Provide exactly **four** option *strings* in `options`. **Do not** prefix option strings with "A)", "B)", etc — options should be raw option text.
- `correct_answer` must be one of "A"|"B"|"C"|"D".
- `correct_answer_text` must be exactly equal to the corresponding element of `options`.
- All option texts must be distinct and plausible. Avoid distractors that are obviously wrong (e.g., unit mismatch, off by factor of 1000).
- For numeric options, include units in the option text (e.g., "29.8 MPa").

### CONSISTENCY & NO-BACKTRACKING RULES
- Never alter the problem data to fit an answer. If a mismatch is detected, DISCARD that question and generate a new one that is internally consistent.
- Do not “work backwards from options.” Compute the correct result first, then compose options (1 correct + 3 plausible distractors).
- Absolutely forbid meta-reasoning or self-correction in `solution_steps` (e.g., “recalculating”, “let’s assume”, “there seems to be a discrepancy”, “work backwards”, “typo”).
- `solution_steps` style for calculation items:
  - Exactly 3–5 short lines, each starting with one of: “Formula:”, “Convert:”, “Substitute:”, “Compute:”, “Final:”.
  - No extra sentences or commentary; each line ≤ 25 words.
  - The **Final** line must repeat the numeric answer with units and rounding.
- For concept items, `solution_steps` must be `[]` (empty).
- Options policy:
  - Generate options AFTER computing the answer.
  - Ensure `correct_answer_text` equals the selected element in `options` (A=0, B=1, C=2, D=3).
  - For numeric questions, every option includes units; distractors reflect realistic slips (rounding, factor-of-10, omitted factor of 2), not nonsense.
  - Keep steps **minimal**: no repeating the same calculation in different units.
  - If SI units are already consistent, do not add conversions. Only convert when units are mismatched.
  - Do not restate or re-check the same formula more than once.


### SOLUTION STEPS RULES
- If `request.kind == "calculation"`:
  - `solution_steps` must be an array of **3–7** concise steps (strings).
  - Steps should include: formula selection, unit conversions, substitution (show numbers using double-escaped LaTeX where relevant), a short arithmetic line, and one concluding line that states the final numeric value with units and rounding.
  - Round intermediate and final numeric results sensibly (2–4 significant figures) and state the rounding rule used.
- If `request.kind != "calculation"`:
  - `solution_steps` must be an empty JSON array [].
- Never put long prose in `solution_steps` — keep them terse, ordered, and actionable.

### MATHEMATICAL / LATEX FORMATTING
- Use LaTeX for all math. Wrap inline math with `$...$` and display math with `$$...$$`.
- Because the output is JSON, **escape every backslash twice** so a LaTeX fraction looks like `"\\frac{{a}}{{b}}"` in the JSON string (which renders as `$\\frac{{a}}{{b}}$` when unescaped).
- Example inline in a JSON string: `"$\\frac{{12}}{{4}} = 3\\ \\text{{Ohms}}$"`.
- Ensure every LaTeX expression appears inside `$...$` or `$$...$$` and that backslashes are doubled.
- Use exactly two backslashes for every LaTeX command in JSON (e.g., "\\\\frac", "\\\\int", "\\\\text").
- Never triple-escape backslashes (avoid \\\\\\int).
- Do not expand algebra more than once; keep expressions in their simplest readable LaTeX form.
- Write units as "\\\\text{{...}}" immediately after the number, with a single space if needed (e.g., "$333.3\\\\,\\\\text{{kN}}$").


### CONTENT & PEDAGOGICAL GUIDELINES
- Align difficulty with Level **{level}** and Semester **{semester}**.
- Questions must be: original, unambiguous, self-contained, and solvable using the provided context + common engineering formulas.
- For calculation questions, always include units and show unit conversions in `solution_steps`.
- Distractors:
  - For calculations: include 3 plausible distractors (e.g., common algebraic slips, rounding variants, unit conversion mistakes).
  - For concept questions: include 3 plausible conceptual distractors that test common misunderstandings.
- Avoid excessive edge cases, trick wording, or ambiguous qualifiers (e.g., "usually", "often", "may").
- If the grounding context lacks enough data to compute a value, either:
  - create a self-contained numeric assumption and state it in the question (and in the steps), **or**
  - do NOT generate the question — prefer safe, answerable items.

### QUALITY & SANITY CHECKS (do these before returning JSON)
1. Confirm `options` contains exactly 4 items and `correct_answer_text` matches one of them exactly.
2. Confirm no option duplicates.
3. For numeric answers, re-calculate the result and ensure the value in `correct_answer_text` matches the `solution_steps` final line.
4. Confirm all LaTeX backslashes are double-escaped.
5. Validate that the full output is legal JSON (single parseable object).


### Numerical Consistency
- The numeric value in the "Final" solution step **must exactly match** the 'correct_answer_text'.
- Never produce mismatched results (e.g., steps yielding 5625 but correct_answer_text = 1125).
- If rounding is required, round consistently across steps, final answer, and correct_answer_text.
- Do not exaggerate or miscompute values; ensure units are realistic (e.g., MPa range for stresses, not GPa unless physically correct).


### TERMINATION
- After the final numeric result is given, stop generating steps.
- Do not continue with alternative derivations, assumptions, or repeated formulas.


### REFERENCE CONTEXT (grounding only)
Use these extracts strictly for background/context. Do not copy text verbatim; rephrase and use them to ensure curriculum alignment:
{context_text}

### FINAL INSTRUCTIONS
- Return exactly {request.question_count} questions that satisfy every rule above.
- Keep `explanation` concise — one paragraph (1–3 sentences) that teaches why the correct answer is right and calls out the key formula(s) in `$...$` form.
- If any constraint cannot be satisfied for a candidate question (e.g., missing numbers in context), skip that question and generate another that is fully answerable.
""").strip()


        return prompt

    def _persist_to_firestore(self, questions: Iterable[Question], *, enable: bool) -> None:
        if not enable or not questions:
            return
        store = self._resolve_firestore()
        if store is None:
            logger.debug("Firestore not configured; skipping persistence")
            return
        for question in questions:
            try:
                store.set_question(question)
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("Failed to persist question for %s: %s", question.course_code, exc)

    def _resolve_firestore(self) -> Optional[Any]:
        if self._firestore is not None:
            return self._firestore
        if FireStore is None:
            return None
        try:
            self._firestore = FireStore()
            return self._firestore
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Could not initialize Firestore: %s", exc)
            self._firestore = None
            return None

    def _update_progress_after_batch(
        self,
        *,
        config: QuestionBatchConfig,
        course: Dict[str, Any],
        topic_title: str,
        subtopic_title: str,
        request: RequestPlan,
        completed_count: int,
    ) -> None:
        """Update progress tracking after a batch is completed."""
        cache = self._cache_for(config.cache_dir)
        course_code = str(course.get("code", ""))

        # Update cache.json with batch completion
        cache.mark_batch_completed(course_code, topic_title, subtopic_title, request.name)

        # Update Firestore GenerationProgress collection
        if config.store_firestore:
            try:
                store = self._resolve_firestore()
                if store:
                    # Calculate total questions for this subtopic (20: 10 theory + 10 calculation)
                    total_questions = 20
                    completed_questions = completed_count

                    # Get current progress to accumulate
                    try:
                        existing_progress = store.db.collection("GenerationProgress").document(f"{course_code}-{topic_title}-{subtopic_title}").get()
                        if existing_progress.exists:
                            data = existing_progress.to_dict()
                            completed_questions += data.get("completed_questions", 0)
                    except Exception:
                        pass  # Continue with current batch count if unable to fetch existing

                    status = "completed" if completed_questions >= total_questions else "in_progress"

                    store.update_generation_progress(
                        course_code=course_code,
                        course_title=course.get("title", ""),
                        department=course.get("department", "Unknown"),
                        status=status,
                        total_topics=1,  # This subtopic
                        completed_topics=1 if status == "completed" else 0,
                        total_questions=total_questions,
                        completed_questions=completed_questions,
                    )
            except Exception as exc:
                logger.warning("Failed to update Firestore progress: %s", exc)

    def _sleep_with_jitter(self, base: float, jitter: float) -> None:
        if base <= 0:
            return
        span = abs(jitter)
        low = max(0.0, base * (1 - span))
        high = base * (1 + span)
        time.sleep(random.uniform(low, high))


class QuestionBatchRunner:
    """Coordinator for orchestrating multiple batch requests."""

    def __init__(self, generator: QuestionGenerator) -> None:
        self.generator = generator

    def run(self, config: QuestionBatchConfig) -> List[Question]:
        return self.generator.generate_course_questions(config)


def _parse_optional_json(value: Optional[str]) -> Optional[Dict[str, Any]]:
    if not value:
        return None
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("RAG where filter must be a JSON object")
        return parsed
    except json.JSONDecodeError as exc:  # pragma: no cover - cli validation
        raise argparse.ArgumentTypeError(f"Invalid JSON: {exc}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate questions with Gemini + RAG")
    parser.add_argument("--course-code", default="all", help="Course code e.g. EEE 301 (default: all courses)")
    parser.add_argument(
        "--courses-json",
        default=str(DEFAULT_COURSES_JSON),
        help="Path to courses.json containing outlines",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_ROOT),
        help="Directory for generation cache",
    )
    parser.add_argument("--rag-topk", type=int, default=30, help="Candidate retrieval pool size")
    parser.add_argument("--rag-final-k", type=int, default=12, help="Context chunks passed to LLM")
    parser.add_argument("--rag-tau", type=float, default=0.35, help="Sampling temperature for RAG")
    parser.add_argument(
        "--rag-min-sim",
        type=float,
        default=0.6,
        help="Minimum similarity threshold for context filtering",
    )
    parser.add_argument(
        "--rag-where",
        type=_parse_optional_json,
        default=None,
        help="Additional metadata filter for Chroma search (JSON object)",
    )
    parser.add_argument(
        "--theory-per-request",
        type=int,
        default=10,
        help="Number of theory questions per Gemini request",
    )
    parser.add_argument(
        "--calc-per-request",
        type=int,
        default=5,
        help="Number of calculation questions per Gemini request",
    )
    parser.add_argument("--no-resume", action="store_true", help="Do not reuse cached generations")
    parser.add_argument(
        "--skip-firestore",
        action="store_true",
        help="Disable persistence to Firestore",
    )
    parser.add_argument(
        "--topics",
        nargs="*",
        default=None,
        help="Optional list of topics to include (case insensitive)",
    )
    parser.add_argument(
        "--subtopics",
        nargs="*",
        default=None,
        help="Optional list of subtopics to include (case insensitive)",
    )
    parser.add_argument("--output-jsonl", help="Path to save generated questions as JSONL")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Gemini model name (e.g. gemini-2.5-flash)",
    )
    parser.add_argument("--temperature", type=float, default=0.25, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=0.85, help="Top-p nucleus sampling value")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=6000,
        help="Maximum tokens Gemini can return per request",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.5,
        help="Base delay between Gemini calls (seconds)",
    )
    parser.add_argument(
        "--delay-jitter",
        type=float,
        default=0.25,
        help="Random jitter fraction applied to delays",
    )
    parser.add_argument(
        "--rag-attempts",
        type=int,
        default=2,
        help="Number of attempts to retrieve RAG context before skipping",
    )
    parser.add_argument(
        "--request-attempts",
        type=int,
        default=2,
        help="Number of retries for Gemini generation",
    )
    parser.add_argument(
        "--no-latex-wrap",
        action="store_true",
        help="Do not automatically wrap calculation steps with LaTeX delimiters",
    )
    structured_group = parser.add_mutually_exclusive_group()
    structured_group.add_argument(
        "--structured-output",
        dest="structured_output",
        action="store_true",
        help="Enable Gemini structured output schema",
    )
    structured_group.add_argument(
        "--no-structured-output",
        dest="structured_output",
        action="store_false",
        help="Disable Gemini structured output schema (default)",
    )
    parser.set_defaults(structured_output=None)
    return parser


def _load_course_standalone(courses_path: Path, course_code: str) -> Dict[str, Any]:
    """Standalone course loader (no class instance needed)."""
    data = json.loads(courses_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("courses.json must be a list of course objects")
    for row in data:
        code = str(row.get("code") or "").strip().lower()
        if code == course_code.strip().lower():
            return row
    raise ValueError(f"Course code '{course_code}' not found in {courses_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    courses_path = Path(args.courses_json)
    course_code = args.course_code or "all"

    if course_code.lower() != "all":
        course = _load_course_standalone(courses_path, course_code)
        courses = [course]
    else:
        data = json.loads(courses_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("courses.json must be a list of course objects")
        courses = [row for row in data if row.get("outline")]  # Only courses with outlines
        if not courses:
            logger.warning("No courses with outlines found in %s", courses_path)
            return 0

    all_questions = []
    common_config = {
        "courses_json_path": courses_path,
        "cache_dir": Path(args.cache_dir),
        "rag_topk": args.rag_topk,
        "rag_final_k": args.rag_final_k,
        "rag_tau": args.rag_tau,
        "rag_min_similarity": args.rag_min_sim,
        "rag_where": args.rag_where,
        "theory_questions_per_request": args.theory_per_request,
        "calc_questions_per_request": args.calc_per_request,
        "resume": not args.no_resume,
        "store_firestore": not args.skip_firestore,
        "request_delay_s": args.request_delay,
        "delay_jitter": args.delay_jitter,
        "gemini_model": args.model,
        "gemini_temperature": args.temperature,
        "gemini_top_p": args.top_p,
        "gemini_max_output_tokens": args.max_output_tokens,
        "request_attempts": args.request_attempts,
        "rag_attempts": args.rag_attempts,
        "latex_wrap_steps": not args.no_latex_wrap,
        "target_topics": args.topics,
        "target_subtopics": args.subtopics,
        "output_path": Path(args.output_jsonl) if args.output_jsonl else None,
    }

    # Initialize Gemini service with explicit API keys to avoid env var fallback
    from services.Gemini.gemini_api_keys import GeminiApiKeys
    from services.Gemini.api_key_manager import ApiKeyManager

    gemini_keys = GeminiApiKeys()
    api_keys = gemini_keys.get_keys()
    api_key_manager = ApiKeyManager(api_keys)

    gemini_service = GeminiService(
        api_key_manager=api_key_manager,
        model=args.model,
        generation_config=GeminiConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            max_output_tokens=args.max_output_tokens,
        ),
    )

    generator = QuestionGenerator(
        gemini_service=gemini_service,
        use_structured=args.structured_output,
    )
    runner = QuestionBatchRunner(generator)

    for course in courses:
        course_code = course.get("code", "unknown")
        config = QuestionBatchConfig(course_code=course_code, **common_config)

        try:
            questions = runner.run(config)
            all_questions.extend(questions)
            logger.info("Generated %d questions for %s", len(questions), course_code)
        except ValidationError as exc:
            logger.error("Validation failed for %s: %s", course_code, exc)
            continue
        except Exception as exc:
            logger.error("Question generation failed for %s: %s", course_code, exc)
            continue

    output_path = common_config["output_path"]
    if output_path:
        write_jsonl(str(output_path), [q.model_dump() for q in all_questions])
        logger.info("Saved %d total questions to %s", len(all_questions), output_path)
    else:
        logger.info("Generated %d total questions across %d courses", len(all_questions), len(courses))

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
