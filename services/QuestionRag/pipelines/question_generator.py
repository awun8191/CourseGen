"""Gemini powered question generation pipeline using RAG + Firestore persistence."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from data_models.gemini_config import GeminiConfig
from data_models.question_model import Question
from services.Gemini.gemini_service import GeminiService

from ..utils import ChromaQuery, CourseProgressCache, MetaData, QuestionCache
from ..utils.batch_utils import write_jsonl
from .config import QuestionBatchConfig, RequestPlan
from .json_utils import QuestionGenerationError, parse_batch_from_raw, dump_failed_payload
from .models import GeminiQuestionBatch
from .prompt_utils import build_question_generation_prompt
from .validation_utils import convert_to_questions
from .question_gen_config import get_question_gen_config

# Import centralized configuration
try:
    from config import load_config
    config = load_config()
except ImportError:
    # Fallback to local config if centralized config not available
    import os
    from pathlib import Path

    class FallbackConfig:
        def __init__(self):
            # Use centralized config for consistency
            central_config = get_question_gen_config()
            self.gemini_default_model = central_config.gemini_model
            self.gemini_temperature = central_config.gemini_temperature
            self.gemini_top_p = central_config.gemini_top_p
            self.gemini_max_output_tokens = central_config.gemini_max_output_tokens
            self.gemini_use_thinking = central_config.use_thinking
            self.gemini_thinking_budget = central_config.thinking_budget
            self.courses_json_path_resolved = Path(os.environ.get("COURSEGEN_COURSES_JSON", "data/textbooks/courses.json")).expanduser()
            self.cache_dir_resolved = Path(os.environ.get("COURSEGEN_CACHE_DIR", "OUTPUT_DATA2/cache")).expanduser()
            self.coursegen_qg_loglevel = os.environ.get("COURSEGEN_QG_LOGLEVEL", "INFO").upper()
            self.coursegen_debug = os.environ.get("COURSEGEN_DEBUG", "").lower() == "true"
            self.coursegen_use_structured = os.environ.get("COURSEGEN_USE_STRUCTURED", "0").lower() in ("1", "true", "yes")

    config = FallbackConfig()

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
logger.setLevel(config.coursegen_qg_loglevel)


class QuestionGenerator:
    """Generate MCQ questions using Gemini with RAG context and caching."""

    def __init__(
        self,
        *,
        gemini_service: Optional[GeminiService] = None,
        rag_client: Optional[ChromaQuery] = None,
        firestore: Optional[Any] = None,
        use_structured: Optional[bool] = None,
        email_service: Optional[Any] = None,
    ) -> None:
        self.gemini = gemini_service
        self.rag = rag_client or ChromaQuery()
        self._firestore = firestore
        self._cache_map: Dict[Path, QuestionCache] = {}
        self._course_store: Dict[Path, List[Dict[str, Any]]] = {}
        if use_structured is None:
            use_structured = config.coursegen_use_structured
        self.use_structured = bool(use_structured)
        self._email_service = email_service

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
        # Check if all API keys are exhausted before starting
        if hasattr(self.gemini, 'api_key_manager'):
            model_name = self.gemini._get_model_name(self.gemini.model)
            if self.gemini.api_key_manager.all_keys_exhausted(model_name):
                logger.error(
                    "🚨 All API keys exhausted before starting course %s. Terminating operations.",
                    course.get("code", "unknown")
                )
                raise RuntimeError(
                    f"🚨 ALL API KEYS EXHAUSTED - TERMINATING OPERATIONS 🚨\n"
                    f"Cannot start processing course {course.get('code', 'unknown')} - all keys exhausted."
                )

        outline = course.get("outline") or []
        progress_cache = CourseProgressCache(
            course_code=str(course.get("code") or "unknown"),
            cache_root=config.cache_dir,
            theory_target=config.theory_questions_per_request,
            calc_target=config.calc_questions_per_request,
        )

        self._notify_course_started(course=course, outline=outline)

        results: List[Question] = []
        normalized_topics = config.normalized_topics()
        normalized_subtopics = config.normalized_subtopics()

        for topic in outline:
            topic_start_time = time.time()
            topic_question_count = 0
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
                    progress=progress_cache,
                )
                results.extend(generated)
                topic_question_count += len(generated)

            self._notify_topic_finished(
                course=course,
                topic_title=topic_title,
                progress=progress_cache,
                topic_question_count=topic_question_count,
                topic_start_time=topic_start_time,
                expected_subtopics=len(topic.get("subtopics") or []),
            )

        self._finalize_course_progress(
            config=config,
            course=course,
            progress=progress_cache,
        )
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
        progress: CourseProgressCache,
    ) -> List[Question]:
        # Check if all API keys are exhausted before processing this subtopic
        if hasattr(self.gemini, 'api_key_manager'):
            model_name = self.gemini._get_model_name(self.gemini.model)
            if self.gemini.api_key_manager.all_keys_exhausted(model_name):
                logger.error(
                    "🚨 All API keys exhausted before processing subtopic %s - %s. Terminating.",
                    topic_title, subtopic_title
                )
                raise RuntimeError(
                    f"🚨 ALL API KEYS EXHAUSTED - TERMINATING OPERATIONS 🚨\n"
                    f"Cannot process subtopic {topic_title} - {subtopic_title} - all keys exhausted."
                )

        cache = self._cache_for(config.cache_dir)
        plan = config.request_plan()
        progress.touch_subtopic(topic_title, subtopic_title)
        if progress.subtopic_state(topic_title, subtopic_title) == "completed" and progress.has_persisted(topic_title, subtopic_title):
            logger.info(
                "Skipping %s - %s (%s); already completed and persisted",
                course.get("code"),
                topic_title,
                subtopic_title,
            )
            return []

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
                progress.mark_request_failed(topic_title, subtopic_title, request.name, "rag_empty")
            progress.mark_subtopic_error(topic_title, subtopic_title, "rag_empty")
            return []

        questions: List[Question] = []
        subtopic_completed = progress.subtopic_state(topic_title, subtopic_title) == "completed"
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
            if status == "in_progress":
                logger.info(
                    "Found interrupted cache entry for %s - %s (%s); marking for retry",
                    course.get("code"),
                    topic_title,
                    request.name,
                )
                cache.mark_failed(key, reason="interrupted", meta=meta)
                status = None
            if status in {"failed", "skipped"}:
                entry = cache.get_entry(key) or {}
                reason = str(entry.get("reason", "")) or status
                logger.info(
                    "Retrying %s cache entry for %s - %s (%s); previous reason: %s",
                    status,
                    course.get("code"),
                    topic_title,
                    request.name,
                    reason,
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
                    progress.mark_request_completed(
                        topic_title,
                        subtopic_title,
                        request.name,
                        len(restored),
                    )
                    subtopic_completed = progress.subtopic_state(topic_title, subtopic_title) == "completed"
                continue

            progress.mark_request_started(topic_title, subtopic_title, request.name)

            context_text, rag_sources = self._format_context(
                rag_contexts, limit=config.rag_context_limit, offset=idx * config.rag_context_limit
            )
            if not context_text or len(rag_sources) < 2:  # Require at least some meaningful context
                logger.warning("Insufficient RAG context (%d sources) for %s; skipping", len(rag_sources), request.name)
                cache.mark_skipped(key, reason="rag_insufficient", meta=meta)
                progress.mark_request_failed(topic_title, subtopic_title, request.name, "rag_insufficient")
                continue

            attempt = 0
            generated: List[Question] = []
            last_error: Optional[Exception] = None
            while attempt < max(1, config.request_attempts):
                try:
                    cache.mark_in_progress(key, meta=meta)
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
                except RuntimeError as exc:  # pragma: no cover - API key exhaustion termination
                    # Handle forced termination when all API keys are exhausted
                    if "ALL API KEYS EXHAUSTED" in str(exc) or "CRITICAL" in str(exc):
                        logger.error(
                            "🚨 CRITICAL: All API keys exhausted - terminating question generation for %s - %s (%s)",
                            course.get("code"),
                            topic_title,
                            request.name,
                        )
                        # Mark this request as failed and exit the entire process
                        cache.mark_failed(key, reason="all_keys_exhausted", meta=meta)
                        progress.mark_request_failed(topic_title, subtopic_title, request.name, "all_keys_exhausted")
                        raise exc  # Re-raise to terminate the entire process
                    else:
                        # Handle other RuntimeErrors as regular errors
                        last_error = exc
                        attempt += 1
                        if attempt >= config.request_attempts:
                            break
                        sleep_for = 1.5 * attempt
                        logger.warning(
                            "Retrying %s after RuntimeError (%s); sleep %.1fs",
                            request.name,
                            exc,
                            sleep_for,
                        )
                        time.sleep(sleep_for)
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
                progress.mark_request_failed(topic_title, subtopic_title, request.name, reason)
                continue

            cache.store(
                key,
                [q.model_dump() for q in generated],
                meta={**meta, "rag_sources": rag_sources},
            )
            is_complete = progress.mark_request_completed(
                topic_title,
                subtopic_title,
                request.name,
                len(generated),
            )
            questions.extend(generated)
            subtopic_completed = progress.subtopic_state(topic_title, subtopic_title) == "completed"

            # Update progress after each batch completion
            self._mark_cache_completion(
                config=config,
                course=course,
                topic_title=topic_title,
                subtopic_title=subtopic_title,
                request=request,
            )

            self._sleep_with_jitter(config.request_delay_s, config.delay_jitter)

        subtopic_completed = progress.subtopic_state(topic_title, subtopic_title) == "completed"
        if subtopic_completed:
            if questions and not progress.has_persisted(topic_title, subtopic_title):
                self._persist_to_firestore(questions, enable=config.store_firestore)
                progress.mark_persisted(topic_title, subtopic_title)
                self._cleanup_subtopic_cache(
                    cache=cache,
                    course_code=str(course.get("code")),
                    topic_title=topic_title,
                    subtopic_title=subtopic_title,
                    plan=plan,
                )
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
        prompt = build_question_generation_prompt(
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
            use_thinking=config.use_thinking,
            thinking_budget=config.thinking_budget,
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
        if config.coursegen_debug:
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
                    batch = parse_batch_from_raw(raw_result)
                except (ValueError, KeyError) as exc:
                    dump_path = dump_failed_payload(raw_result)
                    logger.error(
                        "Could not parse JSON for %s - %s (%s). Saved raw payload to %s",
                        course.get("code"),
                        topic_title,
                        request.name,
                        dump_path or "<memory>",
                    )
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

        return convert_to_questions(
            [q.model_dump() for q in batch.questions],
            course=course,
            topic_title=topic_title,
            subtopic_title=subtopic_title,
            request=request,
            rag_sources=rag_sources,
            wrap_latex=config.latex_wrap_steps,
        )

    def _cleanup_subtopic_cache(
        self,
        *,
        cache: QuestionCache,
        course_code: str,
        topic_title: str,
        subtopic_title: str,
        plan: Sequence[RequestPlan],
    ) -> None:
        for request in plan:
            key = cache.make_key(course_code, topic_title, subtopic_title, request.name)
            cache.clear(key)



        


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

    def _notify_course_started(
        self,
        *,
        course: Dict[str, Any],
        outline: List[Dict[str, Any]],
    ) -> None:
        if not self._email_service:
            return
        try:
            total_topics = len(outline)
            total_subtopics = sum(len(topic.get("subtopics") or []) for topic in outline)
            self._email_service.send_course_started(
                course_code=str(course.get("code", "unknown")),
                course_title=str(course.get("title", "")),
                total_topics=total_topics,
                total_subtopics=total_subtopics,
            )
        except Exception as exc:  # pragma: no cover - notification best effort
            logger.warning(
                "Failed to send course start notification for %s: %s",
                course.get("code", "unknown"),
                exc,
            )

    def _notify_topic_finished(
        self,
        *,
        course: Dict[str, Any],
        topic_title: str,
        progress: CourseProgressCache,
        topic_question_count: int,
        topic_start_time: float,
        expected_subtopics: int,
    ) -> None:
        if not self._email_service:
            return
        try:
            course_code = str(course.get("code", "unknown"))
            course_title = str(course.get("title", ""))
            topic_entry = progress.data.get("topics", {}).get(topic_title, {})
            subtopic_entries = topic_entry.get("subtopics", {})
            total_subtopics = expected_subtopics if expected_subtopics is not None else len(subtopic_entries)
            completed_subtopics = sum(
                1 for entry in subtopic_entries.values() if entry.get("state") == "completed"
            )
            errored_subtopics = sum(
                1 for entry in subtopic_entries.values() if entry.get("state") == "error"
            )
            duration_seconds = time.time() - topic_start_time

            self._email_service.send_topic_finished(
                course_code=course_code,
                course_title=course_title,
                topic_title=topic_title,
                question_count=topic_question_count,
                total_subtopics=total_subtopics,
                completed_subtopics=completed_subtopics,
                errored_subtopics=errored_subtopics,
                duration_seconds=duration_seconds,
            )
        except Exception as exc:  # pragma: no cover - notification best effort
            logger.warning(
                "Failed to send topic completion notification for %s/%s: %s",
                course.get("code", "unknown"),
                topic_title,
                exc,
            )

    def _iter_target_subtopics(
        self,
        course: Dict[str, Any],
        normalized_topics: Optional[set[str]],
        normalized_subtopics: Optional[set[str]],
    ) -> Iterable[tuple[str, str]]:
        """Yield (topic, subtopic) pairs that match current filters."""

        for topic in course.get("outline") or []:
            topic_title = str(topic.get("title") or "").strip()
            if not topic_title:
                continue
            if normalized_topics and topic_title.lower() not in normalized_topics:
                continue
            for subtopic in topic.get("subtopics") or []:
                subtopic_title = str(subtopic).strip()
                if not subtopic_title:
                    continue
                if normalized_subtopics and subtopic_title.lower() not in normalized_subtopics:
                    continue
                yield topic_title, subtopic_title

    def _mark_cache_completion(
        self,
        *,
        config: QuestionBatchConfig,
        course: Dict[str, Any],
        topic_title: str,
        subtopic_title: str,
        request: RequestPlan,
    ) -> None:
        """Persist cache metadata for a completed batch."""
        cache = self._cache_for(config.cache_dir)
        course_code = str(course.get("code", ""))

        # Update cache.json with batch completion
        cache.mark_batch_completed(course_code, topic_title, subtopic_title, request.name)

    def _finalize_course_progress(
        self,
        *,
        config: QuestionBatchConfig,
        course: Dict[str, Any],
        progress: CourseProgressCache,
    ) -> None:
        """Update Firestore once a course finishes processing."""

        if not config.store_firestore:
            return

        try:
            store = self._resolve_firestore()
            if not store:
                return

            normalized_topics = config.normalized_topics()
            normalized_subtopics = config.normalized_subtopics()
            target_pairs = list(
                self._iter_target_subtopics(
                    course,
                    normalized_topics,
                    normalized_subtopics,
                )
            )

            if not target_pairs:
                return

            topics_data = progress.data.get("topics", {})
            questions_per_subtopic = (
                progress.theory_target
                + progress.calc_target
                + progress.calc_target
            )

            total_topics = len(target_pairs)
            completed_topics = 0
            errored_topics = 0
            completed_questions = 0

            for topic_name, subtopic_name in target_pairs:
                topic_entry = topics_data.get(topic_name, {})
                entry = (
                    topic_entry.get("subtopics", {})
                    .get(subtopic_name)
                )
                if not entry:
                    continue

                theory_progress = min(
                    entry.get("theory_progress", 0),
                    progress.theory_target,
                )
                calc_progress = min(
                    entry.get("calculation_progress", 0),
                    progress.calc_target,
                )
                calc2_progress = min(
                    entry.get("calc_progress2", 0),
                    progress.calc_target,
                )
                completed_questions += theory_progress + calc_progress + calc2_progress

                state = entry.get("state", "in_progress")
                if state == "completed":
                    completed_topics += 1
                elif state == "error":
                    errored_topics += 1

            total_questions = questions_per_subtopic * total_topics
            completed_questions = min(completed_questions, total_questions)

            if completed_topics < total_topics or errored_topics > 0:
                # Only record progress once an entire course finishes successfully.
                return

            store.update_generation_progress(
                course_code=str(course.get("code", "")),
                course_title=course.get("title", ""),
                department=course.get("department", "Unknown"),
                status="completed",
                total_topics=total_topics,
                completed_topics=completed_topics,
                total_questions=total_questions,
                completed_questions=completed_questions,
                errored_topics=errored_topics,
            )
        except Exception as exc:
            logger.warning("Failed to update Firestore progress for %s: %s", course.get("code"), exc)

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
        default=str(config.courses_json_path_resolved),
        help="Path to courses.json containing outlines",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(config.cache_dir_resolved),
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
        default=config.gemini_default_model,
        help="Gemini model name (e.g. gemini-2.5-flash)",
    )
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling value")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=10000,
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
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=False,
        help="Enable thinking mode for Gemini models",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=12700,
        help="Thinking budget in tokens (default: 12700)",
    )
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
    global config
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    courses_path = Path(args.courses_json)
    course_code = args.course_code or "all"
    output_path = Path(args.output_jsonl) if args.output_jsonl else None

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

    # Initialize email notifications if enabled
    email_service = None
    try:
        from services.Email.email_service import get_email_service

        candidate = get_email_service()
        if getattr(candidate, "enabled", False):
            email_service = candidate
    except Exception as exc:
        logger.warning("Email service unavailable: %s", exc)
        email_service = None

    if email_service:
        try:
            course_codes = [str(row.get("code", "unknown")) for row in courses]
            email_service.send_generation_started(
                course_codes,
                theory_per_request=args.theory_per_request,
                calc_per_request=args.calc_per_request,
                resume=not args.no_resume,
                store_firestore=not args.skip_firestore,
                model=args.model,
                temperature=args.temperature,
            )
        except Exception as exc:
            logger.warning("Failed to send start notification: %s", exc)

    all_questions = []
    common_config = {
        "courses_json_path": courses_path,
        "cache_dir": Path(args.cache_dir),
        "rag_topk_override": args.rag_topk,
        "rag_final_k_override": args.rag_final_k,
        "rag_tau_override": args.rag_tau,
        "rag_min_similarity_override": args.rag_min_sim,
        "rag_where": args.rag_where,
        "theory_questions_per_request_override": args.theory_per_request,
        "calc_questions_per_request_override": args.calc_per_request,
        "resume": not args.no_resume,
        "store_firestore": not args.skip_firestore,
        "request_delay_override": args.request_delay,
        "delay_jitter_override": args.delay_jitter,
        "gemini_model_override": args.model,
        "gemini_temperature_override": args.temperature,
        "gemini_top_p_override": args.top_p,
        "gemini_max_output_tokens_override": args.max_output_tokens,
        "request_attempts_override": args.request_attempts,
        "rag_attempts_override": args.rag_attempts,
        "latex_wrap_steps": not args.no_latex_wrap,
        "target_topics": args.topics,
        "target_subtopics": args.subtopics,
        "output_path": output_path,
        "use_thinking_override": args.thinking,
        "thinking_budget_override": args.thinking_budget,
        "coursegen_debug_override": (
            config.coursegen_debug if hasattr(config, "coursegen_debug") else False
        ),
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
            use_thinking=args.thinking,
            thinking_budget=args.thinking_budget,
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
        course_start_time = time.time()
        course_question_count = 0

        try:
            questions = runner.run(config)
            all_questions.extend(questions)
            course_question_count = len(questions)
            logger.info("Generated %d questions for %s", len(questions), course_code)

            if email_service:
                try:
                    email_service.send_course_finished(
                        course_code=course_code,
                        course_title=course.get("title", ""),
                        question_count=course_question_count,
                        duration_seconds=time.time() - course_start_time,
                        status="completed",
                    )
                except Exception as exc:
                    logger.warning("Failed to send completion notification for %s: %s", course_code, exc)
        except RuntimeError as exc:
            # Handle forced termination when all API keys are exhausted
            if "ALL API KEYS EXHAUSTED" in str(exc) or "CRITICAL" in str(exc):
                logger.error(
                    "🚨 CRITICAL: All API keys exhausted during %s processing. "
                    "Terminating all question generation operations.",
                    course_code
                )
                # Save any questions generated so far
                if all_questions and output_path:
                    write_jsonl(str(output_path), [q.model_dump() for q in all_questions])
                    logger.info("Saved %d questions generated before termination to %s", len(all_questions), output_path)

                if email_service:
                    try:
                        duration = time.time() - course_start_time
                        email_service.send_course_finished(
                            course_code=course_code,
                            course_title=course.get("title", ""),
                            question_count=course_question_count,
                            duration_seconds=duration,
                            status="error",
                            error=str(exc),
                        )

                        api_manager = getattr(gemini_service, "api_key_manager", None)
                        exhausted = 0
                        total = 0
                        if api_manager:
                            total = len(getattr(api_manager, "api_keys", []) or [])
                            cache_data = getattr(api_manager, "cache_data", {}) or {}
                            key_data = cache_data.get("keys", {})
                            exhausted = len([key for key, meta in key_data.items() if meta.get("exhausted")])

                        email_service.send_api_exhaustion_alert(
                            exhausted_keys=exhausted,
                            total_keys=total,
                            model=getattr(gemini_service, "model", "unknown"),
                            questions_generated=len(all_questions),
                        )
                    except Exception as notification_error:
                        logger.warning("Failed to send detailed email notification: %s", notification_error)

                raise exc  # Re-raise to terminate the entire process
            else:
                logger.error("RuntimeError for %s: %s", course_code, exc)
                if email_service:
                    try:
                        email_service.send_course_finished(
                            course_code=course_code,
                            course_title=course.get("title", ""),
                            question_count=course_question_count,
                            duration_seconds=time.time() - course_start_time,
                            status="error",
                            error=str(exc),
                        )
                    except Exception as notif_error:
                        logger.warning("Failed to send error notification for %s: %s", course_code, notif_error)
                continue
        except ValidationError as exc:
            logger.error("Validation failed for %s: %s", course_code, exc)
            if email_service:
                try:
                    email_service.send_course_finished(
                        course_code=course_code,
                        course_title=course.get("title", ""),
                        question_count=course_question_count,
                        duration_seconds=time.time() - course_start_time,
                        status="error",
                        error=str(exc),
                    )
                except Exception as notif_error:
                    logger.warning("Failed to send validation error notification for %s: %s", course_code, notif_error)
            continue
        except Exception as exc:
            logger.error("Question generation failed for %s: %s", course_code, exc)
            if email_service:
                try:
                    email_service.send_course_finished(
                        course_code=course_code,
                        course_title=course.get("title", ""),
                        question_count=course_question_count,
                        duration_seconds=time.time() - course_start_time,
                        status="error",
                        error=str(exc),
                    )
                except Exception as notif_error:
                    logger.warning("Failed to send failure notification for %s: %s", course_code, notif_error)
            continue

    if output_path:
        write_jsonl(str(output_path), [q.model_dump() for q in all_questions])
        logger.info("Saved %d total questions to %s", len(all_questions), output_path)
    else:
        logger.info("Generated %d total questions across %d courses", len(all_questions), len(courses))

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
