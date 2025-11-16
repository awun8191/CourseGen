"""Configuration for question generation pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..utils import ChromaQuery, CourseProgressCache, MetaData, QuestionCache
from ..utils.batch_utils import validate_answer_in_options, validate_options
from .models import GeminiGeneratedQuestion, GeminiQuestionBatch
from .question_gen_config import get_question_gen_config


# Import centralized configuration
try:
    from config import load_config
    config = load_config()
    REPO_ROOT = config.repo_root
    DEFAULT_COURSES_JSON = config.courses_json_path_resolved
    DEFAULT_CACHE_ROOT = config.cache_dir_resolved
    DEFAULT_MODEL = config.gemini_default_model
except ImportError:
    # Fallback to environment variables if centralized config not available
    import os
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

    # Import centralized config
    _central_config: QuestionGenerationConfig = None

    def __post_init__(self):
        if self._central_config is None:
            self._central_config = get_question_gen_config()

    # Properties that delegate to centralized config
    @property
    def rag_topk(self) -> int:
        if self.rag_topk_override is not None:
            return int(self.rag_topk_override)
        return self._central_config.rag_topk

    @property
    def rag_final_k(self) -> int:
        if self.rag_final_k_override is not None:
            return int(self.rag_final_k_override)
        return self._central_config.rag_final_k

    @property
    def rag_tau(self) -> float:
        if self.rag_tau_override is not None:
            return float(self.rag_tau_override)
        return self._central_config.rag_tau

    @property
    def rag_min_similarity(self) -> float:
        if self.rag_min_similarity_override is not None:
            return float(self.rag_min_similarity_override)
        return self._central_config.rag_min_similarity

    @property
    def theory_questions_per_request(self) -> int:
        if self.theory_questions_per_request_override is not None:
            return int(self.theory_questions_per_request_override)
        return self._central_config.theory_questions_per_request

    @property
    def calc_questions_per_request(self) -> int:
        if self.calc_questions_per_request_override is not None:
            return int(self.calc_questions_per_request_override)
        return self._central_config.calc_questions_per_request

    @property
    def request_delay_s(self) -> float:
        if self.request_delay_override is not None:
            return float(self.request_delay_override)
        return self._central_config.request_delay_s

    @property
    def delay_jitter(self) -> float:
        if self.delay_jitter_override is not None:
            return float(self.delay_jitter_override)
        return self._central_config.delay_jitter

    @property
    def gemini_model(self) -> str:
        if self.gemini_model_override:
            return str(self.gemini_model_override)
        return self._central_config.gemini_model

    @property
    def gemini_calc_model(self) -> str:
        """Model to use specifically for calculation questions."""
        return self._central_config.gemini_calc_model

    @property
    def gemini_temperature(self) -> float:
        if self.gemini_temperature_override is not None:
            return float(self.gemini_temperature_override)
        return self._central_config.gemini_temperature

    @property
    def gemini_top_p(self) -> float:
        if self.gemini_top_p_override is not None:
            return float(self.gemini_top_p_override)
        return self._central_config.gemini_top_p

    @property
    def gemini_max_output_tokens(self) -> int:
        if self.gemini_max_output_tokens_override is not None:
            return int(self.gemini_max_output_tokens_override)
        return self._central_config.gemini_max_output_tokens

    @property
    def request_attempts(self) -> int:
        if self.request_attempts_override is not None:
            return int(self.request_attempts_override)
        return self._central_config.request_attempts

    @property
    def rag_attempts(self) -> int:
        if self.rag_attempts_override is not None:
            return int(self.rag_attempts_override)
        return self._central_config.rag_attempts

    @property
    def rag_context_limit(self) -> int:
        if self.rag_context_limit_override is not None:
            return int(self.rag_context_limit_override)
        return self._central_config.rag_context_limit

    @property
    def default_theory_difficulty_rank(self) -> int:
        return self._central_config.default_theory_difficulty_rank

    @property
    def default_calculation_difficulty_rank(self) -> int:
        return self._central_config.default_calculation_difficulty_rank

    @property
    def use_thinking(self) -> bool:
        if self.use_thinking_override is not None:
            return bool(self.use_thinking_override)
        return self._central_config.use_thinking

    @property
    def thinking_budget(self) -> int:
        if self.thinking_budget_override is not None:
            return int(self.thinking_budget_override)
        return self._central_config.thinking_budget

    @property
    def coursegen_debug(self) -> bool:
        if self.coursegen_debug_override is not None:
            return bool(self.coursegen_debug_override)
        return os.environ.get("COURSEGEN_DEBUG", "false").lower() == "true"

    @property
    def max_topic_workers(self) -> int:
        if self.max_topic_workers_override is not None:
            return int(self.max_topic_workers_override)
        return self._central_config.max_topic_workers

    @property
    def worker_timeout(self) -> int:
        if self.worker_timeout_override is not None:
            return int(self.worker_timeout_override)
        return self._central_config.worker_timeout

    @property
    def worker_retry_attempts(self) -> int:
        if self.worker_retry_attempts_override is not None:
            return int(self.worker_retry_attempts_override)
        return self._central_config.worker_retry_attempts

    @property
    def enable_topic_parallelism(self) -> bool:
        if self.enable_topic_parallelism_override is not None:
            return bool(self.enable_topic_parallelism_override)
        return self._central_config.enable_topic_parallelism

    # Optional overrides for specific use cases
    rag_topk_override: Optional[int] = None
    rag_final_k_override: Optional[int] = None
    rag_tau_override: Optional[float] = None
    rag_min_similarity_override: Optional[float] = None
    theory_questions_per_request_override: Optional[int] = None
    calc_questions_per_request_override: Optional[int] = None
    request_delay_override: Optional[float] = None
    delay_jitter_override: Optional[float] = None
    gemini_model_override: Optional[str] = None
    gemini_temperature_override: Optional[float] = None
    gemini_top_p_override: Optional[float] = None
    gemini_max_output_tokens_override: Optional[int] = None
    request_attempts_override: Optional[int] = None
    rag_attempts_override: Optional[int] = None
    rag_context_limit_override: Optional[int] = None
    use_thinking_override: Optional[bool] = None
    thinking_budget_override: Optional[int] = None
    coursegen_debug_override: Optional[bool] = None
    max_topic_workers_override: Optional[int] = None
    worker_timeout_override: Optional[int] = None
    worker_retry_attempts_override: Optional[int] = None
    enable_topic_parallelism_override: Optional[bool] = None
    rag_where: Optional[Dict[str, Any]] = None
    latex_wrap_steps: bool = True
    target_topics: Optional[Sequence[str]] = None
    target_subtopics: Optional[Sequence[str]] = None
    output_path: Optional[Path] = None
    custom_plan: Optional[List[RequestPlan]] = None
    resume: bool = True
    store_firestore: bool = True

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
        # Generate exactly 20 questions per subtopic: 10 theory + 10 calculation (split across two requests)
        calc_count = self.calc_questions_per_request
        if calc_count != 5:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "Overriding calc_questions_per_request=%s to 5 to honour two 5-question calculation batches",
                calc_count,
            )
            calc_count = 5

        return [
            RequestPlan(
                name="theory-1",
                kind="theory",
                question_count=self.theory_questions_per_request,
                difficulty_rank=self.default_theory_difficulty_rank,
            ),
            RequestPlan(
                name="calculation-1",
                kind="calculation",
                question_count=calc_count,
                difficulty_rank=self.default_calculation_difficulty_rank,
            ),
            RequestPlan(
                name="calculation-2",
                kind="calculation",
                question_count=calc_count,
                difficulty_rank=self.default_calculation_difficulty_rank,
            ),
        ]
