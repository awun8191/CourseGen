"""Configuration for question generation pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..utils import ChromaQuery, CourseProgressCache, MetaData, QuestionCache
from ..utils.batch_utils import validate_answer_in_options, validate_options
from .models import GeminiGeneratedQuestion, GeminiQuestionBatch


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
    DEFAULT_MODEL = os.environ.get("COURSEGEN_QUESTION_MODEL", "gemini-2.5-flash")


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
    # Use centralized configuration values
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
    gemini_temperature: float = 0.15
    gemini_top_p: float = 0.6
    gemini_max_output_tokens: int = 10000
    request_attempts: int = 2
    rag_attempts: int = 2
    rag_context_limit: int = 8
    latex_wrap_steps: bool = True
    target_topics: Optional[Sequence[str]] = None
    target_subtopics: Optional[Sequence[str]] = None
    output_path: Optional[Path] = None
    custom_plan: Optional[List[RequestPlan]] = None
    use_thinking: bool = False
    thinking_budget: int = 12700
    coursegen_debug: bool = False
    default_theory_difficulty_rank: int = int(os.environ.get("COURSEGEN_THEORY_DIFFICULTY_RANK", "2"))
    default_calculation_difficulty_rank: int = int(os.environ.get("COURSEGEN_CALCULATION_DIFFICULTY_RANK", "2"))

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