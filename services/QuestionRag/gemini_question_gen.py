"""Legacy compatibility module for Gemini pipeline.

This module now acts as a thin facade that re-exports the course outline
pipeline (see :mod:`services.QuestionRag.pipelines.course_outline_generator`) and exposes
question-generation utilities from :mod:`services.QuestionRag.pipelines.question_generator`.

The actual course outline logic lives in ``course_outline_generator.py`` and the
question generation logic is expected to live under ``question_generator.py``.
Keeping this shim lets existing imports and CLI invocations keep functioning
while making the separation between the two domains explicit.
"""

from __future__ import annotations

from .pipelines.course_outline_generator import (
    CACHE_DIR,
    CHROMA_COLLECTION,
    CHROMA_OUT_DIR,
    CHROMA_PATH,
    COURSE_DELAY_S,
    COURSES_JSON,
    CourseStore,
    ChromaCourseProgress,
    ChromaCoursesRunner,
    DepartmentRunner,
    GEMINI_THINKING_MODEL,
    GEMMA_MODEL,
    GeminiQuestionGen,
    MAX_OUTPUT_TOKENS,
    ModelClient,
    OutlineCache,
    OutlineProgress,
    QUERY_DELAY_S,
    TOP_P,
    RAG_MAX_TOTAL,
    RAG_MIN_SIM,
    RAG_TAU,
    RAG_TOPK_PER_QUERY,
    SUB_RAG_FINAL_K,
    SUB_RAG_MIN_SIM,
    SUB_RAG_TAU,
    SUB_RAG_TOPK_PER_QUERY,
    TEMPERATURE,
    THINKING_BUDGET,
    TOPIC_DELAY_S,
    DELAY_JITTER_FRAC,
    ENABLE_SUBTOPIC_RAG,
    main as outline_main,
)

try:
    # Question generation utilities should live here going forward.
    from .pipelines.question_generator import QuestionGenerator, QuestionBatchRunner  # noqa: F401
except ImportError:  # pragma: no cover - keep compatibility if file not yet created
    QuestionGenerator = None  # type: ignore
    QuestionBatchRunner = None  # type: ignore

__all__ = [
    # Outline exports
    "CACHE_DIR",
    "CHROMA_COLLECTION",
    "CHROMA_OUT_DIR",
    "CHROMA_PATH",
    "COURSE_DELAY_S",
    "COURSES_JSON",
    "CourseStore",
    "ChromaCourseProgress",
    "ChromaCoursesRunner",
    "DepartmentRunner",
    "GEMINI_THINKING_MODEL",
    "GEMMA_MODEL",
    "GeminiQuestionGen",
    "MAX_OUTPUT_TOKENS",
    "ModelClient",
    "OutlineCache",
    "OutlineProgress",
    "QUERY_DELAY_S",
    "TOP_P",
    "RAG_MAX_TOTAL",
    "RAG_MIN_SIM",
    "RAG_TAU",
    "RAG_TOPK_PER_QUERY",
    "SUB_RAG_FINAL_K",
    "SUB_RAG_MIN_SIM",
    "SUB_RAG_TAU",
    "SUB_RAG_TOPK_PER_QUERY",
    "TEMPERATURE",
    "THINKING_BUDGET",
    "TOPIC_DELAY_S",
    "DELAY_JITTER_FRAC",
    "ENABLE_SUBTOPIC_RAG",
    "outline_main",
    # Question exports (may be None if not implemented yet)
    "QuestionGenerator",
    "QuestionBatchRunner",
]


def main() -> None:
    """Proxy CLI entry point for historical ``gemini_question_gen.py`` usage."""

    outline_main()


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    main()
