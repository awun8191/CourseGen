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
    """CLI entry point: outlines by default, or questions if flagged."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Generate outlines or questions using Gemini and RAG.")
    parser.add_argument("--generate-questions", action="store_true", help="Run question generation instead of outlines")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy", help="Difficulty level for questions")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without API calls")
    parser.add_argument("--department_from", help='For outlines: e.g. "EEE 315"')
    # Add other outline args if needed, but pass unknown to outline_main
    args, unknown = parser.parse_known_args()

    if args.generate_questions:
        if QuestionBatchRunner is None:
            print("Question generation not implemented yet. Run without --generate-questions for outlines.")
            sys.exit(1)
        else:
            from .pipelines.question_generator import main as question_main
            # Ignore --difficulty as it's not supported; pass remaining args
            sys.argv = [sys.argv[0]] + unknown
            question_main()
    else:
        # Call outline_main with remaining args
        sys.argv = [sys.argv[0]] + unknown
        outline_main()


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    main()