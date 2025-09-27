"""Validation utilities for question generation."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from data_models.question_model import Question
from .config import RequestPlan
from .json_utils import QuestionGenerationError


def validate_options(options: List[str]) -> None:
    """Validate that options are properly formatted."""
    if len(options) != 4:
        raise QuestionGenerationError(f"Expected 4 options, got {len(options)}")

    for i, option in enumerate(options):
        if not option or not option.strip():
            raise QuestionGenerationError(f"Option {i+1} is empty")


def validate_answer_in_options(answer_text: str | None, options: List[str]) -> None:
    """Validate that the correct answer text exists in options."""
    if not answer_text:
        raise QuestionGenerationError("Answer text cannot be empty")

    for option in options:
        if answer_text.lower() == option.lower():
            return

    raise QuestionGenerationError(f"Answer text '{answer_text}' not found in options")


def normalize_answer(
    indexes_value: Any,
    answer_value: Any,
    answer_text_value: str | None,
    options: List[str],
) -> tuple[int, str, str]:
    """Normalize answer to index, letter, and text format."""
    letters = ["A", "B", "C", "D"]

    # Prefer explicit index array
    if isinstance(indexes_value, list) and indexes_value:
        raw_idx = indexes_value[0]
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            raise QuestionGenerationError("correct_answer_indexes must contain integers") from None
        if idx < 0 or idx >= len(options):
            raise QuestionGenerationError(
                f"correct_answer_indexes[0]={idx} is out of range for {len(options)} options"
            )
        text = options[idx]
        return idx, letters[idx], text

    if answer_text_value:
        text = answer_text_value.strip()
        for idx, option in enumerate(options):
            if text.lower() == option.lower():
                return idx, letters[idx], option

    if isinstance(answer_value, int):
        idx = answer_value - 1
        if 0 <= idx < len(options):
            return idx, letters[idx], options[idx]

    if isinstance(answer_value, str):
        cleaned = answer_value.strip().upper()
        for idx, letter in enumerate(letters):
            if cleaned in {letter, f"OPTION {letter}", f"{letter}.", f"{letter})"}:
                return idx, letter, options[idx]
        for idx, option in enumerate(options):
            if cleaned.lower() == option.lower():
                return idx, letters[idx], option

    raise QuestionGenerationError("Unable to determine correct answer letter")


def ensure_latex_steps(steps: List[str], *, wrap_latex: bool) -> List[str]:
    """Ensure solution steps are properly formatted with LaTeX."""
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
        if any(token in clean for token in ("$", "\\(", "\\[")):
            formatted.append(clean)
        else:
            formatted.append(f"\\({clean}\\)")
    return formatted


def validate_calculation_question(
    question_text: str,
    steps: List[str],
    answer_text: str,
) -> None:
    """Validate calculation question requirements."""
    # Allow calculation questions without numbers in question text - numbers can be in context or steps
    # if not any(ch.isdigit() for ch in question_text):
    #     raise QuestionGenerationError(
    #         "Calculation question missing numeric context in question text"
    #     )
    if len(steps) < 2:
        raise QuestionGenerationError(
            "Calculation question must include at least two solution steps"
        )
    if not any(ch.isdigit() for ch in steps[-1]):
        raise QuestionGenerationError(
            "Final solution step must include numeric result"
        )
    if not answer_text or not any(ch.isdigit() for ch in answer_text):
        raise QuestionGenerationError(
            "Calculation answer text must contain numeric value"
        )


def format_latex_for_display(latex_text: str) -> str:
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


def difficulty_from_rank(rank: int) -> str:
    """Convert difficulty rank to string representation."""
    if rank <= 3:
        return "Easy"
    if rank <= 6:
        return "Medium"
    return "Hard"


def first(value: Any) -> str | None:
    """Get first value from list or return string value."""
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def convert_to_questions(
    llm_questions: List[Dict[str, Any]],
    *,
    course: Dict[str, Any],
    topic_title: str,
    subtopic_title: str,
    request: RequestPlan,
    rag_sources: List[Dict[str, Any]],
    wrap_latex: bool,
) -> List[Question]:
    """Convert LLM questions to Question objects."""
    if not rag_sources:
        raise QuestionGenerationError("RAG sources are required for question generation")

    questions: List[Question] = []
    level = first(course.get("levels"))
    semester = first(course.get("semesters"))
    course_code = str(course.get("code") or "")
    course_title = str(course.get("title") or "")

    for idx, item in enumerate(llm_questions, start=1):
        options = [str(opt).strip() for opt in item.get("options", [])]
        validate_options(options)

        if any(not option for option in options):
            raise QuestionGenerationError("Options must not be empty")

        normalized_options = {option.lower() for option in options}
        if len(normalized_options) != len(options):
            raise QuestionGenerationError("Options must be unique")

        answer_index, answer_letter, answer_text = normalize_answer(
            item.get("correct_answer_indexes"),
            item.get("correct_answer"),
            item.get("correct_answer_text"),
            options,
        )
        validate_answer_in_options(answer_text, options)

        question_text = str(item.get("question") or "").strip()
        if not question_text:
            raise QuestionGenerationError("Question text is empty")

        explanation = str(item.get("explanation") or "").strip()
        if not explanation:
            raise QuestionGenerationError("Explanation is required")

        steps = [str(step).strip() for step in (item.get("solution_steps") or []) if str(step).strip()]
        if request.kind == "calculation":
            steps = ensure_latex_steps(steps, wrap_latex=wrap_latex)
            # Allow empty solution steps for calculation questions instead of raising error
            if not steps:
                steps = []
            # Removed validation to allow more flexible calculation questions
            # validate_calculation_question(
            #     question_text,
            #     steps,
            #     answer_text,
            # )
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
            difficulty=difficulty_from_rank(request.difficulty_rank),
            question=question_text,
            options=options,
            correct_answer_index=answer_index,
            correct_answer=answer_letter,
            correct_answer_text=answer_text,
            explanation=explanation,
            solution_steps=steps,
            rag_sources=[dict(src) for src in rag_sources],
            extra_metadata={
                "request_name": request.name,
                "question_index": idx,
                "generated_at": __import__("time").time(),
                "correct_answer_index": answer_index,
            },
        )
        questions.append(question)
    return questions
