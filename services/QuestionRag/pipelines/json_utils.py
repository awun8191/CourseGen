"""JSON parsing and normalization utilities for question generation."""

from __future__ import annotations

import json
import os
import random
import re
import string
import time
from pathlib import Path
from typing import Any, Optional

from .models import GeminiQuestionBatch


class QuestionGenerationError(RuntimeError):
    """Raised when the model returns invalid or incomplete data."""


def extract_json_payload(text: str) -> Optional[str]:
    """Extract JSON payload from text response."""
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


def strip_trailing_commas(text: str) -> str:
    """Remove trailing commas from JSON text to fix common formatting issues."""
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


def repair_truncated_json(text: str) -> Optional[str]:
    """Repair truncated JSON by adding missing closing brackets."""
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


def simple_json_load(content: str) -> Any:
    """Load JSON with enhanced error handling and LaTeX escaping."""
    if not content:
        raise ValueError("Empty JSON payload")
    if not isinstance(content, str):
        content = str(content)

    def preprocess_latex_in_math(text: str) -> str:
        """Pre-process LaTeX within $...$ delimiters to ensure proper escaping."""
        if not text or '$' not in text:
            return text
        
        # Find all math expressions and fix backslashes within them
        def fix_math_expr(match):
            content = match.group(1)
            # Double any single backslashes that aren't already doubled
            # This regex looks for backslash not followed by another backslash
            fixed = re.sub(r'(?<!\\)\\(?!\\)', r'\\\\', content)
            return f'${fixed}$'
        
        # Process inline math $...$
        result = re.sub(r'\$([^$]+)\$', fix_math_expr, text)
        return result

    def escape_latex_for_json(text: str) -> str:
        """Escape stray LaTeX backslashes without breaking valid JSON escapes."""

        if not text:
            return text

        valid_escapes = {'"', "\\", '/', 'b', 'f', 'n', 'r', 't'}
        result: list[str] = []
        idx = 0
        length = len(text)

        while idx < length:
            ch = text[idx]
            if ch != "\\":
                result.append(ch)
                idx += 1
                continue

            # Last character being a backslash; double it and move on.
            if idx + 1 >= length:
                result.append("\\\\")
                idx += 1
                continue

            nxt = text[idx + 1]

            # Preserve valid JSON escapes (e.g. \n, \", \u1234).
            if nxt in valid_escapes:
                result.append("\\" + nxt)
                idx += 2
                continue

            if nxt == "u":
                hex_digits = text[idx + 2 : idx + 6]
                if len(hex_digits) == 4 and all(char in string.hexdigits for char in hex_digits):
                    result.append("\\u" + hex_digits)
                    idx += 6
                    continue

            # Treat everything else as a LaTeX command and escape the backslash only.
            result.append("\\\\")
            idx += 1

        return "".join(result)

    def attempt_load(candidate: str) -> Optional[Any]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    def add_candidate(candidates: list[str], value: Optional[str]) -> None:
        if not value:
            return
        if value not in candidates:
            candidates.append(value)

    candidates: list[str] = []
    
    # Try original first
    add_candidate(candidates, content)
    add_candidate(candidates, strip_trailing_commas(content))
    
    # Pre-process LaTeX in math expressions
    preprocessed = preprocess_latex_in_math(content)
    add_candidate(candidates, preprocessed)
    add_candidate(candidates, strip_trailing_commas(preprocessed))

    # Apply full sanitization
    sanitized = escape_latex_for_json(preprocessed)
    add_candidate(candidates, sanitized)
    add_candidate(candidates, strip_trailing_commas(sanitized))

    # Try doubling as last resort (for cases where model used single backslash)
    doubled = sanitized.replace("\\", "\\\\")
    add_candidate(candidates, doubled)
    add_candidate(candidates, strip_trailing_commas(doubled))

    for candidate in candidates:
        result = attempt_load(candidate)
        if result is not None:
            return result

    for candidate in candidates:
        repaired = repair_truncated_json(candidate)
        if not repaired or repaired == candidate:
            continue
        result = attempt_load(repaired)
        if result is not None:
            return result

    raise ValueError("Failed to parse JSON content")


def extract_json_content(text: str) -> Any:
    """Extract and parse JSON content from text response."""
    if not text:
        raise ValueError("No JSON content in empty response")

    # Prefer fenced code blocks
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        snippet = match.group(1).strip()
        if snippet:
            return simple_json_load(snippet)

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
        return simple_json_load(snippet)

    raise ValueError(f"No JSON found in response: {text[:200]}...")


def normalize_solution_steps(value: Any) -> list[str]:
    """Normalize solution steps to a consistent list format."""
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


def _ensure_correct_answer_indexes(question: dict[str, Any]) -> None:
    """Ensure `correct_answer_indexes` exists using legacy fields as fallback."""

    if not isinstance(question, dict):
        return

    indexes = question.get("correct_answer_indexes")
    if isinstance(indexes, list) and indexes:
        return

    options = question.get("options")
    if not isinstance(options, list) or len(options) != 4:
        return

    def _set_index(idx: int) -> None:
        question["correct_answer_indexes"] = [int(idx)]
        if not question.get("correct_answer_text") and 0 <= idx < len(options):
            question["correct_answer_text"] = options[idx]

    letter = question.get("correct_answer")
    if isinstance(letter, str) and letter.strip():
        mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
        idx = mapping.get(letter.strip().upper()[0])
        if idx is not None and 0 <= idx < len(options):
            _set_index(idx)
            return

    answer_text = question.get("correct_answer_text")
    if isinstance(answer_text, str) and answer_text.strip():
        for idx, option in enumerate(options):
            if str(answer_text).strip().lower() == str(option).strip().lower():
                _set_index(idx)
                return


def normalize_question_payload(payload: Any) -> dict[str, Any]:
    """Normalize question payload to standard format."""
    if isinstance(payload, list):
        normalized = []
        for item in payload:
            if isinstance(item, dict):
                item["solution_steps"] = normalize_solution_steps(
                    item.get("solution_steps")
                )
                _ensure_correct_answer_indexes(item)
                normalized.append(item)
        return {"questions": normalized}

    if not isinstance(payload, dict):
        return {"questions": []}

    questions = payload.get("questions")
    if isinstance(questions, list):
        for question in questions:
            if not isinstance(question, dict):
                continue
            question["solution_steps"] = normalize_solution_steps(
                question.get("solution_steps")
            )
            _ensure_correct_answer_indexes(question)
        return payload

    # Handle single-question payloads
    if isinstance(payload, dict) and "question" in payload:
        normalized_question = dict(payload)
        normalized_question["solution_steps"] = normalize_solution_steps(
            normalized_question.get("solution_steps")
        )
        _ensure_correct_answer_indexes(normalized_question)
        return {"questions": [normalized_question]}

    return payload


def parse_batch_from_raw(raw_result: str) -> GeminiQuestionBatch:
    """Parse raw Gemini response into structured batch."""
    parsed_response = extract_json_content(raw_result)
    normalized = normalize_question_payload(parsed_response)
    return GeminiQuestionBatch.model_validate(normalized)


def dump_failed_payload(raw_result: str) -> Optional[Path]:
    """Dump failed payload to debug file."""
    try:
        dump_dirs = [
            Path(
                os.environ.get(
                    "COURSEGEN_DEBUG_DUMP_DIR", str(DEFAULT_CACHE_ROOT / "failed_responses")
                )
            ),
            Path.cwd() / "failed_responses",
        ]

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        random_suffix = f"{random.randint(0, 9999):04d}"

        for dump_root in dump_dirs:
            try:
                dump_root.mkdir(parents=True, exist_ok=True)
                path = dump_root / f"failed_payload_{timestamp}_{random_suffix}.json"
                path.write_text(raw_result, encoding="utf-8")
                return path
            except Exception:
                continue
        return None
    except Exception:
        return None
