import hashlib
import json
from typing import Any, Dict, List


def slugify(value: str) -> str:
    import re
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9\s-]", "", value)
    value = re.sub(r"[\s_]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_options(options: List[str]) -> None:
    if not isinstance(options, list) or len(options) != 4:
        raise ValueError("options must be a list of exactly 4 strings")
    for o in options:
        if not isinstance(o, str):
            raise ValueError("each option must be a string")


def validate_answer_in_options(answer: str, options: List[str]) -> None:
    if answer not in options:
        raise ValueError("answer must be one of the options")


def validate_calc_steps(steps: Dict[str, str]) -> None:
    if not isinstance(steps, dict):
        raise ValueError("steps must be a dict of step_number -> text")
    n = len(steps)
    if n < 3 or n > 8:
        raise ValueError("calculation steps must have 3 to 8 entries")


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

