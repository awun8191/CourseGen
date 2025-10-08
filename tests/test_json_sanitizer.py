import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.QuestionRag.pipelines.question_generator import QuestionGenerator
from services.QuestionRag.pipelines.json_utils import extract_json_content, simple_json_load


class DummyGemini:  # minimal stub for QuestionGenerator
    pass


def make_generator():
    return QuestionGenerator(gemini_service=DummyGemini(), use_structured=False)


def test_extract_json_with_latex_equations():
    generator = make_generator()
    raw = r"""```json
{
  "questions": [
    {
      "question": "A component has failure rate (\(\lambda\)) = 2 \\times 10^{-5}."
    }
  ]
}
```"""
    parsed = extract_json_content(raw)
    assert isinstance(parsed, dict)
    assert parsed["questions"][0]["question"].startswith("A component")


def test_extract_json_with_single_backslashes():
    generator = make_generator()
    raw = r"""```json
{
  "questions": [
    {
      "question": "Compute P(\(Z\geq 1.96\))."
    }
  ]
}
```"""
    parsed = extract_json_content(raw)
    assert parsed["questions"][0]["question"].startswith("Compute")


def test_extract_json_with_spaced_latex_sequences():
    generator = make_generator()
    raw = r"""```json
{
  "questions": [
    {
      "question": "A critical component has failure rate \( \lambda = 2 \times 10^{-5} \)."
    }
  ]
}
```"""
    parsed = extract_json_content(raw)
    assert "lambda" in parsed["questions"][0]["question"].lower()


def test_extract_json_with_trailing_commas():
    generator = make_generator()
    raw = r"""```json
{
  "questions": [
    {
      "question": "A wing section is subjected to a distributed load \( q(x) = q_0 \left(1 - \frac{x}{b}\right) \)."
    },
  ],
}
```"""
    parsed = extract_json_content(raw)
    assert parsed["questions"][0]["question"].startswith("A wing section")
    assert len(parsed["questions"]) == 1


def test_extract_json_repair_truncated_payload():
    raw = r"""```json
{
  "questions": [
    {
      "question": "Determine the shear force distribution \( V(x) \) for a beam.",
      "options": [
        "Option A",
        "Option B",
        "Option C",
        "Option D"
      ]
    }
  ]
```"""
    parsed = extract_json_content(raw)
    assert parsed["questions"][0]["options"][0] == "Option A"


def test_simple_json_load_recovers_invalid_latex_backslashes():
    raw = r"""{"questions": [{"question": "Compute the field strength \mu_0 I / (2 \pi r)"}]}"""
    parsed = simple_json_load(raw)
    question = parsed["questions"][0]["question"]
    assert "mu_0" in question.replace("\\", "")


def test_simple_json_load_preserves_unicode_sequences():
    raw = r"""{"symbol": "\u03bc"}"""
    parsed = simple_json_load(raw)
    assert parsed["symbol"] == "\u03bc"
