import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.QuestionRag.pipelines.config import QuestionBatchConfig, RequestPlan
from services.QuestionRag.pipelines.question_generator import QuestionGenerator


class DummyGemini:
    def __init__(self, raw_result: str) -> None:
        self._raw_result = raw_result
        self.last_prompt: str | None = None

    def generate(self, prompt: str, **_: object) -> dict[str, str]:  # pragma: no cover - simple stub
        self.last_prompt = prompt
        return {"result": self._raw_result}


class DummyRag:
    pass


@pytest.mark.parametrize("question_count", [1])
def test_calculation_prompt_parses_dollar_wrapped_latex(tmp_path, question_count: int) -> None:
    raw_payload = r"""{
  "questions": [
    {
      "question": "Compute the shear stress $\\tau = \\frac{F}{A}$ when $F = 500\\,\\text{N}$ is applied over an area of $50\\,\\text{cm}^2$.",
      "options": [
        "$\\tau = 1.0\\,\\text{MPa}$",
        "$\\tau = 0.1\\,\\text{MPa}$",
        "$\\tau = 10\\,\\text{kPa}$",
        "$\\tau = 0.01\\,\\text{kPa}$",
      ],
      "correct_answer_indexes": [0],
      "correct_answer_text": "$\\tau = 1.0\\,\\text{MPa}$",
      "explanation": "Using $\\tau = \\frac{F}{A}$ with $A = 5.0\\times10^{-3}\\,\\text{m}^2$ gives $\\tau = 1.0\\,\\text{MPa}$.",
      "solution_steps": [
        "Convert $50\\,\\text{cm}^2$ to $\\text{m}^2$ giving $5.0\\times10^{-3}\\,\\text{m}^2$.",
        "Apply $\\tau = \\frac{F}{A}$ to get $\\tau = 1.0\\,\\text{MPa}$.",
            "Final: $\\tau = 1.0\\,\\text{MPa}$.",
      ],
    }
  ],
}
"""

    gemini = DummyGemini(raw_payload)
    generator = QuestionGenerator(
        gemini_service=gemini,
        rag_client=DummyRag(),
        use_structured=False,
    )

    config = QuestionBatchConfig(
        course_code="AAE 331",
        cache_dir=tmp_path,
        store_firestore=False,
        resume=False,
        request_delay_override=0.0,
        delay_jitter_override=0.0,
        theory_questions_per_request_override=question_count,
        calc_questions_per_request_override=question_count,
        request_attempts_override=1,
    )

    request = RequestPlan(
        name="calc-test",
        kind="calculation",
        question_count=question_count,
        difficulty_rank=5,
    )

    course = {
        "code": "AAE 331",
        "title": "Aerodynamic & Structural Analysis",
        "levels": ["300"],
        "semesters": ["FIRST"],
    }

    rag_sources = [
        {
            "ref_id": "ref-1",
            "path": "dummy.pdf",
            "score": 0.9,
            "snippet": "Sample snippet",
        }
    ]

    questions = generator._call_gemini(
        config=config,
        course=course,
        topic_title="Aerodynamic Analysis",
        subtopic_title="Shear Stress",
        request=request,
        context_text="Sample context about shear stress and material properties.",
        rag_sources=rag_sources,
    )

    assert len(questions) == question_count
    question = questions[0]
    assert "${" not in question.question  # no accidental ${ tokens
    assert "$\\tau = \\frac{F}{A}$" in question.question
    assert "$" in question.explanation
    assert question.solution_steps
    assert "$" in question.solution_steps[0]
    assert gemini.last_prompt is not None and "Ensure every LaTeX expression" in gemini.last_prompt
