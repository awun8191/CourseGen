from __future__ import annotations

import json
from pathlib import Path

from services.QuestionRag.pipelines.config import QuestionBatchConfig, RequestPlan
from services.QuestionRag.pipelines.question_generator import (
    QuestionBatchRunner,
    QuestionGenerator,
)


class FakeGeminiService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, prompt: str, **_: object) -> dict:
        self.calls.append(prompt)
        return {
            "questions": [
                {
                    "question": "Sample question?",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer_indexes": [0],
                    "correct_answer_text": "Option A",
                    "explanation": "Because Option A is correct",
                    "solution_steps": ["Step 1"],
                }
            ]
        }


class FakeChromaQuery:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search_with_temperature(self, q: str, **_: object) -> list[dict]:
        self.queries.append(q)
        return [
            {
                "snippet": "Reference text",
                "score": 0.95,
                "meta": {"path": "doc.pdf", "chunk_index": 1},
            },
            {
                "snippet": "Extra reference",
                "score": 0.9,
                "meta": {"path": "doc.pdf", "chunk_index": 2},
            },
        ]


class FakeFireStore:
    def __init__(self) -> None:
        self.saved: list[dict] = []
        self.progress_updates: list[dict] = []

    def set_question(self, question) -> None:  # pragma: no cover - simple container
        self.saved.append(question)

    def update_generation_progress(self, **payload) -> None:  # pragma: no cover - simple container
        self.progress_updates.append(payload)


def _write_courses(tmp_path: Path) -> Path:
    courses_path = tmp_path / "courses.json"
    courses_path.write_text(
        json.dumps(
            [
                {
                    "code": "EEE 101",
                    "title": "Intro Electronics",
                    "levels": ["100"],
                    "semesters": ["FIRST"],
                    "outline": [
                        {"title": "Resistors", "subtopics": ["Ohm's Law"]},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    return courses_path


def test_question_generation_with_cache(tmp_path: Path) -> None:
    courses_path = _write_courses(tmp_path)
    cache_dir = tmp_path / "cache"

    fake_gemini = FakeGeminiService()
    fake_rag = FakeChromaQuery()
    fake_store = FakeFireStore()

    generator = QuestionGenerator(
        gemini_service=fake_gemini,
        rag_client=fake_rag,
        firestore=fake_store,
    )
    config = QuestionBatchConfig(
        course_code="EEE 101",
        courses_json_path=courses_path,
        cache_dir=cache_dir,
        custom_plan=[
            RequestPlan(name="theory-1", kind="theory", question_count=1, difficulty_rank=2),
            RequestPlan(name="calculation-1", kind="calculation", question_count=1, difficulty_rank=5),
            RequestPlan(name="calculation-2", kind="calculation", question_count=1, difficulty_rank=5),
        ],
        theory_questions_per_request_override=1,
        calc_questions_per_request_override=1,
        request_delay_override=0,
    )

    runner = QuestionBatchRunner(generator)
    results = runner.run(config)

    assert len(results) == 3
    assert len(fake_gemini.calls) == 3
    assert len(fake_store.saved) == 3

    # Second run should reuse cache and avoid additional Gemini calls
    results_again = runner.run(config)
    assert len(results_again) == 0
    assert len(fake_gemini.calls) == 3  # unchanged
    assert len(fake_store.saved) == 3  # cached questions do not re-persist

    calc_questions = [q for q in results if q.question_type == "calculation"]
    assert calc_questions
    for question in calc_questions:
        assert all(step.startswith("\\(") or step.startswith("$") for step in question.solution_steps)


def test_generation_skips_when_count_mismatch(tmp_path: Path) -> None:
    courses_path = _write_courses(tmp_path)
    cache_dir = tmp_path / "cache-mismatch"

    fake_gemini = FakeGeminiService()
    fake_rag = FakeChromaQuery()
    fake_store = FakeFireStore()

    generator = QuestionGenerator(
        gemini_service=fake_gemini,
        rag_client=fake_rag,
        firestore=fake_store,
    )
    config = QuestionBatchConfig(
        course_code="EEE 101",
        courses_json_path=courses_path,
        cache_dir=cache_dir,
        custom_plan=[
            RequestPlan(
                name="theory-only",
                kind="theory",
                question_count=2,
                difficulty_rank=5,
            )
        ],
        request_attempts_override=1,
        request_delay_override=0,
    )

    runner = QuestionBatchRunner(generator)
    results = runner.run(config)

    assert results == []
    assert len(fake_gemini.calls) == 1
    assert fake_store.saved == []

    cache = generator._cache_for(cache_dir)
    key_prefix = cache.make_key("EEE 101", "Resistors", "Ohm's Law", "placeholder")
    states = cache.subtopic_request_states(key_prefix, ["theory-only"])
    assert states["theory-only"] == "failed"
