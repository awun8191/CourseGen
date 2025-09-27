from __future__ import annotations

from dataclasses import dataclass

from services.QuestionRag.pipelines.config import QuestionBatchConfig
from services.QuestionRag.pipelines.question_generator import QuestionGenerator
from services.QuestionRag.utils.course_progress import CourseProgressCache


@dataclass
class _FakeFireStore:
    """Collect Firestore progress updates for assertions."""

    calls: list[dict] | None = None

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    def update_generation_progress(
        self,
        *,
        course_code: str,
        course_title: str,
        department: str,
        status: str,
        total_topics: int,
        completed_topics: int,
        total_questions: int,
        completed_questions: int,
        errored_topics: int = 0,
    ) -> None:
        self.calls.append(
            {
                "course_code": course_code,
                "course_title": course_title,
                "department": department,
                "status": status,
                "total_topics": total_topics,
                "completed_topics": completed_topics,
                "total_questions": total_questions,
                "completed_questions": completed_questions,
                "errored_topics": errored_topics,
            }
        )


def test_firestore_course_update_only_on_completion(tmp_path) -> None:
    fake_store = _FakeFireStore()
    generator = QuestionGenerator(firestore=fake_store)

    config = QuestionBatchConfig(course_code="EEE 101", cache_dir=tmp_path)
    course = {
        "code": "EEE 101",
        "title": "Signals",
        "department": "EEE",
        "outline": [
            {
                "title": "Topic 1",
                "subtopics": ["Intro", "Advanced"],
            }
        ],
    }

    progress = CourseProgressCache(
        course_code=config.course_code,
        cache_root=config.cache_dir,
        theory_target=config.theory_questions_per_request,
        calc_target=config.calc_questions_per_request,
    )

    # Prepare entries
    progress.touch_subtopic("Topic 1", "Intro")
    progress.touch_subtopic("Topic 1", "Advanced")

    # First subtopic fully completed
    per_request = config.calc_questions_per_request
    progress.mark_request_completed("Topic 1", "Intro", "theory-1", config.theory_questions_per_request)
    progress.mark_request_completed("Topic 1", "Intro", "calculation-1", per_request)
    progress.mark_request_completed("Topic 1", "Intro", "calculation-2", per_request)

    # Second subtopic only theory completed so far
    progress.mark_request_completed("Topic 1", "Advanced", "theory-1", config.theory_questions_per_request)

    # Firestore should not be updated until course fully completes
    generator._finalize_course_progress(
        config=config,
        course=course,
        progress=progress,
    )

    assert not fake_store.calls, "Firestore should not update before course completion"

    # Complete remaining batches for second subtopic and verify status becomes completed
    progress.mark_request_completed("Topic 1", "Advanced", "calculation-1", per_request)
    progress.mark_request_completed("Topic 1", "Advanced", "calculation-2", per_request)

    generator._finalize_course_progress(
        config=config,
        course=course,
        progress=progress,
    )

    assert len(fake_store.calls) == 1
    final = fake_store.calls[0]
    per_subtopic_total = (
        config.theory_questions_per_request + 2 * config.calc_questions_per_request
    )
    assert final["status"] == "completed"
    assert final["total_topics"] == 2
    assert final["completed_topics"] == 2
    assert final["completed_questions"] == per_subtopic_total * 2
    assert final["errored_topics"] == 0
