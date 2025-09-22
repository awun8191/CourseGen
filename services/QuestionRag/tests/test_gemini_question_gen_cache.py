import json
from pathlib import Path

from services.QuestionRag.pipelines import course_outline_generator as outline


def _make_outline():
    topics = []
    for idx in range(8):
        topics.append(
            {
                "title": f"Topic {idx}",
                "subtopics": [f"Sub {idx}-{j}" for j in range(5)],
                "sources": ["S1"],
            }
        )
    return {"description": "Course description", "topics": topics}


def test_outline_cache_forget_clears_state(tmp_path: Path):
    cache = outline.OutlineCache("EEE", cache_dir=tmp_path)
    cache.mark_present("EEE 201")
    cache.mark_missing("EEE 101")

    assert "EEE 201" in cache.present
    assert "EEE 101" in cache.missing

    cache.forget("EEE 201")
    cache.forget("EEE 101")

    assert "EEE 201" not in cache.present
    assert "EEE 101" not in cache.missing


def test_department_runner_rechecks_missing_when_embeddings_exist(monkeypatch, tmp_path: Path):
    courses_path = tmp_path / "courses.json"
    courses_path.write_text(
        json.dumps([
            {
                "code": "EEE 101",
                "title": "Intro Electronics",
                "description": "",
                "outline": None,
                "levels": ["100"],
            }
        ])
    )

    forget_calls = []
    present_marks = []

    class FakeCache:
        def __init__(self, dept_code: str):
            self.dept = dept_code
            self._missing = True

        def is_missing(self, course_code: str, ttl_hours: float | int | None = None) -> bool:
            return self._missing

        def mark_missing(self, course_code: str):
            self._missing = True

        def mark_present(self, course_code: str):
            present_marks.append(course_code)
            self._missing = False

        def forget(self, course_code: str):
            forget_calls.append(course_code)
            self._missing = False

        def save(self):
            pass

    class FakeProgress:
        def __init__(self, dept_code: str):
            self.records: dict[str, dict] = {}

        def update(self, course_code: str, **fields):
            self.records[course_code] = fields

        def save(self):
            pass

    class FakeGenerator:
        def __init__(self, is_thinking: bool = False):
            self.is_thinking = is_thinking
            self.calls: list[str] = []

        def generate_outline_for_course(self, **kwargs):
            self.calls.append(kwargs["course_code"])
            return _make_outline()

    monkeypatch.setattr(outline, "OutlineCache", FakeCache)
    monkeypatch.setattr(outline, "OutlineProgress", FakeProgress)
    monkeypatch.setattr(outline, "GeminiQuestionGen", FakeGenerator)
    monkeypatch.setattr(outline.DepartmentRunner, "_course_has_embeddings", lambda self, dept, course: True)

    runner = outline.DepartmentRunner(courses_json=courses_path, is_thinking=False)
    runner.build_outlines_for_department(
        "EEE 101",
        skip_existing=False,
        save_each_write=False,
    )

    assert forget_calls == ["EEE 101"]
    assert present_marks == ["EEE 101"]

    data = json.loads(courses_path.read_text())
    assert data[0]["description"] == "Course description"
    assert len(data[0]["outline"]) == 8
