import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_models.question_model import Question
from services.QuestionRag.pipelines.config import QuestionBatchConfig, RequestPlan
from services.QuestionRag.pipelines.models import GeminiGeneratedQuestion, GeminiQuestionBatch
from services.QuestionRag.pipelines.question_generator import QuestionGenerator
from services.QuestionRag.pipelines.worker_pool import TopicWorkerPool
from services.QuestionRag.utils import CourseProgressCache


def _make_question(topic: str, subtopic: str, idx: int) -> Question:
    return Question(
        course_code="TEST 101",
        course_name="Test Course",
        topic_name=topic,
        subtopic_name=subtopic,
        level=None,
        semester=None,
        question_type="theory",
        difficulty_ranking=2,
        difficulty="Easy",
        question=f"Sample question {idx} for {subtopic}?",
        options=["A", "B", "C", "D"],
        correct_answer_index=0,
        correct_answer="A",
        correct_answer_text="Option A",
        explanation="Because.",
        solution_steps=[],
        rag_sources=[],
        extra_metadata={},
    )


def test_topic_worker_pool_returns_question_payloads(tmp_path):
    course = {
        "code": "TEST 101",
        "title": "Test Course",
        "outline": [
            {"title": "Topic 1", "subtopics": ["S1", "S2"]},
            {"title": "Topic 2", "subtopics": ["S3"]},
        ],
    }

    config = QuestionBatchConfig(
        course_code="TEST 101",
        enable_topic_parallelism_override=True,
        max_topic_workers_override=2,
        worker_timeout_override=60,
        worker_retry_attempts_override=0,
    )

    progress = CourseProgressCache(
        course_code="TEST 101",
        cache_root=tmp_path,
        theory_target=10,
        calc_target=5,
    )

    def generator_func(course, topic_title, subtopics, config, progress_cache):
        questions = []
        for idx, subtopic in enumerate(subtopics, start=1):
            questions.append(_make_question(topic_title, subtopic, idx))
        return questions

    pool = TopicWorkerPool(max_workers=2, timeout=30, retry_attempts=0)

    results = pool.process_topics_parallel(
        course=course,
        topics=course["outline"],
        generator_func=generator_func,
        config=config,
        progress_cache=progress,
    )

    assert len(results) == 2
    assert all(result.success for result in results)
    assert sum(result.questions_generated for result in results) == 3
    assert sum(len(result.questions) for result in results) == 3
    assert {question.topic_name for result in results for question in result.questions} == {"Topic 1", "Topic 2"}


def test_call_gemini_trims_extra_questions(monkeypatch, tmp_path):
    extra_question = GeminiGeneratedQuestion(
        question="Q3?",
        options=["A", "B", "C", "D"],
        correct_answer_indexes=[0],
        explanation="Because",
    )
    batch = GeminiQuestionBatch(
        questions=[
            GeminiGeneratedQuestion(
                question="Q1?",
                options=["A", "B", "C", "D"],
                correct_answer_indexes=[0],
                explanation="Because",
            ),
            GeminiGeneratedQuestion(
                question="Q2?",
                options=["A", "B", "C", "D"],
                correct_answer_indexes=[0],
                explanation="Because",
            ),
            extra_question,
        ]
    )

    class StubGemini:
        def generate(self, *args, **kwargs):
            return batch

    generator = QuestionGenerator(gemini_service=StubGemini(), use_structured=False)

    config = QuestionBatchConfig(
        course_code="TEST",
        cache_dir=tmp_path,
    )

    course = {"code": "TEST", "title": "Test"}
    request = RequestPlan(name="theory-1", kind="theory", question_count=2, difficulty_rank=2)
    rag_sources = [{"id": "doc-1", "score": 1.0, "snippet": "context"}]

    questions = generator._call_gemini(
        config=config,
        course=course,
        topic_title="Topic",
        subtopic_title="Subtopic",
        request=request,
        context_text="context",
        rag_sources=rag_sources,
    )

    assert len(questions) == 2
    assert all(q.question.endswith("?") for q in questions)


def test_generate_topic_questions_worker_respects_filters(monkeypatch, tmp_path):
    generator = QuestionGenerator(gemini_service=None, use_structured=False)

    config = QuestionBatchConfig(
        course_code="TEST 101",
        enable_topic_parallelism_override=True,
        target_subtopics=("Include",),
    )

    progress = CourseProgressCache(
        course_code="TEST 101",
        cache_root=tmp_path,
        theory_target=10,
        calc_target=5,
    )

    processed = []

    def fake_generate_for_subtopic(self, *, config, course, topic_title, subtopic_title, progress):
        processed.append(subtopic_title)
        return [_make_question(topic_title, subtopic_title, len(processed))]

    monkeypatch.setattr(
        QuestionGenerator,
        "_generate_for_subtopic",
        fake_generate_for_subtopic,
    )

    course = {"code": "TEST 101", "title": "Test Course"}
    subtopics = ["Include", "Skip"]

    questions = generator._generate_topic_questions_worker(
        course=course,
        topic_title="Topic",
        subtopics=subtopics,
        config=config,
        progress_cache=progress,
    )

    assert processed == ["Include"]
    assert len(questions) == 1
    assert questions[0].subtopic_name == "Include"


def test_generate_course_questions_delegates_to_parallel(monkeypatch):
    generator = QuestionGenerator(gemini_service=None, use_structured=False)
    config = QuestionBatchConfig(
        course_code="TEST 101",
        enable_topic_parallelism_override=True,
    )

    called = {}

    def fake_parallel(self, cfg):
        called["invoked"] = cfg.course_code
        return []

    monkeypatch.setattr(
        QuestionGenerator,
        "generate_course_questions_parallel",
        fake_parallel,
    )

    result = generator.generate_course_questions(config)

    assert result == []
    assert called == {"invoked": "TEST 101"}
