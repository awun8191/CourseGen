"""Test script for the worker pool implementation."""

import time
import logging
from typing import Dict, List, Any
from .config import QuestionBatchConfig
from .worker_pool import TopicWorkerPool
from ..utils import CourseProgressCache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mock_generator_func(
    course: Dict[str, Any],
    topic_title: str,
    subtopics: List[str],
    config: QuestionBatchConfig,
    progress_cache: CourseProgressCache,
) -> List[str]:
    """Mock generator function for testing."""
    logger.info("Mock processing topic '%s' with %d subtopics", topic_title, len(subtopics))

    # Simulate some processing time
    time.sleep(0.1)

    # Return mock questions
    return [f"Question {i} for {topic_title}" for i in range(5)]


def test_worker_pool():
    """Test the worker pool functionality."""
    logger.info("Testing worker pool implementation...")

    # Create a mock course with multiple topics
    course = {
        "code": "TEST 101",
        "title": "Test Course",
        "outline": [
            {
                "title": "Topic 1",
                "subtopics": ["Subtopic 1.1", "Subtopic 1.2", "Subtopic 1.3"]
            },
            {
                "title": "Topic 2",
                "subtopics": ["Subtopic 2.1", "Subtopic 2.2"]
            },
            {
                "title": "Topic 3",
                "subtopics": ["Subtopic 3.1", "Subtopic 3.2", "Subtopic 3.3", "Subtopic 3.4"]
            },
            {
                "title": "Topic 4",
                "subtopics": ["Subtopic 4.1"]
            }
        ]
    }

    # Create a temporary progress cache
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp_dir:
        progress_cache = CourseProgressCache(
            course_code="TEST 101",
            cache_root=Path(temp_dir),
            theory_target=10,
            calc_target=5,
        )

        # Create worker pool with 3 workers
        worker_pool = TopicWorkerPool(
            max_workers=3,
            timeout=60,
            retry_attempts=1,
        )

        # Create a minimal config
        config = QuestionBatchConfig(
            course_code="TEST 101",
            enable_topic_parallelism_override=True,
            max_topic_workers_override=3,
            worker_timeout_override=60,
            worker_retry_attempts_override=1,
        )

        # Test parallel processing
        start_time = time.time()

        results = worker_pool.process_topics_parallel(
            course=course,
            topics=course["outline"],
            generator_func=mock_generator_func,
            config=config,
            progress_cache=progress_cache,
        )

        end_time = time.time()

        # Print results
        logger.info("Processing completed in %.2f seconds", end_time - start_time)
        logger.info("Results summary:")
        for result in results:
            status = "✓" if result.success else "✗"
            logger.info("  %s %s: %d questions, %.2fs",
                       status, result.topic_title, result.questions_generated, result.execution_time)

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        logger.info("Summary: %d successful, %d failed, %d total questions",
                   len(successful), len(failed), sum(r.questions_generated for r in successful))

        # Verify results
        assert len(results) == 4, f"Expected 4 results, got {len(results)}"
        assert len(successful) == 4, f"Expected 4 successful results, got {len(successful)}"
        assert len(failed) == 0, f"Expected 0 failed results, got {len(failed)}"
        assert sum(r.questions_generated for r in successful) == 20, "Expected 20 total questions"

        logger.info("✓ All tests passed!")

        return True


if __name__ == "__main__":
    try:
        test_worker_pool()
        logger.info("✓ Worker pool test completed successfully!")
    except Exception as exc:
        logger.error("✗ Worker pool test failed: %s", exc)
        raise