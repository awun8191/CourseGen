"""Thread-safe worker pool for parallel question generation."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

from data_models.question_model import Question

from .config import QuestionBatchConfig
from ..utils import CourseProgressCache

logger = logging.getLogger(__name__)


@dataclass
class WorkerResult:
    """Result from a worker thread."""

    topic_title: str
    success: bool
    questions_generated: int
    questions: List[Question] = field(default_factory=list)
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class WorkerTask:
    """Task to be executed by a worker."""
    topic_title: str
    subtopics: List[str]
    task_id: str


class TopicWorkerPool:
    """Thread pool for processing multiple topics in parallel."""

    def __init__(
        self,
        max_workers: int = 3,
        timeout: int = 300,
        retry_attempts: int = 2,
    ):
        self.max_workers = max_workers
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self._lock = threading.Lock()
        self._active_tasks: Dict[str, Future] = {}

    def process_topics_parallel(
        self,
        course: Dict[str, Any],
        topics: List[Dict[str, Any]],
        generator_func: Callable[
            [Dict[str, Any], str, List[str], QuestionBatchConfig, CourseProgressCache],
            List[Question],
        ],
        config: QuestionBatchConfig,
        progress_cache: CourseProgressCache,
    ) -> List[WorkerResult]:
        """
        Process multiple topics in parallel using thread pool.

        Args:
            course: Course data dictionary
            topics: List of topic dictionaries to process
            generator_func: Function to generate questions for a topic
            config: Question generation configuration
            progress_cache: Progress tracking cache

        Returns:
            List of WorkerResult objects with processing results
        """
        if not topics:
            return []

        # Filter out topics that don't match the topic filter
        normalized_topics = config.normalized_topics()
        filtered_topics = []
        for topic in topics:
            topic_title = str(topic.get("title", "")).strip()
            if normalized_topics and topic_title.lower() not in normalized_topics:
                logger.debug("Skipping topic '%s' not in filter", topic_title)
                continue
            filtered_topics.append(topic)

        if not filtered_topics:
            logger.warning("No topics match the filter criteria")
            return []

        logger.info(
            "Processing %d topics with %d workers for course %s",
            len(filtered_topics),
            min(self.max_workers, len(filtered_topics)),
            course.get("code", "unknown")
        )

        # Create tasks for each topic
        tasks = []
        for topic in filtered_topics:
            topic_title = str(topic.get("title", "")).strip()
            subtopics = topic.get("subtopics", [])
            if not subtopics:
                continue

            task = WorkerTask(
                topic_title=topic_title,
                subtopics=[str(s).strip() for s in subtopics if str(s).strip()],
                task_id=f"{course.get('code', 'unknown')}_{topic_title}"
            )
            tasks.append(task)

        if not tasks:
            logger.warning("No valid topics with subtopics found")
            return []

        results = []
        failed_tasks = []

        # Process tasks with thread pool
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(tasks))) as executor:
            # Submit all tasks
            future_to_task = {}
            for task in tasks:
                future = executor.submit(
                    self._process_topic_worker,
                    task,
                    course,
                    generator_func,
                    config,
                    progress_cache
                )
                future_to_task[future] = task
                with self._lock:
                    self._active_tasks[task.task_id] = future

            # Collect results as they complete
            for future in as_completed(future_to_task, timeout=self.timeout):
                task = future_to_task[future]
                with self._lock:
                    self._active_tasks.pop(task.task_id, None)

                try:
                    result = future.result(timeout=30)
                    results.append(result)

                    if result.success:
                        logger.info(
                            "✓ Topic '%s' completed: %d questions in %.1fs",
                            result.topic_title,
                            result.questions_generated,
                            result.execution_time
                        )
                    else:
                        logger.error(
                            "✗ Topic '%s' failed: %s",
                            result.topic_title,
                            result.error or "Unknown error"
                        )
                        failed_tasks.append(task)

                except Exception as exc:
                    error_msg = f"Worker exception: {exc}"
                    logger.error("Worker failed for topic '%s': %s", task.topic_title, error_msg)

                    result = WorkerResult(
                        topic_title=task.topic_title,
                        success=False,
                        questions_generated=0,
                        error=error_msg,
                        execution_time=0.0
                    )
                    results.append(result)
                    failed_tasks.append(task)

        # Retry failed tasks if retry attempts remaining
        for attempt in range(self.retry_attempts):
            if not failed_tasks:
                break

            logger.info("Retrying %d failed tasks (attempt %d/%d)",
                       len(failed_tasks), attempt + 1, self.retry_attempts)

            retry_tasks = failed_tasks.copy()
            failed_tasks = []

            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(retry_tasks))) as executor:
                future_to_task = {}
                for task in retry_tasks:
                    future = executor.submit(
                        self._process_topic_worker,
                        task,
                        course,
                        generator_func,
                        config,
                        progress_cache
                    )
                    future_to_task[future] = task
                    with self._lock:
                        self._active_tasks[task.task_id] = future

                for future in as_completed(future_to_task, timeout=self.timeout):
                    task = future_to_task[future]
                    with self._lock:
                        self._active_tasks.pop(task.task_id, None)

                    try:
                        result = future.result(timeout=30)
                        results.append(result)

                        if result.success:
                            logger.info(
                                "✓ Retry successful for topic '%s': %d questions in %.1fs",
                                result.topic_title,
                                result.questions_generated,
                                result.execution_time
                            )
                        else:
                            logger.warning(
                                "✗ Retry failed for topic '%s': %s",
                                result.topic_title,
                                result.error or "Unknown error"
                            )
                            failed_tasks.append(task)

                    except Exception as exc:
                        error_msg = f"Retry worker exception: {exc}"
                        logger.error("Retry worker failed for topic '%s': %s", task.topic_title, error_msg)

                        result = WorkerResult(
                            topic_title=task.topic_title,
                            success=False,
                            questions_generated=0,
                            error=error_msg,
                            execution_time=0.0
                        )
                        results.append(result)
                        failed_tasks.append(task)

        # Log final summary
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        logger.info(
            "Topic processing complete: %d successful, %d failed, %d total questions",
            len(successful),
            len(failed),
            sum(r.questions_generated for r in successful)
        )

        return results

    def _process_topic_worker(
        self,
        task: WorkerTask,
        course: Dict[str, Any],
        generator_func: Callable,
        config: QuestionBatchConfig,
        progress_cache: CourseProgressCache,
    ) -> WorkerResult:
        """Worker function to process a single topic."""
        start_time = time.time()

        try:
            # Call the generator function for this topic
            questions = generator_func(
                course=course,
                topic_title=task.topic_title,
                subtopics=task.subtopics,
                config=config,
                progress_cache=progress_cache,
            ) or []

            execution_time = time.time() - start_time
            return WorkerResult(
                topic_title=task.topic_title,
                success=True,
                questions_generated=len(questions),
                questions=questions,
                execution_time=execution_time,
            )

        except Exception as exc:
            execution_time = time.time() - start_time
            error_msg = f"Topic processing failed: {exc}"
            logger.error("Failed to process topic '%s': %s", task.topic_title, error_msg)

            return WorkerResult(
                topic_title=task.topic_title,
                success=False,
                questions_generated=0,
                questions=[],
                error=error_msg,
                execution_time=execution_time,
            )

    def get_active_task_count(self) -> int:
        """Get the number of currently active tasks."""
        with self._lock:
            return len(self._active_tasks)

    def shutdown(self, wait: bool = True):
        """Shutdown the worker pool."""
        # The ThreadPoolExecutor context manager handles shutdown
        pass
