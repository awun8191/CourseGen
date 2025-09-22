"""Question generation scaffolding separated from course outline logic.

The actual implementation of question batching still needs to be wired in, but
keeping this module distinct from :mod:`course_outline_generator` reflects the
new architecture: outlines are produced first, and any downstream question
pipelines should live here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class QuestionBatchConfig:
    """Configuration holder for question batch generation."""

    course_code: str
    course_title: str
    department: str
    level: Optional[str] = None
    batch_size: int = 50
    calc_questions_per_request: int = 5
    theory_questions_per_request: int = 10


class QuestionGenerator:
    """Placeholder question generator that will call Gemini with RAG prompts."""

    def __init__(self) -> None:
        self._outline_topics: Optional[List[dict]] = None

    def load_outline_topics(self, topics: Iterable[dict]) -> None:
        """Attach outline topics that future batches will rely on."""

        self._outline_topics = list(topics) if topics is not None else None

    def generate_batch(self, config: QuestionBatchConfig) -> List[dict]:
        """Generate a batch of questions (not implemented yet)."""

        raise NotImplementedError(
            "Question generation batches are not implemented; integrate with the "
            "Gemini batch routines when ready."
        )


class QuestionBatchRunner:
    """Coordinator for orchestrating multiple batch requests."""

    def __init__(self, generator: QuestionGenerator) -> None:
        self.generator = generator

    def run(self, config: QuestionBatchConfig) -> List[dict]:
        """Execute the configured batch run using the underlying generator."""

        return self.generator.generate_batch(config)


__all__ = [
    "QuestionBatchConfig",
    "QuestionGenerator",
    "QuestionBatchRunner",
]
