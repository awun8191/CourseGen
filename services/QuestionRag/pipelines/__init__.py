"""Processing pipelines for QuestionRag services."""

from .course_outline_generator import main as outline_main  # noqa: F401

__all__ = [
    "outline_main",
]
