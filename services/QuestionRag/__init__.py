"""QuestionRag service package."""

from . import pipelines, utils  # noqa: F401
from .gemini_question_gen import outline_main  # type: ignore  # re-export legacy entry

__all__ = [
    "pipelines",
    "utils",
    "outline_main",
]
