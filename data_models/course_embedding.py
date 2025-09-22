from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EmbeddingRecord:
    """Lightweight embedding payload for course-level vectors."""

    course_code: str
    embeddings: List[int]
    semester: str
    level: str
    text: Optional[str]


__all__ = ["EmbeddingRecord"]
