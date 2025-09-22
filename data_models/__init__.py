"""Aggregate exports for shared data models."""

from .course_catalog import CourseCatalogEntry, DepartmentCatalog
from .course_embedding import EmbeddingRecord

__all__ = [
    "CourseCatalogEntry",
    "DepartmentCatalog",
    "EmbeddingRecord",
]
