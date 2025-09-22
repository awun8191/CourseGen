from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass
class CourseCatalogEntry:
    """Lightweight course record used during catalog formatting."""

    code: str = ""
    title: str = ""
    level: str = ""
    semester: str = ""
    units: int = 0
    type: str = ""
    is_elective: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DepartmentCatalog:
    """Group of catalog entries offered by the same department/program."""

    names: List[str]
    courses: List[CourseCatalogEntry]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "names": list(self.names),
            "courses": [course.to_dict() for course in self.courses],
        }


__all__ = [
    "CourseCatalogEntry",
    "DepartmentCatalog",
]
