from pydantic import BaseModel
from typing import List, Dict, Optional

from .course_outline import CourseOutline


class CourseModel(BaseModel):
    code: str
    department: List[str]
    department_code: Optional[str] = None
    description: Optional[str] = None
    is_elective: Optional[bool] = None
    level: Optional[str] = None
    semester: Optional[str] = None
    title: str
    topics: Optional[List[str]] = None
    type: str
    units: int


class CourseData(BaseModel):
    courses: List[CourseModel]
