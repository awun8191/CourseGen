from pydantic import BaseModel, Field
from typing import List


class CourseOutline(BaseModel):
    topics: List[str] = Field(..., description="A list of 8 to 10 concise topics")