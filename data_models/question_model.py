from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Question(BaseModel):
    course_code: str = Field(..., description="The course code, e.g. EEE 201")
    course_name: str = Field(..., description="Human readable course title")
    topic_name: str = Field(..., description="Topic title within the outline")
    subtopic_name: str = Field(..., description="Specific subtopic that the question targets")
    level: Optional[str] = Field(None, description="Course level, e.g. 200")
    semester: Optional[str] = Field(None, description="Semester in which the course runs")
    question_type: Literal["theory", "calculation"] = Field(
        ..., description="Whether this question is conceptual or calculation based"
    )
    difficulty_ranking: int = Field(
        ..., description="Integer 1-10 quantifying difficulty", ge=1, le=10
    )
    difficulty: Literal["Easy", "Medium", "Hard"] = Field(
        ..., description="Difficulty bucket derived from ranking"
    )
    question: str = Field(..., description="Main question text")
    options: List[str] = Field(
        ..., description="Exactly four multiple choice options", min_length=4, max_length=4
    )
    correct_answer_index: Optional[int] = Field(
        None,
        description="Zero-based index of the correct option",
        ge=0,
        le=3,
    )
    correct_answer: Literal["A", "B", "C", "D"] = Field(
        ..., description="The correct option letter (A, B, C, or D)"
    )
    correct_answer_text: str = Field(
        ..., description="The option text that corresponds to the correct answer"
    )
    explanation: str = Field(
        ..., description="Explanation of the correct answer grounded in RAG context"
    )
    solution_steps: List[str] = Field(
        default_factory=list,
        description="For calculation questions: ordered list of LaTeX formatted steps",
        max_length=8,
    )
    rag_sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Context snippets used during generation with metadata references",
    )
    extra_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata saved alongside the question",
    )


class QuestionSet(BaseModel):
    questions: List[Question]
