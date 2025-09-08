from pydantic import BaseModel, Field
from typing import List, Literal


class Question(BaseModel):
    course_code: str = Field(..., description="This is the course code for this question")
    course_name: str = Field(..., description="This is the name of the course")
    topic_name: str = Field(
        ..., description="The name of the topic for the question"
    )
    difficulty_ranking: int = Field(
        ..., description="Integer 1-10 quantifying difficulty"
    )
    difficulty: Literal["Easy", "Medium", "Hard"] = Field(
        ..., description="Difficulty derived from ranking"
    )
    question: str = Field(
        ..., description="The main question text"
    )
    options: List[str] = Field(
        ..., description="Exactly 4 multiple choice options", min_items=4, max_items=4
    )
    correct_answer: Literal["A", "B", "C", "D"] = Field(
        ..., description="The correct option letter (A, B, C, or D)"
    )
    explanation: str = Field(
        ..., description="Why the answer is correct (1-2 sentences)"
    )
    solution_steps: List[str] = Field(
        default_factory=list,
        description="For calculation questions: up to 5 concise steps",
        max_items=5,
    )


class QuestionSet(BaseModel):
    questions: List[Question]
