"""Data models for question generation pipeline."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GeminiGeneratedQuestion(BaseModel):
    """Schema describing the expected Gemini JSON payload."""

    question: str = Field(..., description="Main question text")
    options: List[str] = Field(
        ...,
        min_length=4,
        max_length=4,
    )
    correct_answer_indexes: List[int] = Field(
        ..., description="Zero-based indices of correct options (single element)", min_length=1, max_length=4
    )
    correct_answer: Optional[str] = Field(
        default=None, description="Legacy correct option letter (A-D)"
    )
    correct_answer_text: Optional[str] = Field(
        default=None, description="Correct option text"
    )
    explanation: str = Field(..., description="Grounded explanation")
    solution_steps: Optional[List[str]] = Field(
        default=None,
        description="Ordered list of solution steps for calculations",
    )


class GeminiQuestionBatch(BaseModel):
    """Batch of questions returned from Gemini."""

    questions: List[GeminiGeneratedQuestion] = Field(
        ...,
        min_length=1,
        description="Collection of questions returned from Gemini",
    )
