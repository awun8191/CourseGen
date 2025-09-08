from dataclasses import dataclass
from typing import List



@dataclass
class QuestionModel:
    question: str
    options: List[str]
    explanation: str
    steps: List[str]
    answer: str
