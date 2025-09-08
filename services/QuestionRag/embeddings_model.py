from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class EmbeddingModel:
    course_code: str
    embeddings: List[int]
    semester: str
    level: str
    text: Optional[str]
