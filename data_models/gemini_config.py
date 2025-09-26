from pydantic import BaseModel, Field
from typing import Optional, Type

class GeminiConfig(BaseModel):
    """Configuration for Gemini generation requests."""
    temperature: float = Field(0.15, description="Sampling temperature")  # tighter
    max_output_tokens: int = Field(4096, description="Cap to reduce rambling")
    top_p: Optional[float] = Field(0.6, description="Nucleus sampling p value")  # crisper tails
    top_k: Optional[int] = Field(40, description="Top-k sampling value")         # stabilizes options
    response_schema: Optional[Type[BaseModel]] = Field(
        default=None,
        description="Pydantic model describing the desired JSON response format",
    )
    thinking_budget: Optional[int] = Field(
        default=None,
        description="Thinking budget in tokens for thinking models. 0=DISABLED, -1=AUTOMATIC",
    )
    use_thinking: bool = Field(
        default=False,
        description="Enable thinking mode for supported models",
    )
