from pydantic import BaseModel, Field, RootModel
from typing import List


class OCRItem(BaseModel):
    """Individual OCR item with bounding box and text content."""
    box_2d: List[int] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")
    text_content: str = Field(..., description="Text content extracted from the bounding box")


class OCRResponse(RootModel[List[OCRItem]]):
    """Response format for OCR data extraction."""
    pass