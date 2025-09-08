from pydantic import BaseModel, Field


class OCRData(BaseModel):
    """Structured text extracted via Optical Character Recognition.

    This model represents the raw textual output obtained from running OCR on a
    document image. The :attr:`text` field contains the exact text returned by
    the OCR process without any additional metadata or formatting.
    """

    text: str = Field(..., description="Text extracted from a page or document using OCR")

