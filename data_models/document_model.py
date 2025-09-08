from pydantic import Field
from typing import Optional
from datetime import datetime
from .file_data_model import FileDataModel

class Document(FileDataModel):
    """Represents an uploaded document along with its metadata."""

    file_name: str = Field(..., description="The name of the uploaded PDF file.")
    cloudflare_url: str = Field(..., description="The URL of the file stored in Cloudflare R2.")
    upload_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="The timestamp of when the file was uploaded."
    )
    title: Optional[str] = Field(None, description="The title of the course.")
    description: Optional[str] = Field(None, description="A brief description of the course.")
