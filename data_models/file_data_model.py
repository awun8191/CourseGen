from pydantic import BaseModel


class FileDataModel(BaseModel):
    """Model for file data."""
    
    file_path: str
    course_code: str
    department: str
    level: str
    semester: str
    type: str
    
    


