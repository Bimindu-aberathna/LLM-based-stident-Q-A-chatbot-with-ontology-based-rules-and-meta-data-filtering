from fastapi import UploadFile
from pydantic import BaseModel

class DocumentUploadResponse(BaseModel):
    success: bool
    documentName: str
    
class ChunkingTestResponse(BaseModel):
    success: bool
    documentName: str
    normal_chunks: list[str]
    smart_chunks: list[str]
    
class UploadAcademicDocument(BaseModel):
    file: UploadFile    
    name: str
    type: str
    category: str
    course_code: str
    batches_affecting: list[str]
    FromYear: int
    FromSemester: int
    degree_programs: list[str]
    specialization_track: str | None = None
    departments_affecting: list[str]
    faculties_affecting: list[str]
    link: str | None = None
    
class UploadNonAcademicDocument(BaseModel):
    subtype: str  # Fixed: was 'sub_type'
    hierarchy_level: str
    validity: str | None = None
    batches_affecting: list[str] = []  # Empty list means "applies to all"
    specialization_track: str | None = None
    faculties_affecting: list[str] = []  # Empty list means "applies to all"
    degree_programs: list[str] = []  # Empty list means "applies to all"  
    departments_affecting: list[str] = []  # Empty list means "applies to all"
    link: str | None = None
    
class ClearDBResponse(BaseModel):
    success: bool
    message: str
    
class LongTextResponse(BaseModel):
    detail: str
    
class cleanStringRequest(BaseModel):
    string: str

class cleanStringResponse(BaseModel):
    string: str
    
class ChunkDocumentRequest(BaseModel):
    file: UploadFile
class ChunkDocumentResponse(BaseModel):
    success: bool
    smart_chunks: list[str]