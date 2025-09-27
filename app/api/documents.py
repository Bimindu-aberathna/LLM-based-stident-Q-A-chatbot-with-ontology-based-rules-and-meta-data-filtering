from fastapi import APIRouter, UploadFile, File
from app.models.document import DocumentUploadResponse

router = APIRouter()

@router.post("/upload", response_model=DocumentUploadResponse)
def upload_document(file: UploadFile = File(...)):
    content = file.file.read().decode("utf-8", errors="ignore")
    return DocumentUploadResponse(filename=file.filename, content=content)
