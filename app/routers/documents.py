from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List
import json

from app.abstract_factory.Database.chromadb import ChromaDB
from app.models.document import DocumentUploadResponse, UploadAcademicDocument, UploadNonAcademicDocument, ChunkingTestResponse, ClearDBResponse
from app.services.document_service import process_pdf
from app.DocumentPreprocessor.pdf_extractor import PDFExtractor
from app.DocumentPreprocessor.docx_extractor import DocxExtractor
from app.DocumentPreprocessor.text_preprocssor import TextPreprocessor
from app.abstract_factory.Embedder.open_ai_embedder import OpenAIEmbedder as OP

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not (file.filename.endswith('.pdf') or file.filename.endswith('.docx')):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are allowed")
    
    if(file.filename.endswith('.pdf')):
        try:
            pdf_doc = PDFExtractor(file) 
            text_content = pdf_doc.extract_text()
            #Preprocess
            preprocessor = TextPreprocessor()
            text_content = preprocessor.clean_text(text_content)
            text_content = preprocessor.nlp_process(text_content)
            chunks = preprocessor.chunk_text(text_content)
            embedder = OP()
            embeddings = embedder.embed_docs(chunks)
            
            # Create metadata for each chunk
            chunk_metadatas = [
                {
                    "document_name": file.filename,
                    "chunk_index": i,
                    "file_type": "pdf"
                } for i in range(len(chunks))
            ]
            
            database = ChromaDB()  # No document_name needed
            database.store_vectors(embeddings, chunks, metadatas=chunk_metadatas)

            return DocumentUploadResponse(
                success=True,
                documentName=file.filename
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing PDF document: {str(e)}")
    elif(file.filename.endswith('.docx')):
        try:
            docx_doc = DocxExtractor(file) 
            text_content = docx_doc.extract_text()
            #Preprocess
            preprocessor = TextPreprocessor()
            text_content = preprocessor.clean_text(text_content)
            text_content = preprocessor.nlp_process(text_content)
            chunks = preprocessor.chunk_text(text_content)
            embedder = OP()
            embeddings = embedder.embed_docs(chunks)
            database = ChromaDB(document_name=file.filename)
            database.store_vectors(embeddings, chunks)
            return DocumentUploadResponse(
                success=True,
                documentName=file.filename
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing DOCX document: {str(e)}")
   

@router.post("/academic_document", response_model=DocumentUploadResponse)
async def upload_academic_document(
    file: UploadFile = File(...),
    name: str = Form(...),
    type: str = Form(...),
    category: str = Form(...),
    course_code: str = Form(...),
    batches_affecting: str = Form(...),  # Will parse JSON string
    FromYear: int = Form(...),
    FromSemester: int = Form(...),
    degree_programs: str = Form(...),  # Comma-separated string
    specialization_track: str = Form(None),
    departments_affecting: str = Form(...),  # Comma-separated string
    faculties_affecting: str = Form(...),  # Comma-separated string
    link: str = Form(None)
):
    """Handle academic document upload with metadata"""
    
    if not file.filename.endswith(('.pdf', '.docx')):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are allowed")

    try:
        # Parse array inputs
        batches = json.loads(batches_affecting)
        degrees = [d.strip() for d in degree_programs.split(',')]
        departments = [d.strip() for d in departments_affecting.split(',')]
        faculties = [f.strip() for f in faculties_affecting.split(',')]

        # Extract and process document
        extractor = PDFExtractor(file) if file.filename.endswith('.pdf') else DocxExtractor(file)
        text_content = extractor.extract_text()
        
        # Preprocess text
        preprocessor = TextPreprocessor()
        text_content = preprocessor.clean_text(text_content)
        text_content = preprocessor.nlp_process(text_content)
        # chunks = preprocessor.chunk_text(text_content)
        chunks = preprocessor.smart_chunk_text(text_content)
        
        
        # Generate embeddings
        embedder = OP()
        embeddings = embedder.embed_docs(chunks)
        
        # Create metadata for each chunk - converting lists to strings
        chunk_metadatas = [
            {
                # Document identification
                "document_name": file.filename,
                "name": name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "file_type": "pdf" if file.filename.endswith('.pdf') else "docx",
                
                # Document classification
                "type": type,
                "category": category,
                
                # Academic metadata
                "course_code": course_code,
                # Convert lists to comma-separated strings
                "batches_affecting": ",".join(batches),
                "year": FromYear,
                "semester": FromSemester,
                
                # Program and department info
                "degree_programs": ",".join(degrees),
                "specialization_track": specialization_track,
                "departments_affecting": ",".join(departments),
                "faculties_affecting": ",".join(faculties),
                
                # Reference
                "link": link,
                "upload_date": datetime.now().isoformat()
            } for i in range(len(chunks))
        ]
        
        # Store in single collection
        database = ChromaDB()
        database.store_vectors(embeddings, chunks, metadatas=chunk_metadatas)

        return DocumentUploadResponse(
            success=True,
            documentName=file.filename
        )
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in batches_affecting")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.post("/non_academic_document", response_model=DocumentUploadResponse)
async def upload_non_academic_document(request: UploadNonAcademicDocument):
    if not (request.file.filename.endswith('.pdf') or request.file.filename.endswith('.docx')):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are allowed")

    if(request.file.filename.endswith('.pdf')):
        try:
            pdf_doc = PDFExtractor(request.file) 
            text_content = pdf_doc.extract_text()
            #Preprocess
            preprocessor = TextPreprocessor()
            text_content = preprocessor.clean_text(text_content)
            text_content = preprocessor.nlp_process(text_content)
            # chunks = preprocessor.chunk_text(text_content)
            chunks = preprocessor.smart_chunk_text(text_content)
            embedder = OP()
            embeddings = embedder.embed_docs(chunks)
            
            # Create metadata for each chunk
            chunk_metadatas = [
                {
                    # Document identification
                    "document_name": request.file.filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_type": "pdf" if request.file.filename.endswith('.pdf') else "docx",
                    
                    # Document classification
                    "type": "non-academic",
                    "subtype": request.subtype,
                    
                    "hierarchy_level": request.hierarchy_level,
                    "validity": request.validity or None,
                    
                    # Program and department info
                    "degree_programs": list(request.degree_programs),  # List of affected programs
                    "specialization_track": request.specialization_track or None,  # Optional
                    "departments_affecting": list(request.departments_affecting),  # List of departments
                    "faculties_affecting": list(request.faculties_affecting),  # List of faculties
                    
                    # Reference
                    "link": request.link,
                    "upload_date": datetime.now().isoformat()  # Add upload timestamp
                } for i in range(len(chunks))
            ]
            
            database = ChromaDB() 
            database.store_vectors(embeddings, chunks, metadatas=chunk_metadatas)

            return DocumentUploadResponse(
                success=True,
                documentName=request.file.filename
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing PDF document: {str(e)}")
    elif(request.file.filename.endswith('.docx')):
        try:
            docx_doc = DocxExtractor(request.file) 
            text_content = docx_doc.extract_text()
            #Preprocess
            preprocessor = TextPreprocessor()
            text_content = preprocessor.clean_text(text_content)
            text_content = preprocessor.nlp_process(text_content)
            chunks = preprocessor.chunk_text(text_content)
            embedder = OP()
            embeddings = embedder.embed_docs(chunks)
            chunk_metadatas = [
                {
                    # Document identification
                    "document_name": request.file.filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_type": "pdf" if request.file.filename.endswith('.pdf') else "docx",
                    
                    # Document classification
                    "type": "non-academic",
                    "subtype": request.sub_type,
                    
                    "hierarchy_level": request.hierarchy_level,
                    "validity": request.validity or None,
                    
                    # Program and department info
                    "degree_programs": list(request.degree_programs),  # List of affected programs
                    "specialization_track": request.specialization_track or None,  # Optional
                    "departments_affecting": list(request.departments_affecting),  # List of departments
                    "faculties_affecting": list(request.faculties_affecting),  # List of faculties
                    
                    # Reference
                    "link": request.link,
                    "upload_date": datetime.now().isoformat()  # Add upload timestamp
                } for i in range(len(chunks))
            ]
            database = ChromaDB(document_name=request.file.filename)
            database.store_vectors(embeddings, chunks, metadatas=chunk_metadatas)
            return DocumentUploadResponse(
                success=True,
                documentName=request.file.filename
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing DOCX document: {str(e)}")
        
@router.delete("/delete_all", response_model=ClearDBResponse)
async def clear_database():
    """Endpoint to clear all documents from the database."""
    try:
        database = ChromaDB()
        database.clear_collection()
        return ClearDBResponse(
            success=True,
            message="All documents have been successfully deleted from the database."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")