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
async def upload_non_academic_document(
    file: UploadFile = File(...),
    subtype: str = Form(...),
    hierarchy_level: str = Form(...),
    validity: str = Form(None),
    degree_programs: str = Form(...),
    specialization_track: str = Form(None),
    departments_affecting: str = Form(...),
    faculties_affecting: str = Form(...),
    batches_affecting: str = Form(...),
    link: str = Form(None)
):
    """Handle non-academic document upload with metadata"""
    
    print(f"=== NON-ACADEMIC DOCUMENT UPLOAD START ===")
    print(f"File: {file.filename}")
    print(f"Content type: {file.content_type}")
    print(f"Subtype: {subtype}")
    print(f"Hierarchy: {hierarchy_level}")
    print(f"Raw degree_programs: {degree_programs}")
    print(f"Raw departments: {departments_affecting}")
    print(f"Raw faculties: {faculties_affecting}")
    print(f"Raw batches: {batches_affecting}")
    
    if not file.filename.endswith(('.pdf', '.docx')):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are allowed")

    try:
        # Parse JSON array inputs with better error handling
        print("=== PARSING JSON ARRAYS ===")
        
        def parse_json_array_or_all(json_string: str, field_name: str) -> List[str]:
            """Parse JSON array string, return ['all'] if empty"""
            try:
                if not json_string or json_string.strip() == "":
                    print(f"{field_name}: Empty string, returning ['all']")
                    return ["all"]
                
                parsed = json.loads(json_string)
                print(f"{field_name}: Parsed JSON = {parsed}")
                
                if not parsed:  # Empty list
                    print(f"{field_name}: Empty list, returning ['all']")
                    return ["all"]
                    
                return parsed
            except json.JSONDecodeError as e:
                print(f"JSON decode error in {field_name}: {e}")
                return ["all"]
            except Exception as e:
                print(f"Unexpected error parsing {field_name}: {e}")
                return ["all"]

        degrees = parse_json_array_or_all(degree_programs, "degree_programs")
        departments = parse_json_array_or_all(departments_affecting, "departments_affecting")
        faculties = parse_json_array_or_all(faculties_affecting, "faculties_affecting")
        batches = parse_json_array_or_all(batches_affecting, "batches_affecting")
        
        print(f"Final parsed arrays:")
        print(f"  degrees: {degrees}")
        print(f"  departments: {departments}")
        print(f"  faculties: {faculties}")
        print(f"  batches: {batches}")

        # Extract and process document
        print("=== DOCUMENT EXTRACTION ===")
        try:
            if file.filename.endswith('.pdf'):
                extractor = PDFExtractor(file)
            else:
                extractor = DocxExtractor(file)
            
            text_content = extractor.extract_text()
            print(f"Extracted text length: {len(text_content)}")
            
            if not text_content or len(text_content.strip()) < 10:
                raise ValueError("Document contains insufficient text content")
                
        except Exception as e:
            print(f"Document extraction error: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to extract text from document: {str(e)}")
        
        # Preprocess text
        print("=== TEXT PREPROCESSING ===")
        try:
            preprocessor = TextPreprocessor()
            cleaned_text = preprocessor.clean_text(text_content)
            processed_text = preprocessor.nlp_process(cleaned_text)
            smart_chunks = preprocessor.smart_chunk_text(processed_text)
            regular_chunks = preprocessor.chunk_text(processed_text)
            chunks = smart_chunks if smart_chunks else regular_chunks
            
            if not chunks:
                # Fallback: create a single chunk from the processed text
                print("No chunks generated, creating single chunk fallback")
                chunks = [processed_text] if processed_text.strip() else ["No content available"]
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=f"Failed to process document text: {str(e)}")
        
        # Generate embeddings
        try:
            embedder = OP()
            embeddings = embedder.embed_docs(chunks)
            if not embeddings:
                raise ValueError("No embeddings generated")
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {str(e)}")
        
        # Create metadata for each chunk
        try:
            chunk_metadatas = [
                {
                    # Document identification
                    "document_name": file.filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_type": "pdf" if file.filename.endswith('.pdf') else "docx",
                    
                    # Document classification
                    "type": "non-academic",
                    "subtype": subtype,
                    
                    # Non-academic specific
                    "hierarchy_level": hierarchy_level,
                    "validity": validity if validity else "",  # Changed: None → ""
                    
                    # Program and department info (convert arrays to comma-separated strings)
                    "degree_programs": ",".join(degrees),
                    "specialization_track": specialization_track if specialization_track else "",  # Changed: None → ""
                    "departments_affecting": ",".join(departments),
                    "faculties_affecting": ",".join(faculties),
                    "batches_affecting": ",".join(batches),
                    
                    # Reference
                    "link": link if link else "",  # Changed: None → ""
                    "upload_date": datetime.now().isoformat()
                } for i in range(len(chunks))
            ]

            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create metadata: {str(e)}")
        
        # Store in database
        try:
            database = ChromaDB()
            database.store_vectors(embeddings, chunks, metadatas=chunk_metadatas)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to store in database: {str(e)}")

        return DocumentUploadResponse(
            success=True,
            documentName=file.filename
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON format in array fields: {str(e)}")
    except Exception as e:
        print(f"Unexpected error in upload_non_academic_document: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
            
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

