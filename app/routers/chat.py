import json
from fastapi import APIRouter, HTTPException
from app.abstract_factory.Database.chromadb import ChromaDB
from app.abstract_factory.Embedder.open_ai_embedder import OpenAIEmbedder
from app.models.chat import ChatRequest, ChatResponse, GeneralKnowledgeResponse, GeneralKnowledgeRequest, StudentQueryRequest, TstStudentQueryResponse, IncompleteQueryResponse
import os
from app.services.LLM_Services import LLMServiceManager

from openai import OpenAI
import re

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # Load API key and model from environment variables
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    
    if not api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not set. Configure it in your environment or .env file.")
    
    # Embed the user query using embedding mode
    if not request.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(request.message) > 2000:
        raise HTTPException(status_code=400, detail="Message is too long. Please limit to 2000 characters.")
    if any(ord(ch) < 32 for ch in request.message):
        raise HTTPException(status_code=400, detail="Message contains control characters.")
    
    # Retrieve relevant documents from the vector database
    database = ChromaDB(document_name="example")
    
    # Generate a response using the LLM with the retrieved documents as context

@router.post("/askGKquestion", response_model=GeneralKnowledgeResponse)
async def ask_general_knowledge(request: GeneralKnowledgeRequest):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    
    if not api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not set. Configure it in your environment or .env file.")

    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": request.question},
            ],
        )
        answer = resp.choices[0].message.content
        return GeneralKnowledgeResponse(
            model=model,
            question=request.question,
            answer=answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {e}")
    
    
@router.post("/query", response_model=TstStudentQueryResponse)
async def chat_endpoint(request: StudentQueryRequest):
    # Load API key and model
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not set.")

    # Validate user input
    if not request.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(request.message) > 2000:
        raise HTTPException(status_code=400, detail="Message is too long. Please limit to 2000 characters.")
    # Allow letters, numbers, spaces, and common punctuation for academic questions
    if (re.match(r'^[a-zA-Z0-9\s.,!?\'"&()-]+$', request.message) is None):
        raise HTTPException(status_code=400, detail="Message contains invalid characters. Please use only letters, numbers, spaces, and basic punctuation.")

    # Prompt with strict JSON format
    prompt = f"""You are an AI assistant for University of Kelaniya's academic and administrative support system.

TASK: Analyze the user's query and decide how to proceed.
REWRITING INSTRUCTION: 
- If the query is valid, produce a HIGH-RECALL rewritten_query optimized for vector similarity search.
- Correct Grammatical errors, but do not change meaning.
- Expand only obvious abbreviations according to context and user's courses completed list (e.g. "dept" -> "department", "uni" -> "university")

USER PROFILE:
- Batch: {request.batch}
- Department: {request.department}  
- Degree Program: {request.degree_program}
- Faculty: {request.faculty}
- Current Year: {request.current_year}
- Specialization: {request.specialization}
- Courses Completed: {request.courses_done}
- Course Codes: {request.course_codes}
**Courses done & Course codes are in order. Course Code Structure: "INTE 12523", means 1: first year 2: second semester, 3: nummber of credits


DECISION RULES:
1. "success" - If query is about ANY university-related topic including:
   - Academic subjects, courses, lectures, assignments, exams
   - University policies, procedures, administrative matters
   - Student services, academic programs, registration
   - Research topics related to their field of study
   - Campus facilities, library, IT services
   - Dress code, attire, presentation guidelines
   - Graduation requirements, academic regulations
   - Timetables, schedules, deadlines
   - Faculty information, contact details
   - ANY university student information needs
   
2. "invalid" - ONLY if query is completely unrelated to university life:
   - Dating advice, personal relationships
   - Entertainment recommendations (movies, games)
   - Cooking recipes, travel plans
   - Weather, news, sports scores
   - For any other information need, return "success"
   
3. "incomplete" - If query is too vague (like "help", "what?", "tell me more" without context)

For "success" cases, rewrite the query to be more specific and searchable, adding context from the user's academic profile when relevant.

EXAMPLES:
- "What is the dress code?" → status: "success" (university policy)
- "Tell me about presentation attire" → status: "success" (academic requirement)
- "What movies should I watch?" → status: "invalid" (entertainment)
- "Help me" → status: "incomplete" (too vague)

USER QUERY: "{request.message}"

Respond in JSON format:
{{
  "status": "success" | "invalid" | "incomplete",
  "rewritten_query": "detailed rewritten query for document search"
}}"""


    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    client = OpenAI(api_key=api_key)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        raw_answer = resp.choices[0].message.content.strip()
        print(f"Raw LLM Response: {raw_answer}")  # Debug print

        # Parse JSON response safely
        try:
            parsed = json.loads(raw_answer)
            status = parsed.get("status", "incomplete")
            rewritten_query = parsed.get("rewritten_query", "")
        except json.JSONDecodeError as json_error:
            print(f"JSON Parse Error: {json_error}")
            status = "incomplete"
            rewritten_query = "Failed to parse response"
            
        print(f"Parsed - Status: {status}, Query: {rewritten_query}")  # Debug print
            
        # Handle incomplete status
        if status == "incomplete":
            return TstStudentQueryResponse(
                model=model,
                question=request.message,
                answer="Your query is incomplete. Please provide more details.",
                status=status,
                message="Your query is incomplete. Please provide more details."
            )
            
        # Handle invalid status
        elif status == "invalid":
            return TstStudentQueryResponse(
                model=model,
                question=request.message,
                answer="Your query is invalid. Please ask a relevant academic or administrative question.",
                status=status,
                message="Your query is invalid. Please ask a relevant academic or administrative question related to the University of Kelaniya."
            )
            
        # Handle success status
        elif status == "success":
            try:
                # Embed the rewritten query and retrieve relevant documents from ChromaDB
                try:
                    embedder = OpenAIEmbedder()
                    query_embedding = embedder.embed_query(rewritten_query)  # Remove await - it's likely sync
                    
                    if not query_embedding or len(query_embedding) == 0:
                        raise ValueError("Embedding returned empty vector")
                        
                    print(f"Successfully embedded query with {len(query_embedding)} dimensions")
                    
                except Exception as embed_error:
                    print(f"Embedding failed: {embed_error}")
                    return TstStudentQueryResponse(
                        model=model,
                        question=request.message,
                        answer=f"Failed to process query: {str(embed_error)}",
                        status="incomplete",
                        message="Error generating query embeddings."
                    )
                dbInstance = ChromaDB()
                academic_chunks, non_academic_chunks = dbInstance.retrieve_similar_with_metadata(
                    query_embedding, 
                    request, 
                    top_k=7,
                    similarity_threshold=0.3
                )
                # Now you can pass them separately to LLM
                if academic_chunks or non_academic_chunks:
                    # Create separate contexts
                    academic_context = "\n\n=== ACADEMIC CONTENT ===\n" + "\n\n".join(academic_chunks) if academic_chunks else ""
                    non_academic_context = "\n\n=== NON-ACADEMIC CONTENT ===\n" + "\n\n".join(non_academic_chunks) if non_academic_chunks else ""
                    llm_service_manager = LLMServiceManager()
                    llm_response = llm_service_manager.generate_response(
                        question=request.message,
                        academic_context=academic_context,
                        non_academic_context=non_academic_context,
                        studentMetadata=request
                    )

                    return TstStudentQueryResponse(
                        model=llm_response.get("model", model),
                        question=request.message,
                        answer=llm_response.get("answer", "No answer generated."),
                        status=llm_response.get("status", "incomplete"),
                        message="Query processed successfully." if llm_response.get("success", False) else llm_response.get("error", "Error in LLM response.")
                    )             
                else:
                    answer_message = "No relevant documents found for your query. The database might be empty or no documents match your criteria."
                
                return TstStudentQueryResponse(
                    model=model,
                    question=request.message,
                    answer=answer_message,
                    status=status,
                    message="Query processed successfully."
                )
                
            except Exception as retrieval_error:
                print(f"Retrieval Error: {retrieval_error}")
                return TstStudentQueryResponse(
                    model=model,
                    question=request.message,
                    answer=f"Error retrieving documents: {str(retrieval_error)}",
                    status="incomplete",
                    message="Error occurred during document retrieval."
                )
        
        # Fallback for unexpected status
        else:
            return TstStudentQueryResponse(
                model=model,
                question=request.message,
                answer="Unexpected response status.",
                status="incomplete",
                message="Unexpected response status from query analysis."
            )

    except Exception as e:
        print(f"General Error: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {str(e)}")


@router.post("/baselineQuery", response_model=TstStudentQueryResponse)
async def baseline_chat_endpoint(request: StudentQueryRequest):
    # Load API key and model
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not set.")

    # Validate user input
    if not request.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if len(request.message) > 2000:
        raise HTTPException(status_code=400, detail="Message is too long. Please limit to 2000 characters.")
    # Allow letters, numbers, spaces, and common punctuation for academic questions
    if (re.match(r'^[a-zA-Z0-9\s.,!?\'"&()-]+$', request.message) is None):
        raise HTTPException(status_code=400, detail="Message contains invalid characters. Please use only letters, numbers, spaces, and basic punctuation.")

    # Prompt with strict JSON format
    prompt = f"""You are an AI assistant for University of Kelaniya's academic and administrative support system.

TASK: Analyze the user's query and decide how to proceed.
REWRITING INSTRUCTION: 
- If the query is valid, produce a HIGH-RECALL rewritten_query optimized for vector similarity search.
- Correct Grammatical errors, but do not change meaning.
- Expand only obvious abbreviations according to context and user's courses completed list (e.g. "dept" -> "department", "uni" -> "university")

USER PROFILE:
- Batch: {request.batch}
- Department: {request.department}  
- Degree Program: {request.degree_program}
- Faculty: {request.faculty}
- Current Year: {request.current_year}
- Specialization: {request.specialization}
- Courses Completed: {request.courses_done}
- Course Codes: {request.course_codes}
**Courses done & Course codes are in order. Course Code Structure: "INTE 12523", means 1: first year 2: second semester, 3: nummber of credits


DECISION RULES:
1. "success" - If query is about ANY university-related topic including:
   - Academic subjects, courses, lectures, assignments, exams
   - University policies, procedures, administrative matters
   - Student services, academic programs, registration
   - Research topics related to their field of study
   - Campus facilities, library, IT services
   - Dress code, attire, presentation guidelines
   - Graduation requirements, academic regulations
   - Timetables, schedules, deadlines
   - Faculty information, contact details
   - ANY university student information needs
   
2. "invalid" - ONLY if query is completely unrelated to university life:
   - Dating advice, personal relationships
   - Entertainment recommendations (movies, games)
   - Cooking recipes, travel plans
   - Weather, news, sports scores
   - For any other information need, return "success"
   
3. "incomplete" - If query is too vague (like "help", "what?", "tell me more" without context)

For "success" cases, rewrite the query to be more specific and searchable, adding context from the user's academic profile when relevant.

EXAMPLES:
- "What is the dress code?" → status: "success" (university policy)
- "Tell me about presentation attire" → status: "success" (academic requirement)
- "What movies should I watch?" → status: "invalid" (entertainment)
- "Help me" → status: "incomplete" (too vague)

USER QUERY: "{request.message}"

Respond in JSON format:
{{
  "status": "success" | "invalid" | "incomplete",
  "rewritten_query": "detailed rewritten query for document search"
}}"""


    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    client = OpenAI(api_key=api_key)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        raw_answer = resp.choices[0].message.content.strip()
        print(f"Raw LLM Response: {raw_answer}")  # Debug print

        # Parse JSON response safely
        try:
            parsed = json.loads(raw_answer)
            status = parsed.get("status", "incomplete")
            rewritten_query = parsed.get("rewritten_query", "")
        except json.JSONDecodeError as json_error:
            print(f"JSON Parse Error: {json_error}")
            status = "incomplete"
            rewritten_query = "Failed to parse response"
            
        print(f"Parsed - Status: {status}, Query: {rewritten_query}")  # Debug print
            
        # Handle incomplete status
        if status == "incomplete":
            return TstStudentQueryResponse(
                model=model,
                question=request.message,
                answer="Your query is incomplete. Please provide more details.",
                status=status,
                message="Your query is incomplete. Please provide more details."
            )
            
        # Handle invalid status
        elif status == "invalid":
            return TstStudentQueryResponse(
                model=model,
                question=request.message,
                answer="Your query is invalid. Please ask a relevant academic or administrative question.",
                status=status,
                message="Your query is invalid. Please ask a relevant academic or administrative question related to the University of Kelaniya."
            )
            
        # Handle success status
        elif status == "success":
            try:
                # Embed the rewritten query and retrieve relevant documents from ChromaDB
                try:
                    embedder = OpenAIEmbedder()
                    query_embedding = embedder.embed_query(rewritten_query)  # Remove await - it's likely sync
                    
                    if not query_embedding or len(query_embedding) == 0:
                        raise ValueError("Embedding returned empty vector")
                        
                    print(f"Successfully embedded query with {len(query_embedding)} dimensions")
                    
                except Exception as embed_error:
                    print(f"Embedding failed: {embed_error}")
                    return TstStudentQueryResponse(
                        model=model,
                        question=request.message,
                        answer=f"Failed to process query: {str(embed_error)}",
                        status="incomplete",
                        message="Error generating query embeddings."
                    )
                dbInstance = ChromaDB()
                data_chunks = dbInstance.baseline_retriever(
                    query_embedding, 
                    request, 
                    top_k=7,
                    similarity_threshold=0.3
                )
                # Now you can pass them separately to LLM
                if data_chunks:
                    # Create separate contexts
                    context = "\n\n".join(data_chunks)
                    llm_service_manager = LLMServiceManager()
                    llm_response = llm_service_manager.generate_baseline_response(
                        question=request.message,
                        context=context
                    )

                    return TstStudentQueryResponse(
                        model=llm_response.get("model", model),
                        question=request.message,
                        answer=llm_response.get("answer", "No answer generated."),
                        status=llm_response.get("status", "incomplete"),
                        message="Query processed successfully." if llm_response.get("success", False) else llm_response.get("error", "Error in LLM response.")
                    )             
                else:
                    answer_message = "No relevant documents found for your query. The database might be empty or no documents match your criteria."
                
                return TstStudentQueryResponse(
                    model=model,
                    question=request.message,
                    answer=answer_message,
                    status=status,
                    message="Query processed successfully."
                )
                
            except Exception as retrieval_error:
                print(f"Retrieval Error: {retrieval_error}")
                return TstStudentQueryResponse(
                    model=model,
                    question=request.message,
                    answer=f"Error retrieving documents: {str(retrieval_error)}",
                    status="incomplete",
                    message="Error occurred during document retrieval."
                )
        
        # Fallback for unexpected status
        else:
            return TstStudentQueryResponse(
                model=model,
                question=request.message,
                answer="Unexpected response status.",
                status="incomplete",
                message="Unexpected response status from query analysis."
            )

    except Exception as e:
        print(f"General Error: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI request failed: {str(e)}")
