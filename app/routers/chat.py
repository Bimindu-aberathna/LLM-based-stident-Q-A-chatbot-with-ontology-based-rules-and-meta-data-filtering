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
    if (re.match(r'^[a-zA-Z0-9\s.,!?\'"-]+$', request.message) is None):
        raise HTTPException(status_code=400, detail="Message contains invalid characters. Please use only letters, numbers, spaces, and basic punctuation.")
    
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
   - Medical advice, legal advice
   - Weather, news, sports scores
   
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

#     prompt = f"""You are an AI assistant for University of Kelaniya's academic and administrative support system.
# Decide query status and (if success) produce a HIGH-RECALL rewritten_query optimized for vector similarity.

# USER PROFILE
# Batch: {request.batch}
# Department: {request.department}
# Degree Program: {request.degree_program}
# Faculty: {request.faculty}
# Current Year: {request.current_year}
# Current Semester: {request.current_sem}
# Specialization: {request.specialization}
# Course Codes: {request.course_codes}

# STATUS RULES
# success: University academic / administrative / policy / timetable / exam / lecture / coursework / research / service related info needs or any information needs.
# invalid: Purely personal, entertainment content.
# incomplete: Too vague (e.g. 'tell me more', 'help', '??', or missing object like 'When is it?' after no prior context).

# REWRITING GOAL (only when status=success)
# Produce a single flat search string (no sentences) that:
# 1. Preserves original intent terms.
# 2. Adds high-value synonyms & morphological variants.
# 3. Normalizes domain phrases (exam → examination, timetable → schedule).
# 4. Expands temporal references (e.g. 'December' -> 'December Dec 2025 2024' if month given and year absent; use current/next academic year logic).
# 5. Adds ONLY relevant user context tokens (faculty, department, degree program, year, semester, batch) when they sharpen the search.
# 6. Include course code(s) ONLY if question clearly refers to a specific course (explicit code present OR mentions 'this course', 'INTE', etc.).
# 7. If dress code / attire query → add: attire dress code clothing guidelines presentation formal policy.
# 8. If exam timetable / schedule query → add: exam examination timetable schedule assessment dates times venue paper.
# 9. If result/publication query → add: results grades publication release.
# 10. If assignment / submission query → add: assignment submission deadline due date coursework.
# 11. If policy query → add: policy regulation procedure requirement guideline.
# 12. Remove stop/filler words. Max ~65 tokens.

# DO NOT
# - Invent data.
# - Add hallucinated course codes.
# - Use punctuation beyond spaces (except keep exact course codes).
# - Return anything except strict JSON.

# EXAMPLES
# User: "december exam timetable"
# Rewritten: "december dec 2025 examination exam timetable schedule assessment dates times exam timetable faculty {request.faculty} department {request.department} batch {request.batch}"
# User: "What is the presentation dress code?"
# Rewritten: "presentation dress code attire clothing formal guidelines policy university faculty {request.faculty} department {request.department}"

# USER QUERY: "{request.message}"

# Respond ONLY in JSON:
# {{
#   "status": "success" | "invalid" | "incomplete",
#   "rewritten_query": "expanded high recall search query (empty string if status != success)"
# }}"""

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
                    top_k=5,
                    similarity_threshold=0.3
                )

                print(f"Retrieved - Academic: {len(academic_chunks)}, Non-Academic: {len(non_academic_chunks)}")
                if non_academic_chunks:
                    print(f"Sample0000000000000000000000000000000000000 Non-Academic Chunk Metadata: {"\n\n".join(non_academic_chunks)} ____00000000000000000000000")
                if academic_chunks:
                    print(f"Sample0000000000000000000000000000000000000 Academic Chunk Metadata: {"\n\n".join(academic_chunks)} ____00000000000000000000000")

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
                    
                   
                    
                    print("///////////////////////////////////////////////////////////////////////////////////////////////")   
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
