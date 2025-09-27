import json
from fastapi import APIRouter, HTTPException
from app.abstract_factory.Database.chromadb import ChromaDB
from app.abstract_factory.Database.chromadb import ChromaDB
from app.abstract_factory.Embedder.open_ai_embedder import OpenAIEmbedder
from app.models.chat import ChatRequest, ChatResponse, GeneralKnowledgeResponse, GeneralKnowledgeRequest, StudentQueryRequest, TstStudentQueryResponse, IncompleteQueryResponse
import os
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
    if (re.match(r'^[a-zA-Z0-9\s\?\,"\'\&]+$', request.message) is None):
        raise HTTPException(status_code=400, detail="Message contains invalid characters.")

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

DECISION RULES:
1. "success" - If query is about:
   - Academic subjects, courses, lectures, assignments
   - University policies, procedures, administrative matters
   - Student services, academic programs
   - Research topics related to their field of study
   - Questions about course content, syllabus, curriculum
   
2. "invalid" - Only if query is clearly:
   - Personal advice unrelated to academics
   - Entertainment, sports, non-academic topics
   - Completely unrelated to university context
   
3. "incomplete" - If query is too vague and needs clarification

For "success" cases, rewrite the query to be more specific and searchable, adding context from the user's academic profile when relevant.

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
                embedder = OpenAIEmbedder()
                query_embedding = ""
                try:
                    embedder = OpenAIEmbedder()
                    query_embedding = await embedder.embed_query(rewritten_query)
                    print(f"Successfully embedded query with {len(query_embedding)} dimensions")
                except Exception as embed_error:
                    print(f"Embedding error: {embed_error}")
                    
                    try:
                        query_embedding = embedder.embed_query(rewritten_query)
                        print(f"Embedded with sync call: {len(query_embedding)} dimensions")
                        
                    except Exception as sync_error:
                        print(f"Sync embedding also failed: {sync_error}")
                        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {sync_error}")
                        print(f"Query Embedding: {len(query_embedding)} dimensions")  # Debug print
                
                dbInstance = ChromaDB()
                results = dbInstance.retrieve_similar_with_metadata(query_embedding, request)
                print(f"Retrieved {len(results) if results else 0} documents")  # Debug print
                
                # Format results for response
                if results:
                    context_docs = "\n\n".join(results)
                    answer_message = f"Found {len(results)} relevant documents based on your query:\n\n{context_docs[:1500]}..."
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
