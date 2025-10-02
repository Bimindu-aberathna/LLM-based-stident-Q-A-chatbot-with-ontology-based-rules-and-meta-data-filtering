"""
LLM Service Manager

This module manages LLM interactions for the RAG-based student Q&A system.
It handles prompt generation, context integration, and response processing.
"""

from typing import List, Dict, Any, Optional
from app.models.chat import StudentQueryRequest, TstStudentQueryResponse
from .openai_response_generator import OpenAIResponseGenerator


class LLMServiceManager:
    """
    Manager class for LLM services in the RAG-based student Q&A system.
    
    This class handles:
    - Intelligent prompt generation based on student context
    - Integration of retrieved document chunks
    - Response formatting and validation
    - Academic domain-specific processing
    """
    
    def __init__(self, llm_provider: str = "openai"):
        """
        Initialize the LLM Service Manager.
        
        Args:
            llm_provider (str): The LLM provider to use (default: "openai")
        """
        self.llm_provider = llm_provider
        if llm_provider.lower() == "openai":
            self.llm_generator = OpenAIResponseGenerator()
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    
    def generate_response(
        self, 
        question: str, 
        academic_context: str, 
        non_academic_context: str, 
        studentMetadata: StudentQueryRequest
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive response for the student query with improved prompting.
        
        Args:
            question: The student's question
            academic_context: Academic context from RAG retrieval
            non_academic_context: Non-academic context from RAG retrieval
            studentMetadata: Student metadata and context
            
        Returns:
            Dict containing the formatted response
        """
        
        # Build comprehensive prompt
        prompt = self._build_comprehensive_prompt(
            question, academic_context, non_academic_context, studentMetadata
        )
        
        try:
            # Get response from LLM
            llm_response = self.llm_generator.GenerateResponse(prompt)
            
            if llm_response.get("success", True):
                response_data = llm_response.get("response", llm_response)
                
                # Handle different response formats
                if isinstance(response_data, dict):
                    answer = response_data.get("answer", str(response_data))
                    status = response_data.get("status", "complete")
                else:
                    answer = str(response_data)
                    status = "complete"
                
                return {
                    "success": True,
                    "answer": answer,
                    "status": status,
                    "model": llm_response.get("model", "gpt-3.5-turbo"),
                    "usage": llm_response.get("usage", {}),
                    "confidence": response_data.get("confidence", 0.8) if isinstance(response_data, dict) else 0.8
                }
            else:
                return {
                    "success": False,
                    "answer": "I apologize, but I encountered an error while processing your request. Please try again.",
                    "status": "error",
                    "error": llm_response.get("error", "Unknown error")
                }
                
        except Exception as e:
            return {
                "success": False,
                "answer": "I apologize, but I encountered an unexpected error. Please try again later.",
                "status": "error",
                "error": str(e)
            }
    
    def _build_comprehensive_prompt(
        self, 
        question: str, 
        academic_context: str, 
        non_academic_context: str, 
        student_metadata: StudentQueryRequest
    ) -> str:
        """
        Build a comprehensive, well-structured prompt for the LLM.
        
        Args:
            question: Student's question
            academic_context: Academic context from RAG
            non_academic_context: Non-academic context from RAG
            student_metadata: Student information
            
        Returns:
            str: Formatted prompt
        """
        
        # Format student context
        student_context = self._format_student_context(student_metadata)
        
        prompt = f"""You are a university academic advisor. Answer the student's question using only the provided information.

STUDENT INFO:
{student_context}

ACADEMIC CONTEXT:
{academic_context if academic_context.strip() else "No academic information found."}

NON-ACADEMIC CONTEXT (Priority Ordered):
{non_academic_context if non_academic_context.strip() else "No administrative information found."}

INSTRUCTIONS:
- Use ONLY the information provided in above ACADEMIC CONTEXT & NON-ACADEMIC CONTEXT. If necessary information is missing, respond with "Database do not have sufficient information. Please contact the university administration for further assistance."
- Always respond in JSON format with 'answer' and 'status' fields. 
- Make sure 'answer' is a single string, not a list or array.
- Never make up answers or use external knowledge
- If the answer is not in the provided context, respond with "Database do not have sufficient information. Please contact the university administration for further assistance."
- The non-academic context may show chunks annotated like [SCORE: 112.5]. Higher SCORE means higher priority (ontology relevance + freshness).
- NEVER invent or adjust any SCORE values. Only use scores exactly as shown. If no scores appear, ignore this rule.
- When multiple chunks give conflicting statements about the SAME subject:
  1. Always prefer the chunk with the HIGHER SCORE.
  2. If SCORES are identical, prefer the one explicitly stating it is newer (latest dates / later upload_date wording).
  3. If still tied, mention BOTH and mark the answer as "partial" (status = "partial").
- Do not mix older and newer directives unless tie cannot be resolved.
- If information is missing or unclear, say so.
- Be direct and specific.
- Course codes like "INTE 12553": INTE=department, 1=year, 2=semester, 3=credits.

QUESTION: {question}

Respond in JSON format:
{{"answer": "your response here", "status": "complete/partial/insufficient"}}"""

        return prompt
    
    def _format_student_context(self, student_metadata: StudentQueryRequest) -> str:
        """Format student metadata into readable context."""
        context_lines = [
            f"• **Batch**: {student_metadata.batch}",
            f"• **Department**: {student_metadata.department}",
            f"• **Degree Program**: {student_metadata.degree_program}",
            f"• **Faculty**: {student_metadata.faculty}",
            f"• **Current Academic Year**: {student_metadata.current_year}",
            f"• **Current Semester**: {student_metadata.current_sem}",
            f"• **Specialization**: {student_metadata.specialization}"
        ]
        
        if student_metadata.courses_done:
            context_lines.append(f"• **Completed Courses**: {', '.join(student_metadata.courses_done)}")
        
        if student_metadata.course_codes:
            context_lines.append(f"• **Current Course Codes**: {', '.join(student_metadata.course_codes)}")
        
        return "\n".join(context_lines)
    
    def _format_retrieved_context(self, academic_context: str, non_academic_context: str) -> str:
        """Format retrieved contexts into a structured format."""
        # This method is no longer needed as we handle contexts separately in the new prompt
        return ""