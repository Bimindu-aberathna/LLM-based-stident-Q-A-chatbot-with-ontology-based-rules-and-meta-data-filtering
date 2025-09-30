from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class GeneralKnowledgeRequest(BaseModel):
    question: str

class GeneralKnowledgeResponse(BaseModel):
    model: str
    question: str
    answer: str
    
class StudentQueryRequest(BaseModel):
    message: str
    batch: str
    department: str
    degree_program: str
    faculty: str
    current_year: str
    current_sem: str
    specialization: str
    course_codes: list[str] = []
    courses_done: list[str] = []
    
    
class TstStudentQueryResponse(BaseModel):
    model: str
    question: str
    answer: str
    status: str
    
class IncompleteQueryResponse(BaseModel):
    answer: str
    message: str = "Your query is incomplete. Please provide more details."
    status: str = "incomplete"
    