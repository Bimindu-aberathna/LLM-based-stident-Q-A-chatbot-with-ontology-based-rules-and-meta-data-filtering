from fastapi import APIRouter
from app.models.chat import ChatRequest, ChatResponse

router = APIRouter()

@router.post("/send", response_model=ChatResponse)
def send_message(request: ChatRequest):
    # Dummy response for now
    return ChatResponse(response=f"Echo: {request.message}")
