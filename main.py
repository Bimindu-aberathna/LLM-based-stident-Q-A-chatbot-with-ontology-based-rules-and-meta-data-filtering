
from fastapi import FastAPI
from app.routers import chat, documents
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="LLM Chatbot API", description="API for LLM-based chatbot with document processing")

# Include routers
app.include_router(chat.router)
app.include_router(documents.router)

@app.get("/")
async def root():
    return {"message": "LLM Chatbot API is running"}
