
from fastapi import FastAPI
from app.routers import chat, documents
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="LLM Chatbot API", description="API for LLM-based chatbot with document processing")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(chat.router)
app.include_router(documents.router)

@app.get("/")
async def root():
    return {"message": "LLM Chatbot API is running"}
