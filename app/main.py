from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models import ChatRequest, ChatResponse
from app.agent import process_chat

app = FastAPI(title="SHL Assessment Recommendation Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import os

@app.get("/health")
def health():
    key = os.getenv("Groq_API_KEY", os.getenv("GROQ_API_KEY", ""))
    return {
        "status": "ok",
        "has_groq_key": bool(key),
        "groq_key_prefix": key[:7] if key else "none"
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    return process_chat(request.messages)
