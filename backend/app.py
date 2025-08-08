"""Simple FastAPI app for the booking agent."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os
import uuid
import logging
from booking_client import BookingAPIClient
from agent import BookingAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Restaurant Booking Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize API client and agent
api_client = BookingAPIClient(
    base_url="http://localhost:8547",
    bearer_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFwcGVsbGErYXBpQHJlc2RpYXJ5LmNvbSIsIm5iZiI6MTc1NDQzMDgwNSwiZXhwIjoxNzU0NTE3MjA1LCJpYXQiOjE3NTQ0MzA4MDUsImlzcyI6IlNlbGYiLCJhdWQiOiJodHRwczovL2FwaS5yZXNkaWFyeS5jb20ifQ.g3yLsufdk8Fn2094SB3J3XW-KdBc0DY9a2Jiu_56ud8",
    restaurant_name="TheHungryUnicorn"
)

# Initialize Ollama agent with better configuration
agent = BookingAgent(
    api_client=api_client,
    model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
    temperature=0.3,
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)


class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.post("/chat")
async def chat(msg: ChatMessage) -> ChatResponse:
    """Process chat message."""
    session_id = msg.session_id or str(uuid.uuid4())
    
    try:
        response = agent.process_message(msg.message, session_id)
        return ChatResponse(response=response, session_id=session_id)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset/{session_id}")
async def reset(session_id: str):
    """Reset session."""
    agent.clear_memory(session_id)
    return {"message": "Session reset"}


@app.get("/")
async def root():
    """Serve frontend."""
    if os.path.exists("../frontend/index.html"):
        return FileResponse("../frontend/index.html")
    return {"message": "Chat API running"}


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)