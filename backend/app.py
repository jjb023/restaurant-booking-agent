"""FastAPI application for the booking agent."""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import os
import logging
from dotenv import load_dotenv
import uuid
from booking_client import BookingAPIClient
from agent import BookingAgent

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Restaurant Booking Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the booking client and agent
api_client = BookingAPIClient(
    base_url=os.getenv("BOOKING_API_URL", "http://localhost:8547"),
    bearer_token=os.getenv("BOOKING_API_TOKEN", "test_token"),
    restaurant_name="TheHungryUnicorn"
)

booking_agent = BookingAgent(api_client)

sessions = {}


class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Process a chat message and return the agent's response."""
    try:
        session_id = message.session_id or str(uuid.uuid4())
        
        if session_id not in sessions:
            sessions[session_id] = BookingAgent(api_client)
        
        agent = sessions[session_id]
        
        response = agent.process_message(message.message, session_id)
        
        return ChatResponse(response=response, session_id=session_id)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset/{session_id}")
async def reset_session(session_id: str):
    """Reset a conversation session."""
    if session_id in sessions:
        sessions[session_id].clear_memory(session_id)
        return {"message": "Session reset successfully"}
    return {"message": "Session not found"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)