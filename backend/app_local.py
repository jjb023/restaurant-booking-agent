"""FastAPI application using local LLM."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import logging
import uuid
from booking_client import BookingAPIClient
from agent_local import BookingAgent  # Import the local agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Restaurant Booking Agent (Local LLM)", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the booking client and agent
api_client = BookingAPIClient(
    base_url="http://localhost:8547",
    bearer_token="test_token",
    restaurant_name="TheHungryUnicorn"
)

# Use mistral or llama2 for the local model
booking_agent = BookingAgent(api_client, model_name="mistral")

# Store active sessions
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
        # Create session if not provided
        session_id = message.session_id or str(uuid.uuid4())
        
        # Get or create agent for session
        if session_id not in sessions:
            sessions[session_id] = BookingAgent(api_client, model_name="mistral")
        
        agent = sessions[session_id]
        
        # Process the message
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