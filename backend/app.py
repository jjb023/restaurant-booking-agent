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
# Use llama3.2:3b 
agent = BookingAgent(
    api_client=api_client,
    model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),  # Changed to llama3.2:3b
    temperature=0.1,  # Keep low for consistency
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
        logger.error(f"Error processing message: {e}")
        # Provide a more helpful error message
        if "ollama" in str(e).lower():
            error_msg = "Cannot connect to Ollama. Please ensure:\n1. Ollama is installed\n2. Ollama server is running (ollama serve)\n3. Model is pulled (ollama pull llama3.2:3b)"
        else:
            error_msg = f"Error: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/reset/{session_id}")
async def reset(session_id: str):
    """Reset session."""
    agent.clear_memory(session_id)
    return {"message": "Session reset", "session_id": session_id}


@app.get("/")
async def root():
    """Serve frontend."""
    frontend_path = "../frontend/index.html"
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "Chat API running on http://localhost:8000", "docs": "http://localhost:8000/docs"}


@app.get("/health")
async def health():
    """Health check with service status."""
    try:
        # Check if we can reach Ollama
        import requests
        ollama_status = "unknown"
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                ollama_status = f"running ({len(models)} models)"
            else:
                ollama_status = "not responding"
        except:
            ollama_status = "not running"
        
        # Check if we can reach the booking API
        booking_api_status = "unknown"
        try:
            resp = requests.get("http://localhost:8547/docs", timeout=2)
            if resp.status_code == 200:
                booking_api_status = "running"
            else:
                booking_api_status = "not responding"
        except:
            booking_api_status = "not running"
        
        return {
            "status": "healthy",
            "services": {
                "ollama": ollama_status,
                "booking_api": booking_api_status
            }
        }
    except Exception as e:
        return {
            "status": "partial",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🦄 TheHungryUnicorn Booking Agent")
    print("="*60)
    print(f"✅ Starting server on http://localhost:8000")
    print(f"📚 API Documentation: http://localhost:8000/docs")
    print(f"🌐 Web Interface: http://localhost:8000")
    print("\nMake sure these services are running:")
    print("1. Ollama: ollama serve")
    print("2. Booking API: cd ../Restaurant-Booking-Mock-API-Server && python -m app")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)